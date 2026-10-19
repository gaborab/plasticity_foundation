# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
from . import utils as utils
from einops import rearrange
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def train_class_batch(model, samples, target, mask, criterion, ch_names, channel_mask=None):
    """
    MODIFIED: returns the loss over the masked-in target bins and the model outputs of one batch.
    target and mask are flattened to (batch, n_outputs), mask is a boolean tensor of the same shape.
    channel_mask: (batch, n_chans) boolean tensor of the hidden channels (masked-channel reconstruction), None hides nothing.
    """
    outputs = model(samples, ch_names, channel_mask=channel_mask)
    loss = criterion(outputs[mask], target[mask])
    return loss, outputs


def target_channel_masks(masks, target_ch, neighbour_table):
    """
    ADDED: returns (hidden, loss_mask) of masked-channel reconstruction with one target channel per sample.
    masks: (batch, n_chans, ...) boolean validity mask, target_ch: (batch,) target channel indices,
    neighbour_table: (n_chans, n_chans) boolean cluster table (core.utils.build_neighbor_mask), all on the same device.
    hidden is the (batch, n_chans) cluster of every sample's target channel, passed to the model as channel_mask.
    loss_mask keeps the bins of masks of the target channel only, flattened to (batch, n_outputs).
    Without a neighbour_table nothing is hidden (None) and loss_mask is masks flattened.
    """
    if neighbour_table is None:
        # no masking: nothing is hidden and the valid bins of every channel are scored
        return None, masks.flatten(1)

    # 1. hidden channels: the row of every sample's target channel in the cluster table (the target and its nearest electrodes)
    hidden = neighbour_table[target_ch]  # (batch, n_chans)

    # 2. target channel only: True at every sample's target channel, False at all other channels, the hidden neighbours included
    batch_size, n_chans = masks.shape[:2]
    is_target = torch.zeros(batch_size, n_chans, dtype=torch.bool, device=masks.device)
    is_target[torch.arange(batch_size, device=masks.device), target_ch] = True  # (batch, n_chans)

    # 3. scored bins: the valid bins (masks) of the target channel. is_target gets size-1 dimensions for the rest of the
    #    target, (freqs, time) for wavelet targets or (time,) for autoencoder targets, so it broadcasts over them
    trailing_dims = [1] * (masks.dim() - 2)
    loss_mask = masks & is_target.view(batch_size, n_chans, *trailing_dims)  # (batch, n_chans, ...)

    # flattened like the model output and the targets, (batch, n_chans * ...)
    return hidden, loss_mask.flatten(1)


class MaskedErrorSums:
    """
    ADDED: running float64 sums of the scored targets and errors on the device, for the pooled loss, MSE, MAE and R2
    of an evaluation without storing predictions or copying them to the CPU.
    """

    def __init__(self, device):
        # n, sum t, sum t^2, sum (t - o)^2, sum |t - o|, sum of the loss weighted by n
        self.sums = torch.zeros(6, dtype=torch.float64, device=device)

    def update(self, output, target, loss):
        """Adds the scored (1-D) outputs and targets of one forward pass and its mean loss."""
        t, e = target.double(), (target - output).double()
        n = t.numel()
        self.sums[0] += n
        self.sums[1] += t.sum()
        self.sums[2] += (t ** 2).sum()
        self.sums[3] += (e ** 2).sum()
        self.sums[4] += e.abs().sum()
        self.sums[5] += loss.double() * n

    def results(self):
        """Returns {"loss", "mse", "mae", "r2"} pooled over all scored bins, R2 = 1 - SS_res / SS_tot."""
        n, s, s2, se2, sae, sloss = self.sums.tolist()
        return {"loss": sloss / n, "mse": se2 / n, "mae": sae / n, "r2": 1 - se2 / (s2 - s * s / n)}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, neighbour_table=None):
    """
    MODIFIED: trains the model for one epoch on batches of (input, target, metadata, mask, target_ch, eval_chs) from FinetuneDataset.
    neighbour_table: (n_chans, n_chans) boolean cluster table on device for masked-channel reconstruction: every sample's
    target_ch is hidden with its cluster and the loss uses the target channel's masked-in bins only. None trains on all channels.
    The loss, MSE and R2 are logged every print_freq steps only, as reading them from the GPU synchronizes it.
    Returns the averages of the logged stats.
    """
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if update_freq is not None:
        print(f"using gradient accumulation: {update_freq} steps ")

    optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, masks, target_ch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True) / 100
        samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)

        # flatten to shape (batch_size, n_channels*time) or (batch_size, n_channels*n_freqs*time) for spectral targets
        targets = targets.to(device, dtype=torch.float32, non_blocking=True).flatten(1)
        masks = masks.to(device, non_blocking=True)
        # hidden cluster of every sample's target channel, and the loss over the target channel's masked-in bins only
        hidden, loss_mask = target_channel_masks(masks, target_ch.to(device, non_blocking=True), neighbour_table)

        with torch.cuda.amp.autocast():
            loss, output = train_class_batch(
                model, samples, targets, loss_mask, criterion, input_chans, channel_mask=hidden)

        log_step = data_iter_step % print_freq == 0 or step == num_training_steps_per_epoch - 1
        loss_detached = loss.detach()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss = loss / update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

        # MODIFIED: stats are read from the GPU every print_freq steps only, reading them synchronizes the GPU
        if not log_step:
            continue
        loss_value = loss_detached.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss_scale_value = loss_scaler._scaler.get_scale()

        o, t = output.detach().float()[loss_mask], targets[loss_mask]
        mse = torch.nn.functional.mse_loss(o, t).item()
        r2 = (1 - ((t - o) ** 2).sum() / ((t - t.mean()) ** 2).sum()).item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mse=mse)
        metric_logger.update(r2=r2)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.set_step(it)
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(mse=mse, head="loss")
            log_writer.update(r2=r2, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], criterion=torch.nn.SmoothL1Loss(beta=0.1), is_binary=True,
             neighbour_table=None):
    """
    MODIFIED: evaluates the model on batches of (input, target, metadata, mask, target_ch, eval_chs) from FinetuneDataset.
    neighbour_table: (n_chans, n_chans) boolean cluster table on device for masked-channel reconstruction: one forward pass
    per evaluation channel in eval_chs, each with that channel's cluster hidden and scored on that channel only.
    None scores all channels in a single pass. Every bin is used if a batch has no mask.
    The errors are summed on the device (MaskedErrorSums), metrics and is_binary are kept for compatibility.
    Returns {"loss", "mse", "mae", "r2"} over the scored bins of all batches pooled.
    """
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    error_sums = MaskedErrorSums(device)
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        EEG = batch[0]
        target = batch[1]

        EEG = EEG.float().to(device, non_blocking=True) / 100
        EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
        mask = batch[3].to(device, non_blocking=True) if len(batch) > 3 else torch.ones_like(target, dtype=torch.bool, device=device)
        target = target.to(device, dtype=torch.float32, non_blocking=True).flatten(1)

        # one pass per evaluation channel with masking, a single pass over all channels without
        eval_chs = batch[5].to(device, non_blocking=True) if neighbour_table is not None else [None]
        for j in range(eval_chs.shape[1] if neighbour_table is not None else 1):
            target_ch = eval_chs[:, j] if neighbour_table is not None else None
            hidden, loss_mask = target_channel_masks(mask, target_ch, neighbour_table)
            with torch.cuda.amp.autocast():
                output = model(EEG, input_chans=input_chans, channel_mask=hidden)
                loss = criterion(output[loss_mask], target[loss_mask])
            error_sums.update(output.float()[loss_mask], target[loss_mask], loss)

    ret = error_sums.results()
    print(f'* {header} Loss {ret["loss"]:.4f} MSE {ret["mse"]:.4f} MAE {ret["mae"]:.4f} R2 {ret["r2"]:.4f}')
    return ret
