import time, datetime
from functools import partial
from collections import OrderedDict
from pathlib import Path

import torch

import core.plot_data_processing
from core import settings, io
from core import utils as core_utils
from core.architectures import EEGAutoencoder, EEGSpectralDecoder, LocationDecoder
from core.datasets import KFoldSplitter
import torch.nn as nn

from libs.labram import utils
from libs.labram.engine_for_finetuning import train_one_epoch, evaluate, evaluate_classification, reconstruction_step, classification_step
from libs.labram.optim_factory import create_optimizer, LayerDecayValueAssigner
from libs.labram.utils import NativeScalerWithGradNormCount
from timm.utils import ModelEmaV2 as ModelEma
import argparse
import os
import json
import numpy as np

# Load datasets with 6-fold cross-validation splits


# Create sampler and dataloader for train and val

# Select model to train, initialize using timm.create_model()

# -----------------------------------
# This script can be implemented to run the full fine-tuning pipeline, including loading datasets,
# creating dataloaders, initializing the model, and training/evaluating across folds.
# -----------------------------------

CHECKPOINT_PATH = settings.CHECKPOINT_DIR / "labram-base.pth"


WARMUP_EPOCHS = 8
WARMUP_STEPS = -1  # overloads epochs if set to >0
LAYER_DECAY = 0.9 #original 0.9
MIN_LR = 1e-6
CLIP_GRAD = None
BATCH_SIZE = 64
UPDATE_FREQ = 1
USE_EMA = False
N_EVAL_CHANNELS = 3  # evaluation target channels per validation sample with --masked, see core.utils.balanced_eval_channels
SAVE_PLOTS = True




def get_args():
    """Parses the command line arguments of the fine-tuning run."""
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--target_mode', type=str, default='wavelet')
    parser.add_argument('--k_folds', type=int, default=6)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--huber_beta', type=float, default=0.1)
    parser.add_argument('--no_pretrained', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)
    parser.add_argument('--model_desc', type=str, default='')
    parser.add_argument('--masked', action='store_true', default=False,
                        help='Masked-channel reconstruction: hide a target channel and its nearest electrodes, train on the target channel only')
    parser.add_argument('--n_neighbors', type=int, default=settings.N_NEIGHBOURS,
                        help='Nearest electrodes hidden together with the target channel with --masked, 0 hides the target channel only')
    parser.add_argument('--hide_interpolated', action=argparse.BooleanOptionalAction, default=True,
                        help='When --masked: hide every trial\'s interpolated channels from the input as well (same mask token), '
                             'so their spline estimates cannot carry signal of the hidden target. --no-hide_interpolated keeps them as inputs')
    parser.add_argument("--task_classify", action='store_true', default=False,
                        help="Enable classification task instead of autoencoding or spectral decoding. The model will be trained to classify the input EEG epochs into classes instead of reconstructing the input or predicting spectral targets.")
    return parser.parse_args()


def main(opts):
    """
    This function can be implemented to run the full fine-tuning pipeline, including loading datasets, creating
    dataloaders, initializing the model, and training/evaluating across folds.
    """
    if opts.task_classify:
        opts.target_mode = 'classify'  # the splitter loads the inputs only, the labels come from the conditions
    # validation metric of the best checkpoint: balanced accuracy for classification, R2 for reconstruction
    best_metric = 'balanced_accuracy' if opts.task_classify else 'r2'

    torch.manual_seed(settings.SEED)
    kfold = KFoldSplitter(input_data_dir=settings.FINETUNE_INPUT_EPOCHS_DIR, target_data_dir=settings.FINETUNE_WAVELET_DIR, target_mode=opts.target_mode, seed=settings.SEED, k_folds=opts.k_folds, no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
    folds = kfold.split_within_subjects()
    # (freqs, time) masks of the target bins the loss and the target stats use in every patch
    masks = None
    if opts.target_mode == 'wavelet':
        masks = {patch: core_utils.valid_wavelet_bins(settings.FREQS, patch_window, settings.LOSS_WINDOWS[patch], settings.RSFREQ, **settings.WAVELET_PARAMS,
                                                      pool=settings.TARGET_POOL)
                 for patch, patch_window in settings.PATCH_PARAMS['patch_windows'].items()}
        print("Masked-in target bins per patch:", {patch: int(mask.sum()) for patch, mask in masks.items()})
    metrics = ["mse", "mae", "r2"]  # evaluation metrics for autoencoding
    ch_names = settings.CHORDER_STANDARD
    # cut off "-REF" and "EEG " from channel names for compatibility with get_input_chans function
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    if opts.model_name is None:
        print("Please specify model name")
        return

    # masked-channel reconstruction: interpolated channels are never targets, a target is hidden with its cluster
    trial_interpolations = None
    neighbour_table = None
    if opts.masked:
        trial_interpolations = io.load_interpolated_channels(settings.FINETUNE_DIR / "epochs_manifest.csv")
        neighbour_table = core_utils.build_neighbor_mask(n_neighbours=opts.n_neighbors, n_chans=settings.N_CHANNELS)
        print(f"Masked-channel reconstruction: every target channel is hidden with its {opts.n_neighbors} nearest electrodes, "
              f"{N_EVAL_CHANNELS} evaluation channels per validation sample"
              + (", interpolated channels are hidden from the input" if opts.hide_interpolated else ""))
    hide_interpolated = opts.masked and opts.hide_interpolated

    LR = opts.lr
    WEIGHT_DECAY = opts.weight_decay
    WEIGHT_DECAY_END = 2 * WEIGHT_DECAY  # higher end wd helps to lock in learnt features
    EPOCHS = opts.epochs
    USE_PRETRAINED = not opts.no_pretrained

    OPTIMIZER_ARGS = {
        "opt": "adamw",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "opt_eps": 1e-8,
        "opt_betas": None
    }

    run_metadata = {
        "hyperparameters": {
            "target_mode": opts.target_mode,
            "k_folds": opts.k_folds,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "warmup_steps": WARMUP_EPOCHS,
            "warmup_steps_end": WARMUP_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "lr": LR,
            "pretrained": USE_PRETRAINED,
            "loss_windows": settings.LOSS_WINDOWS if opts.target_mode == 'wavelet' else None,
            "masked": opts.masked,
            "n_neighbors": opts.n_neighbors if opts.masked else None,
            "n_eval_channels": N_EVAL_CHANNELS if opts.masked else None,
            "hide_interpolated": hide_interpolated,
            "target_pool": settings.TARGET_POOL if opts.target_mode == 'wavelet' else None,
            "task_classify": opts.task_classify,
            "freq_window": settings.SPECTRAL_DECODER_PARAMS['freq_window'] if opts.target_mode == 'wavelet' else None,
        },
        "description": opts.model_desc
    }

    output_dir_base = settings.FINETUNE_OUTPUT_DIR / opts.model_name
    output_dir_base.mkdir(parents=True, exist_ok=True)
    if USE_PRETRAINED:
        # Handle loading state_dict from pretrained checkpoint
        print("Load ckpt from %s" % CHECKPOINT_PATH)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        checkpoint_model = None
        for model_key in ["model", "module"]:
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if checkpoint_model is not None:
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        checkpoint_model = core_utils.add_prefix_dict_keys(checkpoint_model, prefix='model')

    total_time = 0
    best_scores = []
    best_val_epochs = []
    i = 1
    # ----------- CROSS-VALIDATION LOOP ------------
    for fold in folds:
        print(f"\n=== Fold {i}/{len(folds)} ===")
        print(fold.describe())
        # create output and log directory
        print("Creating output and log directories...")
        output_dir = output_dir_base / str(i)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        if Path(output_dir / "log.txt").exists():
            print(f"WARNING: log.txt already exists in {output_dir}. Deleting old log.txt...")
            Path(output_dir / "log.txt").unlink()

        with open(output_dir / "keys.json", "w") as f:
            json.dump(fold.keys, f)

        print(f"Initializing model with {opts.target_mode} head...")
        # initialize LaBraM
        model = None
        if opts.target_mode == 'wavelet' or opts.target_mode == 'superlet':
            model = EEGSpectralDecoder(**settings.SPECTRAL_DECODER_PARAMS, pretrained=USE_PRETRAINED)
        elif opts.target_mode == 'autoencoder':
            model = EEGAutoencoder(**settings.AUTOENCODER_PARAMS, pretrained=USE_PRETRAINED)
        elif opts.target_mode == 'classify':
            model = LocationDecoder(**settings.LOCATION_DECODER_PARAMS, pretrained=USE_PRETRAINED)
        else:
            raise ValueError(
                f"Invalid target mode: {opts.target_mode}. Must be 'wavelet', 'superlet', 'autoencoder' or 'classify'.")

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(dev)


        if USE_PRETRAINED:
            print("Setting pretrained weights from checkpoint...")
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            for key in all_keys:
                if "relative_position_index" in key:
                    checkpoint_model.pop(key)

            utils.load_state_dict(model, checkpoint_model, prefix='')

        model.to(device)
        fold_neighbour_table = torch.as_tensor(neighbour_table, device=device) if neighbour_table is not None else None

        # the loss of one training batch: classification or (masked-channel) reconstruction
        if opts.task_classify:
            train_step = classification_step
        else:
            train_step = partial(reconstruction_step, neighbour_table=fold_neighbour_table, hide_interpolated=hide_interpolated)

        # INITIALIZING
        # initialize samplers
        print(f"Initializing datasets with {opts.target_mode} target mode...")
        # the targets are normalized with mu and sigma of the pooled masked training targets
        if opts.task_classify:
            train_dataset, val_dataset, test_dataset = fold.initialize_class_datasets(
                epoch_window=settings.EPOCH_WINDOW, patch_window=settings.CLASS_PATCH_WINDOW, fs=settings.RSFREQ,
                condition_classes=settings.CONDITION_CLASS_IDS)
            print(f"Class samples: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        else:
            train_dataset, val_dataset, test_dataset = fold.initialize_datasets(patch_params=settings.PATCH_PARAMS,
                                                                            target_mode=opts.target_mode, masks=masks, interpolations=trial_interpolations,
                                                                            seed=settings.SEED, n_eval_channels=N_EVAL_CHANNELS)
        if fold.target_stats is not None:
            print(f"Target stats: {fold.target_stats}")
            # inference has to denormalize the model outputs with the stats of the fold
            with open(output_dir / "target_stats.json", mode="w") as f:
                json.dump(fold.target_stats.to_dict(), f)

        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        # initialize dataloaders
        print("Initializing dataloaders...")
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
        val_loader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_layers = model.get_num_layers()
        print(f"Model initialized to {dev}, {torch.cuda.get_device_name(0)}")
        # set exponential moving average (EMA) model if enabled
        model_ema = None
        if USE_EMA:
            print("Creating EMA model")
            model_ema = ModelEma(
                model,
                decay=0.9999,
                device='',
                resume='')
            print("Using EMA with decay = %.8f" % 0.9999)
        # initialize optimizer, learning rate and weight decay schedules, and loss function
        print("Initializing parameters and utils...")
        if opts.task_classify:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            criterion = nn.SmoothL1Loss(beta=opts.huber_beta) if not opts.mse else nn.MSELoss()
        total_batch_size = BATCH_SIZE * UPDATE_FREQ
        num_training_steps_per_epoch = len(train_dataset) // total_batch_size
        assigner = LayerDecayValueAssigner(
            list(LAYER_DECAY ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        optimizer = create_optimizer(
            args=argparse.Namespace(**OPTIMIZER_ARGS),
            model=model,
            skip_list=model.no_weight_decay(),
            get_num_layer=assigner.get_layer_id,
            get_layer_scale=assigner.get_scale
        )
        loss_scaler = NativeScalerWithGradNormCount()
        lr_schedule_values = utils.cosine_scheduler(
            LR, MIN_LR, EPOCHS, num_training_steps_per_epoch,
            warmup_epochs=WARMUP_EPOCHS, warmup_steps=WARMUP_STEPS,
        )
        wd_schedule_values = utils.cosine_scheduler(
            WEIGHT_DECAY, WEIGHT_DECAY_END, EPOCHS, num_training_steps_per_epoch)
        # initialize log writer
        log_writer = utils.TensorboardLogger(log_dir=log_dir)
        start_time = time.time()
        best_score = float("-inf")  # the first validation always saves a best checkpoint, even with a negative R2
        best_val_epoch = 0
        # --------- TRAINING LOOP ----------
        print(f"Training for {EPOCHS} epochs")
        for epoch in range(EPOCHS):
            print(f"{epoch + 1}/{EPOCHS}")
            log_writer.set_step(epoch * num_training_steps_per_epoch * UPDATE_FREQ)
            train_dataset.set_epoch(epoch)  # selects every sample's training target channel of this epoch from its permutation
            # TRAINING
            train_stats = train_one_epoch(
                model, criterion, train_loader, optimizer,
                device, epoch, loss_scaler, CLIP_GRAD, model_ema,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=UPDATE_FREQ,
                ch_names=ch_names, step=train_step
            )
            print("train statistics: ", train_stats)
            print(f"Saving model to {output_dir}")
            if epoch % 10 == 0:
                utils.save_model(
                    args=argparse.Namespace(**{"output_dir": output_dir}),
                    model=model,
                    model_without_ddp=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_ema=model_ema
                )
            if epoch % 3 == 0 or epoch == EPOCHS - 1:
                print(f" Evaluating on validation set at epoch {epoch}")
                if opts.task_classify:
                    val_stats = evaluate_classification(val_loader, model, device, criterion, header='Val:', ch_names=ch_names,
                                                        n_classes=settings.N_LOCATION_CLASSES)
                else:
                    val_stats = evaluate(val_loader, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
                                         is_binary=False, criterion=criterion, neighbour_table=fold_neighbour_table,
                                         hide_interpolated=hide_interpolated)
                print(f"{best_metric} of the network on the {len(val_dataset)} val EEG: {val_stats[best_metric]:.3f}")

                print(f"Logging val stats to {log_dir}")
                write_stat_logs(log_writer, val_stats, epoch)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_stats.items()},
                             'epoch': epoch,
                             'n_parameters': n_parameters}

                log_writer.flush()
                with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if val_stats[best_metric] > best_score:
                    best_score = val_stats[best_metric]
                    best_val_epoch = epoch
                    print(f"New best {best_metric} on val: {best_score:.4f}. Saving model.")
                    utils.save_model(
                        args=argparse.Namespace(**{"output_dir": output_dir}),
                        model=model,
                        model_without_ddp=model,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best_val",
                        model_ema=model_ema
                    )

        # save the best validation score of the fold
        stats = {
            "val": best_score,
        }
        best_scores.append(stats)
        best_val_epochs.append(best_val_epoch)

        fold_time = time.time() - start_time
        fold_time_str = str(datetime.timedelta(seconds=int(fold_time)))
        print('Fold training time {}'.format(fold_time_str))
        total_time += fold_time
        i += 1
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("=" * 30)
    print(f"Training finished for {i-1} folds.")
    print("Results:")
    for i, fold_res in enumerate(best_scores):
        print(f"      {i}", fold_res)
    print("")
    print("Average results: ")
    mean_val_score = np.mean([fold_res['val'] for fold_res in best_scores])
    print(f"     validation {best_metric}: ", mean_val_score)

    run_metadata["results"] = {
        f"{best_metric}_per_fold": best_scores,
        f"mean_val_{best_metric}": float(mean_val_score),
        f"max_{best_metric}_val_epochs": best_val_epochs,
    }

    if opts.task_classify:
        run_metadata["model_params"] = {**settings.LOCATION_DECODER_PARAMS, "patch_window": settings.CLASS_PATCH_WINDOW,
                                      "classes": settings.LOCATION_CLASSES, "loss_type": "cross_entropy_label_smoothing_0.1"}
    else:
        run_metadata["model_params"] = settings.SPECTRAL_DECODER_PARAMS if opts.target_mode == 'wavelet' or opts.target_mode == 'superlet' else settings.AUTOENCODER_PARAMS
        run_metadata["model_params"]["loss_type"] = "mse" if opts.mse else f"huber_{opts.huber_beta}"

    print(f"Total training time: {total_time_str}")

    with open(output_dir_base / "metadata.json", mode="w") as f:
        json.dump(run_metadata, f)


def write_stat_logs(log_writer, val_stats, epoch):
    """Writes the known validation metrics of val_stats to the tensorboard log at the given epoch."""
    for key, value in val_stats.items():
        if key == 'accuracy':
            log_writer.update(accuracy=value, head="val", step=epoch)
        elif key == 'balanced_accuracy':
            log_writer.update(balanced_accuracy=value, head="val", step=epoch)
        elif key == 'f1_weighted':
            log_writer.update(f1_weighted=value, head="val", step=epoch)
        elif key == 'pr_auc':
            log_writer.update(pr_auc=value, head="val", step=epoch)
        elif key == 'roc_auc':
            log_writer.update(roc_auc=value, head="val", step=epoch)
        elif key == 'cohen_kappa':
            log_writer.update(cohen_kappa=value, head="val", step=epoch)
        elif key == 'loss':
            log_writer.update(loss=value, head="val", step=epoch)
        elif key == 'mse':
            log_writer.update(mse=value, head="val", step=epoch)
        elif key == 'mae':
            log_writer.update(mae=value, head="val", step=epoch)
        elif key == 'r2':
            log_writer.update(r2=value, head="val", step=epoch)


if __name__ == "__main__":
    print("Running general fine-tuning pipeline...")
    opts = get_args()

    main(opts)
