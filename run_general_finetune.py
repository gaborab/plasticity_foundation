import time, datetime
from pathlib import Path

import torch
import libs.labram.modeling_finetune
import settings
from finetune_dataset import KFoldSplitter
import torch.nn as nn
from timm.models import create_model


from libs.labram import utils
from libs.labram.engine_for_finetuning import train_one_epoch, evaluate
from libs.labram.optim_factory import create_optimizer, LayerDecayValueAssigner
from libs.labram.utils import NativeScalerWithGradNormCount
from timm.utils import ModelEma
import argparse
import os
import json
# Load datasets with 6-fold cross-validation splits


# Create sampler and dataloader for train and val

# Select model to train, initialize using timm.create_model()

# -----------------------------------
# This script can be implemented to run the full fine-tuning pipeline, including loading datasets,
# creating dataloaders, initializing the model, and training/evaluating across folds.
# -----------------------------------



LR = 5e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 5e-2
EPOCHS = 30

WARMUP_EPOCHS = 5
WARMUP_STEPS = -1 # overloads epochs if set to >0
WEIGHT_DECAY_END = 2 * WEIGHT_DECAY # higher end wd helps to lock in learnt features
LAYER_DECAY = 0.9
CLIP_GRAD = None
BATCH_SIZE = 64
UPDATE_FREQ = 1
USE_EMA = False

OPTIMIZER_ARGS = {
    "opt": "adamw",
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "opt_eps": 1e-8,
    "opt_betas": None
}

def get_args():
    parser = argparse.ArgumentParser('LaBraM fine-tuning and evaluation script for EEG classification', add_help=False)
    parser.add_argument('--model_name', type=str)

    return parser.parse_args()

def main(opts):
    """
    This function can be implemented to run the full fine-tuning pipeline, including loading datasets, creating
    dataloaders, initializing the model, and training/evaluating across folds.
    """
    torch.manual_seed(settings.SEED)
    kfold = KFoldSplitter(k_folds=6)
    datasets = kfold.split_by_subject()
    nb_classes = 1  # regression task
    metrics = ["mse", "mae", "r2"]  # evaluation metrics for regression
    ch_names = settings.CHORDER_STANDARD
    # cut off "-REF" and "EEG " from channel names for compatibility with get_input_chans function
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    if opts.model_name is None:
        print("Please specify model name")
        return
    
    output_dir_base = settings.FINETUNE_OUTPUT_DIR / opts.model_name
    output_dir_base.mkdir(parents=True, exist_ok=True)

    i = 1
    # ----------- CROSS-VALIDATION LOOP ------------
    for dataset in datasets:
        print(f"\n=== Fold {i}/{len(datasets)} ===")
        tensors = dataset.get_datasets()
        print(f"  Train subjects: {list(dataset.training_epochs.subjects)}")
        print(f"  Val subjects: {list(dataset.validation_epochs.subjects)}")
        print(f"  Test subjects: {list(dataset.testing_epochs.subjects)}")
        # create output and log directory
        print("Creating output and log directories...")
        output_dir = output_dir_base / str(i)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # INITIALIZING
        # initialize samplers
        print("Initializing dataloaders...")
        train_dataset = tensors["train"]
        val_dataset = tensors["val"]
        test_dataset = tensors["test"]
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
        # initialize dataloaders
        train_loader = torch.utils.data.DataLoader(train_dataset, sampler = train_sampler)
        val_loader = torch.utils.data.DataLoader(val_dataset, sampler = val_sampler)
        test_loader = torch.utils.data.DataLoader(test_dataset, sampler = test_sampler)

        print("Initializing model with regression head...")
        # initialize LaBraM
        model = create_model(
            settings.LABRAM_VERSION,
            pretrained=False, # True?
        )
        input_dim = model.embed_dim
        hidden_dim = input_dim // 2  # Common heuristic
        target_dim = train_dataset.target_dim
        # set model head as an MLP for regression
        model.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, target_dim),
        )
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(dev)
        model.to(device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_layers = model.get_num_layers()
        print(f"Model initialized to {dev}")
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
        criterion = nn.MSELoss()
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
        max_r2 = 0.0
        max_r2_test = 0.0
        # --------- TRAINING LOOP ----------
        print(f"Training for {EPOCHS} epochs")
        for epoch in range(EPOCHS):
            print(f"{epoch+1}/{EPOCHS+1}")
            log_writer.set_step(epoch * num_training_steps_per_epoch * UPDATE_FREQ)

            # TRAINING
            train_stats = train_one_epoch(
                model, criterion, train_loader, optimizer,
                device, epoch, loss_scaler, CLIP_GRAD, model_ema,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=UPDATE_FREQ,
                ch_names=ch_names, is_binary=False
            )
            print(f"saving model to {output_dir}")
            utils.save_model(
                args={"output_dir": output_dir}, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

            val_stats = evaluate(val_loader, model, device, header='Val:', ch_names=ch_names, metrics=metrics,
                                 is_binary=False)
            print(f"R2 of the network on the {len(tensors['val'])} val EEG: {val_stats['r2']:.2f}")
            test_stats = evaluate(test_loader, model, device, header='Test:', ch_names=ch_names, metrics=metrics,
                                  is_binary=False)
            print(f"R2 of the network on the {len(tensors['test'])} test EEG: {test_stats['r2']:.2f}")

            print(f"Logging val and test stats to {log_dir}")
            write_stat_logs(log_writer, val_stats, test_stats, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            log_writer.flush()
            with open(os.path.join(output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if val_stats['r2'] > max_r2:
                max_r2 = val_stats['r2']
                print(f"New best R2 on val: {max_r2:.4f}. Saving model.")
                utils.save_model(
                    args={"output_dir": output_dir}, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="best_val", model_ema=model_ema
                )
            if test_stats['r2'] > max_r2_test:
                max_r2_test = test_stats['r2']
                print(f"New best R2 on test: {max_r2_test:.4f}. Saving model.")
                utils.save_model(
                    args={"output_dir": output_dir}, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch="best_test", model_ema=model_ema
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        i += 1

def write_stat_logs(log_writer, val_stats, test_stats, epoch):
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
    for key, value in test_stats.items():
        if key == 'accuracy':
            log_writer.update(accuracy=value, head="test", step=epoch)
        elif key == 'balanced_accuracy':
            log_writer.update(balanced_accuracy=value, head="test", step=epoch)
        elif key == 'f1_weighted':
            log_writer.update(f1_weighted=value, head="test", step=epoch)
        elif key == 'pr_auc':
            log_writer.update(pr_auc=value, head="test", step=epoch)
        elif key == 'roc_auc':
            log_writer.update(roc_auc=value, head="test", step=epoch)
        elif key == 'cohen_kappa':
            log_writer.update(cohen_kappa=value, head="test", step=epoch)
        elif key == 'loss':
            log_writer.update(loss=value, head="test", step=epoch)
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





