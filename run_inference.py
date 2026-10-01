import argparse
import time
import datetime
import h5py
import numpy as np
import torch
from torch import utils as torch_utils

from core.architectures import EEGAutoencoder, EEGSpectralDecoder
from core.datasets import KFoldSplitter, FinetuneDataset
from core.io import EpochIO, save_dict_to_hdf, save_inference_trial_hdf
from core import settings
from libs.labram import utils
from einops import rearrange
from core import utils as core_utils

def run_reconstruction(model, dataset, ch_names, device, output_file, freqs=None, freq_bands=None):
    sampler = torch_utils.data.SequentialSampler(dataset)
    loader = torch_utils.data.DataLoader(dataset, sampler=sampler, batch_size=256)

    with h5py.File(output_file, "w") as h5file:
        with torch.no_grad():
            input_chans = utils.get_input_chans(ch_names)
            for i, batch in enumerate(loader):
                print(f"Reconstructing trials in batch {i} (batch size: {loader.batch_size})")
                EEG  = batch[0]
                target = batch[1]
                metadata = batch[2] if len(batch) > 2 else None
                subjects = metadata['subject']
                conditions = metadata['condition']
                ids = metadata['id']
                EEG = EEG.float().to(device, non_blocking=True) / 100
                EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
                target = target.to(device, dtype=torch.float32, non_blocking=True)

                with torch.cuda.amp.autocast():
                    output = model.get_unflattened_output(EEG, input_chans=input_chans)

                output_np = output.detach().float().cpu().numpy()
                target_np = target.detach().float().cpu().numpy()
                # calculate MAE, bandwise MAE, bandwise R2 and R2 metrics for each trial and save to disk
                print(f"Saving trial outputs and metrics to disk...")
                for j in range(output_np.shape[0]):
                    trial_output = output_np[j]
                    trial_target = target_np[j]
                    subject, condition, idx = subjects[j], conditions[j], ids[j]

                    window_metrics = {}
                    for name, window in settings.METRIC_WINDOWS.items():
                        print("Calculating metrics for window:", name)
                        cut_output = core_utils.cut_to_timewindow(trial_output, settings.TIME_WINDOW, window, settings.RSFREQ)
                        cut_target = core_utils.cut_to_timewindow(trial_target, settings.TIME_WINDOW, window, settings.RSFREQ)
                        window_metrics[name] = core_utils.get_inference_metrics(cut_output, cut_target, freqs=freqs,
                                                                   freq_bands=freq_bands, eps=settings.EPS)
                    save_inference_trial_hdf(subject, condition, idx, h5file, trial_target, trial_output, window_metrics)


def get_args():
    parser = argparse.ArgumentParser(description='Run inference with a trained model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model architecture to use for inference')
    parser.add_argument('--target_mode', type=str, default='wavelet', help='Target mode for the model (e.g., wavelet, superlet)')
    parser.add_argument('--postpatch', action='store_true', help='Whether to run inference on postpatch data')
    return parser.parse_args()

def main(opts):
    print(f"Running inference with {opts.model_name} model, postpatch mode: {opts.postpatch}")
    total_time = 0
    io_handler = EpochIO(ext='.pt')
    # loads data in format {subject: {condition: [(id1, epoch_tensor1), (id2, epoch_tensor2), ...]}}
    print("Loading baseline data...")
    finetune_all_data = io_handler.load_all_epochs(settings.FINETUNE_EPOCHS_DIR)
    postpatch_all_data = None
    if opts.postpatch:
        print("Loading postpatch data...")
        postpatch_all_data = io_handler.load_all_epochs(settings.PATCH_EPOCHS_DIR)

    splitter = KFoldSplitter(data_dir=settings.FINETUNE_EPOCHS_DIR, seed=settings.SEED, all_data=finetune_all_data, k_folds=6, no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
    folds = splitter.split_by_subject()

    ch_names = settings.CHORDER_STANDARD
    # cut off "-REF" and "EEG " from channel names for compatibility with get_input_chans function
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

    bands = settings.FREQ_BANDS if opts.target_mode in ["wavelet", "superlet"] else None

    for i, fold in enumerate(folds):
        start_time = time.time()
        print("="*30)
        print(f"Running inference for fold {i + 1}/{len(folds)}")
        testing_ids = list(fold.testing_epochs.keys())
        print("Testing subjects for this fold:", testing_ids)
        _, _, testing_dataset = fold.initialize_datasets(target_mode=opts.target_mode, wavelet_dir=settings.FINETUNE_WAVELET_DIR, log=True, normalize=True)

        print("Setting up model with target mode:", opts.target_mode)
        if opts.target_mode == "autoencoder":
            model_params = settings.AUTOENCODER_PARAMS
            model = EEGAutoencoder(**model_params)
        elif opts.target_mode in ["wavelet", "superlet"]:
            model_params = settings.SPECTRAL_DECODER_PARAMS
            model = EEGSpectralDecoder(**model_params)
        else:
            raise ValueError(f"Invalid target mode: {opts.target_mode}")

            # loading checkpoint into model
        checkpoint_path = settings.FINETUNE_OUTPUT_DIR / opts.model_name / str(i + 1) / "checkpoint-best_val.pth"
        print("using checkpoint:", checkpoint_path)
        if not checkpoint_path.exists():
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        # set model to eval mode and move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        output_path = settings.INFERENCE_DIR / opts.model_name / str(i + 1)
        baseline_file = output_path / "baseline_reconstruction.h5"
        output_path.mkdir(parents=True, exist_ok=True)
        print("Running baseline reconstruction...")
        run_reconstruction(model, testing_dataset, ch_names, device, baseline_file, settings.FREQS, bands)
        print("Baseline reconstruction finished.")

        # run reconstruction on postpatch data if enabled (on the same subjects as the baseline reconstruction)
        if opts.postpatch:

            postpatch_file = output_path / "postpatch_reconstruction.h5"

            postpatch_fold_items = {subject: postpatch_all_data[subject] for subject in testing_ids}
            postpatch_dataset = FinetuneDataset(postpatch_fold_items, target_mode=opts.target_mode, wavelet_dir=settings.PATCH_WAVELET_DIR, window_size=settings.WINDOW_SIZE, log=True, normalize=True)
            print("Running post-patch reconstruction...")
            run_reconstruction(model, postpatch_dataset, ch_names, device, postpatch_file, settings.FREQS, bands)
            print("Post-patch reconstruction finished.")
        fold_time = time.time() - start_time
        print(f"Fold time: {str(datetime.timedelta(seconds=int(fold_time)))}")
        total_time += fold_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Inference finished for model {opts.model_name}. Elapsed time: {total_time_str}")



if __name__ == '__main__':
    opts = get_args()
    main(opts)