import argparse
import json
import shutil
import time
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch
from torch import utils as torch_utils

from core.architectures import EEGAutoencoder, EEGSpectralDecoder
from core.datasets import KFoldSplitter, FinetuneDataset
from core.io import save_inference_trial_hdf, load_interpolated_channels
from core import settings, io
from core.io_models import TargetStats
from libs.labram import utils
from einops import rearrange
from core import utils as core_utils


def reconstruct_batch(model, eeg, input_chans, n_chans, neighbour_table=None):
    """
    Returns the (batch, n_chans, ...) reconstruction of a batch. Masked (neighbour_table given): row c comes from the pass
    that hides channel c with its cluster in every sample. Unmasked: one pass.
    """
    if neighbour_table is None:
        with torch.cuda.amp.autocast():
            return model.get_unflattened_output(eeg, input_chans=input_chans).float()
    output = None
    for ch in range(n_chans):
        hidden = neighbour_table[ch].expand(eeg.shape[0], -1)  # (batch, n_chans), the same cluster for every sample
        with torch.cuda.amp.autocast():
            ch_output = model.get_unflattened_output(eeg, input_chans=input_chans, channel_mask=hidden)[:, ch].float()
        if output is None:
            output = torch.empty(eeg.shape[0], n_chans, *ch_output.shape[1:], device=eeg.device)
        output[:, ch] = ch_output
    return output


def run_reconstruction(model, dataset, ch_names, device, output_file, valid_bins=None, band_indices=None,
                       neighbour_table=None):
    """
    Reconstructs every sample of dataset and saves target, output and metrics per trial and patch to output_file
    ({subject}/{condition}/{id}/{patch}, see io.save_inference_trial_hdf). The metrics are computed on every window of
    settings.METRIC_WINDOWS, cut from the patch in time.
    valid_bins: {patch: (freqs, time) bool mask} of the bins computed from the patch's own data (utils.valid_wavelet_bins),
    the other bins are NaN in the metrics. None (raw-signal targets) scores every bin.
    dataset.target_stats converts the model outputs back to dB, the dataset has to return dB targets (normalize=False).
    neighbour_table: (n_chans, n_chans) bool cluster table on device for masked-channel reconstruction, None for one unmasked pass.
    Interpolated channels (metadata["allowed"] False) are NaN in the saved output and left out of the metrics,
    and every trial group stores them in its "interpolated" attribute.
    """
    sampler = torch_utils.data.SequentialSampler(dataset)
    loader = torch_utils.data.DataLoader(dataset, sampler=sampler, batch_size=256)
    with h5py.File(output_file, "w") as h5file, torch.no_grad():
        input_chans = utils.get_input_chans(ch_names)
        for i, batch in enumerate(loader):
            print(f"Reconstructing trials in batch {i} (batch size: {loader.batch_size})")
            eeg, targets, metadata = batch[0], batch[1], batch[2]
            eeg = eeg.float().to(device, non_blocking=True) / 100
            eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)

            output = reconstruct_batch(model, eeg, input_chans, eeg.shape[1], neighbour_table)
            if dataset.target_stats is not None:
                output = dataset.target_stats.denormalize(output)  # the model predicts normalized targets
            output_np = output.cpu().numpy()
            target_np = targets.float().numpy()  # dB targets
            allowed = metadata['allowed'].numpy()

            print(f"Saving trial outputs and metrics to disk...")
            for j in range(output_np.shape[0]):
                trial_output, trial_target = output_np[j], target_np[j]
                trial_output[~allowed[j]] = np.nan  # interpolated channels are never targets
                subject, condition, idx, patch = metadata['subject'][j], metadata['condition'][j], int(metadata['id'][j]), metadata['patch'][j]

                # bins reaching into the patch's reflect padding are left out of the metrics
                scored_output = np.where(valid_bins[patch], trial_output, np.nan) if valid_bins is not None else trial_output
                window_metrics = {}
                for name, windows in settings.METRIC_WINDOWS.items():
                    cut_output = core_utils.cut_to_timewindow(scored_output, settings.PATCH_WINDOWS[patch], windows[patch], settings.RSFREQ)
                    cut_target = core_utils.cut_to_timewindow(trial_target, settings.PATCH_WINDOWS[patch], windows[patch], settings.RSFREQ)
                    window_metrics[name] = core_utils.get_inference_metrics(cut_output, cut_target, band_indices=band_indices, eps=settings.EPS)
                save_inference_trial_hdf(subject, condition, idx, patch, h5file, trial_target, trial_output, window_metrics)
                h5file[f"{subject}/{condition}/{idx}"].attrs["interpolated"] = np.flatnonzero(~allowed[j])


def load_run_settings(model_dir: Path, opts) -> Dict[str, Any]:
    """
    Returns the settings of the training run inference has to match ({"target_mode", "masked", "n_neighbors", "k_folds"})
    from the run's metadata.json. Settings older metadata doesn't record fall back to the CLI arguments,
    k_folds to the number of fold folders in model_dir.
    """
    hyperparameters = {}
    metadata_path = model_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            hyperparameters = json.load(f).get("hyperparameters", {})
    else:
        print(f"WARNING: {metadata_path} not found, using the CLI arguments")
    fallbacks = {
        "target_mode": opts.target_mode,
        "masked": opts.masked,
        "n_neighbors": opts.n_neighbors,
        "k_folds": len([d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()]),
    }
    run_settings = {}
    for name, fallback in fallbacks.items():
        if hyperparameters.get(name) is not None:
            run_settings[name] = hyperparameters[name]
        else:
            run_settings[name] = fallback
            print(f"WARNING: {name} not in the run's metadata, using {fallback}")
    return run_settings


def load_target_stats(fold_dir: Path) -> Optional[TargetStats]:
    """Returns the target stats saved with a fold (target_stats.json), None if the fold has none."""
    stats_path = fold_dir / "target_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        return TargetStats.from_dict(json.load(f))


def get_args():
    """Parses the command line arguments of the inference run."""
    parser = argparse.ArgumentParser(description='Run inference with a trained model')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model architecture to use for inference')
    parser.add_argument('--target_mode', type=str, default='wavelet',
                        help='Fallback if the run\'s metadata.json doesn\'t record it: target mode of the model (wavelet or autoencoder)')
    parser.add_argument('--postpatch', action='store_true', help='Whether to run inference on postpatch data')
    parser.add_argument('--masked', action='store_true',
                        help='Fallback if the run\'s metadata.json doesn\'t record it: masked-channel reconstruction, '
                             'hide every target channel with its nearest electrodes in turn')
    parser.add_argument('--n_neighbors', type=int, default=settings.N_NEIGHBOURS,
                        help='Fallback if the run\'s metadata.json doesn\'t record it: nearest electrodes hidden together with the target channel')
    parser.add_argument('--injection', action='store_true', default=False,
                        help='Whether to run injection test for masking after inference')
    return parser.parse_args()


def main(opts):
    print(f"Running inference with {opts.model_name} model, postpatch mode: {opts.postpatch}")
    total_time = 0
    model_dir = settings.FINETUNE_OUTPUT_DIR / opts.model_name
    run_settings = load_run_settings(model_dir, opts)
    print(f"Run settings: {run_settings}")
    target_mode, masked, k_folds = run_settings["target_mode"], run_settings["masked"], run_settings["k_folds"]
    spectral = target_mode in ["wavelet", "superlet"]

    # the baseline test sets are rebuilt from the keys saved during training
    splitter = KFoldSplitter(input_data_dir=settings.FINETUNE_INPUT_EPOCHS_DIR, target_data_dir=settings.FINETUNE_WAVELET_DIR,
                             target_mode=target_mode, seed=settings.SEED, k_folds=k_folds, no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
    folds = splitter.split_from_keys(io.load_training_split_keys(model_dir, k_folds))

    # every post-patching trial is reconstructed once, by the fold its stratified within-subject split assigns it to
    postpatch_folds = None
    if opts.postpatch:
        postpatch_splitter = KFoldSplitter(input_data_dir=settings.PATCH_INPUT_EPOCHS_DIR, target_data_dir=settings.PATCH_WAVELET_DIR,
                                           target_mode=target_mode, seed=settings.SEED, k_folds=k_folds,
                                           no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
        postpatch_folds = postpatch_splitter.split_within_subjects()

    # (freqs, time) masks of the bins computed from each patch's own data, the metrics leave the other bins out
    valid_bins = None
    if spectral:
        valid_bins = {patch: core_utils.valid_wavelet_bins(settings.FREQS, window, window, settings.RSFREQ, **settings.WAVELET_PARAMS)
                      for patch, window in settings.PATCH_WINDOWS.items()}

    ch_names = settings.CHORDER_STANDARD
    # cut off "-REF" and "EEG " from channel names for compatibility with get_input_chans function
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

    band_indices = settings.FREQ_BAND_INDICES if spectral else None

    # masked-channel reconstruction: interpolated channels are never targets, a target is hidden with its cluster
    trial_interpolations_prepatch = None
    trial_interpolations_postpatch = None
    neighbour_table = None
    if masked:
        trial_interpolations_prepatch = load_interpolated_channels(settings.FINETUNE_DIR / "epochs_manifest.csv")
        if opts.postpatch:
            trial_interpolations_postpatch = load_interpolated_channels(settings.PATCH_DIR / "epochs_manifest.csv")
        neighbour_table = core_utils.build_neighbor_mask(n_neighbours=run_settings["n_neighbors"], n_chans=settings.N_CHANNELS)
        print(f"Masked-channel reconstruction: every target channel is hidden with its {run_settings['n_neighbors']} nearest electrodes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, fold in enumerate(folds):
        start_time = time.time()
        print("=" * 30)
        print(f"Running inference for fold {i + 1}/{len(folds)}")
        fold_dir = model_dir / str(i + 1)
        checkpoint_path = fold_dir / "checkpoint-best_val.pth"
        print("using checkpoint:", checkpoint_path)
        if not checkpoint_path.exists():
            print(f"WARNING: no best checkpoint for fold {i + 1}, skipping it")
            continue

        # the model predicts normalized targets, the stats convert its outputs back to dB
        target_stats = load_target_stats(fold_dir)
        if spectral and target_stats is None:
            raise FileNotFoundError(f"Target stats file not found for fold {i + 1}: {fold_dir / 'target_stats.json'}")
        print(f"Target stats: {target_stats}")

        testing_dataset = FinetuneDataset(epochs_dict=fold.testing_epochs, **settings.PATCH_PARAMS, target_mode=target_mode,
                                          log=True, normalize=False, interpolations=trial_interpolations_prepatch, target_stats=target_stats)

        print("Setting up model with target mode:", target_mode)
        if target_mode == "autoencoder":
            model = EEGAutoencoder(**settings.AUTOENCODER_PARAMS)
        elif spectral:
            model = EEGSpectralDecoder(**settings.SPECTRAL_DECODER_PARAMS)
        else:
            raise ValueError(f"Invalid target mode: {target_mode}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        # set model to eval mode and move to device
        model.to(device)
        model.eval()
        fold_neighbour_table = torch.as_tensor(neighbour_table, device=device) if neighbour_table is not None else None

        output_path = settings.INFERENCE_DIR / opts.model_name / str(i + 1)
        baseline_file = output_path / "baseline_reconstruction.h5"
        shutil.rmtree(output_path, ignore_errors=True)  # delete existing output directory for this fold if it exists
        output_path.mkdir(parents=True, exist_ok=True)
        print("Running baseline reconstruction...")
        run_reconstruction(model, testing_dataset, ch_names, device, baseline_file, valid_bins, band_indices,
                           fold_neighbour_table)
        print("Baseline reconstruction finished.")

        if opts.postpatch:
            print("Running post-patch reconstruction...")
            postpatch_file = output_path / "postpatch_reconstruction.h5"
            with open(output_path / "postpatch_keys.json", "w", encoding="utf-8") as f:
                json.dump({"test": postpatch_folds[i].keys["test"]}, f)
            postpatch_dataset = FinetuneDataset(postpatch_folds[i].testing_epochs, **settings.PATCH_PARAMS, target_mode=target_mode,
                                                target_stats=target_stats, log=True, normalize=False, interpolations=trial_interpolations_postpatch)
            run_reconstruction(model, postpatch_dataset, ch_names, device, postpatch_file, valid_bins, band_indices,
                               fold_neighbour_table)
            print("Post-patch reconstruction finished.")
        fold_time = time.time() - start_time
        print(f"Fold time: {str(datetime.timedelta(seconds=int(fold_time)))}")
        total_time += fold_time
        if opts.injection:
            print("Running injection test...")
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Inference finished for model {opts.model_name}. Elapsed time: {total_time_str}")


if __name__ == '__main__':
    opts = get_args()
    main(opts)
