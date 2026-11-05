"""
Ridge baseline of a masked LaBraM reconstruction run, evaluated like run_inference.py: the same folds (the run's saved
keys), per fold one ridge model per target channel fitted on the baseline training patches (ridge_inj_test.fit_ridge_models,
alpha picked on the validation trials), every channel of the test patches predicted from the channels visible when its
cluster is hidden, wavelet-transformed as the targets and saved with run_inference.save_reconstruction_metrics.
With --postpatch the post-patching trials are reconstructed as well, every trial by the fold run_inference.py assigns it to.

Outputs: INFERENCE_DIR/<model_name>_ridge/<fold>/{baseline,postpatch}_reconstruction.h5 (and postpatch_keys.json), read like
the LaBraM run's files (e.g. notebooks/max_t_permutation.ipynb with model_name = "<model_name>_ridge"), and
INFERENCE_DIR/<model_name>_ridge/alpha_sweep.csv.

    python ridge_inference.py --model_name masking_10_19_2 --postpatch
"""
import argparse
import datetime
import json
import shutil
import time

import h5py
import numpy as np
import pandas as pd
import torch
from torch import utils as torch_utils

from core import settings, io
from core import utils as core_utils
from core.datasets import KFoldSplitter
from core.io import load_interpolated_channels
from ridge_inj_test import fit_ridge_models, predict_channels
from run_inference import build_test_dataset, save_reconstruction_metrics


def run_ridge_reconstruction(models, dataset, neighbour_table, output_file, valid_bins, band_indices):
    """
    The ridge counterpart of run_inference.run_reconstruction: every channel of every patch sample of dataset (FinetuneDataset
    with dB targets) predicted by its ridge model, wavelet-transformed as the targets (utils.wavelet_target) and saved
    with save_reconstruction_metrics, so the file reads as a LaBraM {session}_reconstruction.h5.
    """
    loader = torch_utils.data.DataLoader(dataset, sampler=torch_utils.data.SequentialSampler(dataset), batch_size=256)
    with h5py.File(output_file, "w") as h5file:
        for i, batch in enumerate(loader):
            print(f"Predicting patches in batch {i + 1}/{len(loader)}")
            inputs, targets, metadata = batch[0].float().numpy(), batch[1], batch[2]
            predicted = predict_channels(models, inputs, neighbour_table)  # (patches, channels, times)
            output = np.stack([core_utils.wavelet_target(torch.from_numpy(patch), settings.RSFREQ, settings.FREQS, settings.PADDING,
                                                         settings.WAVELET_PARAMS, settings.TARGET_POOL).numpy()
                               for patch in predicted])  # dB, (patches, channels, freqs, time bins)
            save_reconstruction_metrics(output, targets.float().numpy(), metadata, metadata["allowed"].numpy(), h5file,
                                        settings.TARGET_FS, band_indices, valid_bins)


def get_args():
    parser = argparse.ArgumentParser(description="Ridge baseline of a masked reconstruction run, saved as run_inference.py saves the run")
    parser.add_argument('--model_name', type=str, required=True,
                        help='masked LaBraM reconstruction run trained on the baseline, whose folds and masking the ridge uses')
    parser.add_argument('--postpatch', action='store_true', default=False,
                        help='reconstruct the post-patching trials as well, as run_inference.py --postpatch')
    parser.add_argument('--target_mode', type=str, help='fallback if training did not log target_mode')
    parser.add_argument('--k_folds', type=int, default=6, help='fallback if training did not log k_folds')
    opts = parser.parse_args()
    opts.from_csv = False  # io.load_run_settings checks the target settings unless from_csv
    return opts


def main(opts):
    model_dir = settings.FINETUNE_OUTPUT_DIR / opts.model_name
    run_settings = io.load_run_settings(model_dir, opts, settings.SPECTRAL_DECODER_PARAMS["freq_window"], settings.TARGET_POOL)
    print(f"Run settings: {run_settings}")
    if not run_settings["masked"]:
        raise ValueError("The ridge baseline needs a masked reconstruction run: it predicts every channel with its cluster hidden")
    k_folds = run_settings["k_folds"]
    output_root = settings.INFERENCE_DIR / f"{opts.model_name}_ridge"
    print(f"Ridge baseline of {opts.model_name}, postpatch: {opts.postpatch}, outputs in {output_root}")

    print("Loading training split keys and building baseline inference folds...")
    splitter = KFoldSplitter(input_data=settings.FINETUNE_INPUT_EPOCHS_DIR, target_data_dir=settings.FINETUNE_WAVELET_DIR,
                             target_mode="wavelet", seed=settings.SEED, k_folds=k_folds, no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
    folds = splitter.split_from_keys(io.load_training_split_keys(model_dir, k_folds))
    # every post-patching trial is reconstructed once, by the fold its stratified within-subject split assigns it to (as run_inference.py)
    postpatch_folds = None
    if opts.postpatch:
        print("Building post-patching inference folds...")
        postpatch_splitter = KFoldSplitter(input_data=settings.PATCH_INPUT_EPOCHS_DIR, target_data_dir=settings.PATCH_WAVELET_DIR,
                                           target_mode="wavelet", seed=settings.SEED, k_folds=k_folds,
                                           no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
        postpatch_folds = postpatch_splitter.split_within_subjects()

    ch_names = [name.split(' ')[-1].split('-')[0] for name in settings.CHORDER_STANDARD]
    # the metric settings of run_inference.py: bins reaching into a patch's reflect padding are left out
    valid_bins = {patch: core_utils.valid_wavelet_bins(settings.FREQS, window, window, settings.RSFREQ, **settings.WAVELET_PARAMS,
                                                       pool=settings.TARGET_POOL)
                  for patch, window in settings.PATCH_WINDOWS.items()}
    trial_interpolations = load_interpolated_channels(settings.FINETUNE_DIR / "epochs_manifest.csv")
    trial_interpolations_postpatch = load_interpolated_channels(settings.PATCH_DIR / "epochs_manifest.csv") if opts.postpatch else None
    neighbour_table = core_utils.build_neighbor_mask(n_neighbours=run_settings["n_neighbors"], n_chans=settings.N_CHANNELS)
    print(f"Every target channel hidden with its {run_settings['n_neighbors']} nearest electrodes; interpolated channels are "
          f"never targets and stay visible inputs")

    total_time, sweeps = 0, []
    for i, fold in enumerate(folds):
        start_time = time.time()
        print("=" * 30)
        print(f"Fold {i + 1}/{len(folds)}: fitting the ridge models")
        models, sweep = fit_ridge_models(fold.training_epochs, fold.validation_epochs, trial_interpolations, neighbour_table, ch_names)
        sweep.insert(0, "fold", i + 1)
        sweeps.append(sweep)

        output_path = output_root / str(i + 1)
        shutil.rmtree(output_path, ignore_errors=True)  # an earlier run's files of this fold
        output_path.mkdir(parents=True, exist_ok=True)
        print("Running baseline reconstruction...")
        run_ridge_reconstruction(models, build_test_dataset(False, fold.testing_epochs, trial_interpolations, "wavelet", None),
                                 neighbour_table, output_path / "baseline_reconstruction.h5", valid_bins, settings.FREQ_BAND_INDICES)
        if opts.postpatch:
            print("Running post-patch reconstruction...")
            with open(output_path / "postpatch_keys.json", "w", encoding="utf-8") as f:
                json.dump({"test": postpatch_folds[i].keys["test"]}, f)
            postpatch_dataset = build_test_dataset(False, postpatch_folds[i].testing_epochs, trial_interpolations_postpatch, "wavelet", None)
            run_ridge_reconstruction(models, postpatch_dataset, neighbour_table, output_path / "postpatch_reconstruction.h5",
                                     valid_bins, settings.FREQ_BAND_INDICES)
        fold_time = time.time() - start_time
        print(f"Fold time: {datetime.timedelta(seconds=int(fold_time))}")
        total_time += fold_time

    pd.concat(sweeps).to_csv(output_root / "alpha_sweep.csv", index=False)
    print(f"Ridge inference of {opts.model_name} finished in {datetime.timedelta(seconds=int(total_time))}, outputs in {output_root}")


if __name__ == '__main__':
    main(get_args())
