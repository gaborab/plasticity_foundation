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

from core.architectures import EEGAutoencoder, EEGSpectralDecoder, LocationDecoder
from core.datasets import KFoldSplitter, FinetuneDataset, ClassFinetuneDataset
from core.io import save_inference_trial_hdf, load_interpolated_channels
from core import settings, io
from core.io_models import TargetStats
from libs.labram import utils
from einops import rearrange
from core import utils as core_utils
from core.decoding import data as decoding_data
from run_general_finetune import pseudo_session


def unmasked_pass(model, eeg, input_chans, interpolated=None):
    """
    Returns the (batch, n_chans, ...) output of one pass that hides nothing but the interpolated channels (None hides nothing):
    the reconstruction of every channel, or the per-channel logits (batch, n_chans, n_classes) of a LocationDecoder.
    """
    with torch.cuda.amp.autocast():
        return model.get_unflattened_output(eeg, input_chans=input_chans, channel_mask=interpolated).float()


def masked_pass(model, eeg, input_chans, neighbour_table, interpolated=None, classify=False):
    """
    Runs one pass per channel c that hides c's cluster (neighbour_table row c) in every sample, plus the interpolated
    channels, and returns the (batch, n_chans, ...) results, row c from the pass hiding c's cluster:
    reconstruction: the reconstruction of channel c, the hidden target.
    classify: the class logits with c's cluster hidden (occlusion), the mean over all channel tokens of that pass.
    """
    n_chans = neighbour_table.shape[0]
    output = None
    for ch in range(n_chans):
        hidden = neighbour_table[ch].expand(eeg.shape[0], -1)  # (batch, n_chans), the same cluster for every sample
        if interpolated is not None:
            hidden = hidden | interpolated
        with torch.cuda.amp.autocast():
            pass_output = model.get_unflattened_output(eeg, input_chans=input_chans, channel_mask=hidden).float()
        ch_output = pass_output.mean(dim=1) if classify else pass_output[:, ch] # get occlusion logits for classification, not the masked channel's contribution
        if output is None:
            output = torch.empty(eeg.shape[0], n_chans, *ch_output.shape[1:], device=eeg.device)
        output[:, ch] = ch_output
    return output


def reconstruct_batch(model, eeg, input_chans, neighbour_table=None, interpolated=None, classify=False):
    """
    Returns (output, occlusion) of a batch.
    Reconstruction: output is the masked pass if neighbour_table is given (a model trained with masking), otherwise the
    unmasked pass; occlusion is None.
    Classification: output is the unmasked pass, the per-channel logits (batch, n_chans, n_classes), whose mean over the
    channels is the class logits. occlusion is the masked pass, the class logits with each channel's cluster hidden
    (batch, n_chans, n_classes), only for a model trained with masking (neighbour_table given), otherwise None.
    interpolated: (batch, n_chans) bool tensor on the device of channels hidden in every pass (interpolated channels), or None.
    """
    if classify:
        occlusion = masked_pass(model, eeg, input_chans, neighbour_table, interpolated, classify=True) if neighbour_table is not None else None
        return unmasked_pass(model, eeg, input_chans, interpolated), occlusion
    if neighbour_table is not None:
        return masked_pass(model, eeg, input_chans, neighbour_table, interpolated), None
    return unmasked_pass(model, eeg, input_chans, interpolated), None


def run_reconstruction(model, dataset, ch_names, device, output_file, valid_bins=None, band_indices=None,
                       neighbour_table=None, hide_interpolated=False, target_fs=settings.TARGET_FS, classify=False):
    """
    Runs the model on every sample of dataset and saves the results per trial to output_file (see io.save_inference_trial_hdf).
    Reconstruction (FinetuneDataset): target, output and metrics per trial and patch, {subject}/{condition}/{id}/{patch}.
    The metrics are computed on every window of settings.METRIC_WINDOWS, cut from the patch in time with target_fs
    (time bins per second of the targets). valid_bins: {patch: (freqs, time) bool mask} of the bins computed from the
    patch's own data (utils.valid_wavelet_bins), the other bins are NaN in the metrics. None (raw-signal targets) scores
    every bin. dataset.target_stats converts the model outputs back to dB, the dataset has to return dB targets (normalize=False).
    Classification (classify, ClassFinetuneDataset): per trial {subject}/{condition}/{id} the class id as target, the
    per-channel logits (n_chans, n_classes) as output, whose mean over the channels is the class logits, and for a model
    trained with masking the occlusion logits (n_chans, n_classes): row c the class logits with c's cluster hidden.
    neighbour_table: (n_chans, n_chans) bool cluster table on device for a model trained with masking, None for one unmasked pass.
    hide_interpolated: hide the interpolated channels of every sample from the input as well, as training did.
    Interpolated channels (metadata["allowed"] False) are NaN in the saved reconstruction and left out of the metrics, and
    NaN rows of the occlusion logits (a cluster centred on an interpolated channel was never hidden in training). Every
    trial group stores them in its "interpolated" attribute.
    """
    sampler = torch_utils.data.SequentialSampler(dataset)
    loader = torch_utils.data.DataLoader(dataset, sampler=sampler, batch_size=256)
    with h5py.File(output_file, "w") as h5file, torch.no_grad():
        input_chans = utils.get_input_chans(ch_names)
        for i, batch in enumerate(loader):
            print(f"Reconstructing trials in batch {i} (batch size: {loader.batch_size})")
            eeg, targets, metadata = batch[0], batch[1], batch[2]

            eeg = core_utils.prepare_eeg(eeg, device)  # (batch, n_chans, n_patches, 200)
            interpolated = ~metadata['allowed'].to(device) if hide_interpolated else None
            output, occlusion = reconstruct_batch(model, eeg, input_chans, neighbour_table, interpolated, classify)
            if dataset.target_stats is not None:
                output = dataset.target_stats.denormalize(output)  # the model predicts normalized targets
            output_np = output.cpu().numpy()
            target_np = targets.float().numpy()
            occlusion_np = occlusion.cpu().numpy() if occlusion is not None else None
            allowed = metadata['allowed'].numpy()

            print(f"Saving trial outputs and metrics to disk...")
            if classify:
                save_class_metrics(output_np, occlusion_np, target_np, metadata, allowed, h5file)
            else:
                save_reconstruction_metrics(output_np, target_np, metadata, allowed, h5file, target_fs, band_indices,
                                            valid_bins)


def save_reconstruction_metrics(output, target, metadata, allowed, h5file, target_fs, band_indices, valid_bins):
    for j in range(output.shape[0]):
        trial_output, trial_target = output[j], target[j]
        trial_output[~allowed[j]] = np.nan  # interpolated channels are never targets
        subject, condition, idx, patch = metadata['subject'][j], metadata['condition'][j], int(metadata['id'][j]), \
            metadata['patch'][j]

        # bins reaching into the patch's reflect padding are left out of the metrics
        scored_output = np.where(valid_bins[patch], trial_output, np.nan) if valid_bins is not None else trial_output
        window_metrics = {}
        for name, windows in settings.METRIC_WINDOWS.items():
            cut_output = core_utils.cut_to_timewindow(scored_output, settings.PATCH_WINDOWS[patch], windows[patch],
                                                      target_fs)
            cut_target = core_utils.cut_to_timewindow(trial_target, settings.PATCH_WINDOWS[patch], windows[patch],
                                                      target_fs)
            window_metrics[name] = core_utils.get_inference_metrics(cut_output, cut_target, band_indices=band_indices,
                                                                    eps=settings.EPS)
        save_inference_trial_hdf(subject, condition, idx, h5file, trial_target, trial_output, window_metrics,
                                 patch=patch)
        h5file[f"{subject}/{condition}/{idx}"].attrs["interpolated"] = np.flatnonzero(~allowed[j])


def save_class_metrics(output, occlusion_output, target, metadata, allowed, h5file):
    """Saves the class id, the per-channel logits and the occlusion logits (if any) of every trial of a batch, see run_reconstruction."""
    for j in range(output.shape[0]):
        trial_output, trial_target = output[j], int(target[j])  # the class id

        trial_occlusion_output = occlusion_output[j] if occlusion_output is not None else None
        if trial_occlusion_output is not None:
            trial_occlusion_output[~allowed[j]] = np.nan  # clusters centred on interpolated channels were never hidden in training
        subject, condition, idx = metadata['subject'][j], metadata['condition'][j], int(metadata['id'][j])

        save_inference_trial_hdf(subject, condition, idx, h5file, trial_target, trial_output, window_metrics={},
                                 patch=None, trial_occlusion_output=trial_occlusion_output)
        h5file[f"{subject}/{condition}/{idx}"].attrs["interpolated"] = np.flatnonzero(~allowed[j])


def load_run_settings(model_dir: Path, opts) -> Dict[str, Any]:
    """
    Returns the settings of the training run inference has to match ({"target_mode", "masked", "n_neighbors",
    "hide_interpolated", "task_classify", "postpatch", "k_folds"}) from the run's metadata.json. Settings older metadata
    doesn't record fall back to the CLI arguments, task_classify and postpatch to False (runs before they existed, trained
    on the baseline), k_folds to the number of fold folders in model_dir.
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
        "hide_interpolated": opts.hide_interpolated,
        "task_classify": False,
        "postpatch": False,
        "k_folds": len([d for d in model_dir.iterdir() if d.is_dir() and d.name.isdigit()]),
    }
    run_settings = {}
    for name, fallback in fallbacks.items():
        if hyperparameters.get(name) is not None:
            run_settings[name] = hyperparameters[name]
        else:
            run_settings[name] = fallback
            print(f"WARNING: {name} not in the run's metadata, using {fallback}")
    # pseudo-session runs (run_general_finetune.py --pseudo_session), None for every other run
    for name in ("pseudo_session", "split_seed", "plant_condition", "plant_attenuation"):
        run_settings[name] = hyperparameters.get(name)
    return run_settings


def load_target_stats(fold_dir: Path) -> Optional[TargetStats]:
    """Returns the target stats saved with a fold (target_stats.json), None if the fold has none."""
    stats_path = fold_dir / "target_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        return TargetStats.from_dict(json.load(f))


def build_test_dataset(task_classify, epochs, interpolations, target_mode, target_stats):
    """The dataset inference runs on: a ClassFinetuneDataset for classification, a FinetuneDataset with dB targets otherwise."""
    if task_classify:
        return ClassFinetuneDataset(epochs, epoch_window=settings.EPOCH_WINDOW, patch_window=settings.CLASS_PATCH_WINDOW,
                                    fs=settings.RSFREQ, condition_classes=settings.CONDITION_CLASS_IDS, interpolations=interpolations)
    return FinetuneDataset(epochs, **settings.PATCH_PARAMS, target_mode=target_mode, log=True, normalize=False,
                           interpolations=interpolations, target_stats=target_stats)


def get_args():
    """Parses the command line arguments of the inference run."""
    parser = argparse.ArgumentParser(description='Run inference with a trained model')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model architecture to use for inference')
    parser.add_argument('--target_mode', type=str, default='wavelet',
                        help='Fallback if the run\'s metadata.json doesn\'t record it: target mode of the model (wavelet or autoencoder)')
    parser.add_argument('--postpatch', action='store_true',
                        help='Whether to run inference on postpatch data as well (a model trained on the baseline). '
                             'A model trained on post-patching data (run_general_finetune.py --postpatch) is always '
                             'evaluated on its own post-patching test sets only')
    parser.add_argument('--masked', action='store_true',
                        help='Fallback if the run\'s metadata.json doesn\'t record it: masked-channel reconstruction, '
                             'hide every target channel with its nearest electrodes in turn')
    parser.add_argument('--n_neighbors', type=int, default=settings.N_NEIGHBOURS,
                        help='Fallback if the run\'s metadata.json doesn\'t record it: nearest electrodes hidden together with the target channel')
    parser.add_argument('--hide_interpolated', action='store_true',
                        help='Fallback if the run\'s metadata.json doesn\'t record it: interpolated channels were hidden from the input '
                             'in training (masked-channel reconstruction)')
    parser.add_argument('--class_occlusion_injection', type=float, nargs='+', default=None,
                        help='Occlusion check of a masked class run trained on the baseline: for every attenuation given, the '
                             'location-specific posterior response of --injection_conditions is attenuated '
                             '(core.decoding.data.plant_class_attenuation, as the pseudo-session runs) in every fold\'s baseline '
                             'test trials, and only these are run, saved as the post-patching session of '
                             '{model_name}_inj_{attenuation}. The regular outputs are not run or touched')
    parser.add_argument('--injection_conditions', type=str, nargs='+', default=['right_2'],
                        help='Conditions attenuated with --class_occlusion_injection (default R_mid, as the LDA checks)')

    return parser.parse_args()


def main(opts):
    print(f"Running inference with {opts.model_name} model, postpatch mode: {opts.postpatch}")
    total_time = 0
    model_dir = settings.FINETUNE_OUTPUT_DIR / opts.model_name
    run_settings = load_run_settings(model_dir, opts)
    print(f"Run settings: {run_settings}")
    target_mode, masked, k_folds, task_classify = run_settings["target_mode"], run_settings["masked"], run_settings[
        "k_folds"], run_settings["task_classify"]
    spectral = target_mode in ["wavelet", "superlet"]
    # the session the model was trained on: its test sets are rebuilt from the keys saved during training and saved as
    # {session}_reconstruction.h5
    session = "postpatch" if run_settings["postpatch"] else "baseline"
    if session == "postpatch":
        input_dir, target_dir, session_dir = settings.PATCH_INPUT_EPOCHS_DIR, settings.PATCH_WAVELET_DIR, settings.PATCH_DIR
        if opts.postpatch:
            print("Model trained on post-patching data: evaluated on its own post-patching test sets only")
    else:
        input_dir, target_dir, session_dir = settings.FINETUNE_INPUT_EPOCHS_DIR, settings.FINETUNE_WAVELET_DIR, settings.FINETUNE_DIR
    input_data = input_dir
    # a pseudo-session run: the same half of the baseline (and plant) as in training, saved as baseline (pre) or postpatch (post)
    pseudo = run_settings["pseudo_session"] is not None
    if pseudo:
        session = "baseline" if run_settings["pseudo_session"] == "pre" else "postpatch"
        input_data = pseudo_session(run_settings["pseudo_session"], run_settings["split_seed"], run_settings["plant_condition"],
                                    run_settings["plant_attenuation"])
        print(f"Pseudo {run_settings['pseudo_session']} run (split seed {run_settings['split_seed']}, plant "
              f"{run_settings['plant_condition']} {run_settings['plant_attenuation']}): saved as the {session} session")

    print(f"Loading training split keys and building {session} inference folds...")
    splitter = KFoldSplitter(input_data=input_data,
                             target_data_dir=target_dir,
                             target_mode=target_mode, seed=settings.SEED, k_folds=k_folds,
                             no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
    folds = splitter.split_from_keys(io.load_training_split_keys(model_dir, k_folds))

    ch_names = settings.CHORDER_STANDARD
    # cut off "-REF" and "EEG " from channel names for compatibility with get_input_chans function
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

    # occlusion injection check: the baseline epochs with the location-specific posterior response of the injection
    # conditions attenuated, one splitter per attenuation (the templates are the subjects' class means over all their
    # baseline trials), from which every fold's test trials are selected
    injected_splitters = {}
    if opts.class_occlusion_injection:
        if not (task_classify and masked and session == "baseline" and not pseudo):
            raise ValueError("--class_occlusion_injection needs a masked class run trained on the baseline")
        posterior = [name in settings.POSTERIOR_CHANNELS for name in ch_names]
        for attenuation in opts.class_occlusion_injection:
            injected = decoding_data.plant_class_attenuation(splitter.input_target_data, opts.injection_conditions, attenuation,
                                                             posterior, settings.EPOCH_WINDOW, settings.RSFREQ,
                                                             settings.PLANT_ATTENUATION_WINDOW, list(settings.LOCATION_CLASSES))
            injected_splitters[attenuation] = KFoldSplitter(input_data=injected, target_data_dir=None, target_mode=target_mode,
                                                            seed=settings.SEED, k_folds=k_folds,
                                                            no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
        print(f"Occlusion injection: the location-specific response of {opts.injection_conditions} attenuated by "
              f"{opts.class_occlusion_injection} on {settings.POSTERIOR_CHANNELS}, {settings.PLANT_ATTENUATION_WINDOW} s; "
              f"only the injected test trials are run")

    # baseline model: every post-patching trial is reconstructed once, by the fold its stratified within-subject split assigns it to
    postpatch_folds = None
    run_postpatch = opts.postpatch and session == "baseline" and not injected_splitters and not pseudo
    if run_postpatch:
        print("Building post-patching inference folds...")
        postpatch_splitter = KFoldSplitter(input_data=settings.PATCH_INPUT_EPOCHS_DIR,
                                           target_data_dir=settings.PATCH_WAVELET_DIR,
                                           target_mode=target_mode, seed=settings.SEED, k_folds=k_folds,
                                           no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
        postpatch_folds = postpatch_splitter.split_within_subjects()

    # (freqs, time) masks of the bins computed from each patch's own data, the metrics leave the other bins out
    valid_bins = None
    if spectral:
        valid_bins = {patch: core_utils.valid_wavelet_bins(settings.FREQS, window, window, settings.RSFREQ,
                                                           **settings.WAVELET_PARAMS,
                                                           pool=settings.TARGET_POOL)
                      for patch, window in settings.PATCH_WINDOWS.items()}

    band_indices = settings.FREQ_BAND_INDICES if spectral else None

    # the interpolated channels of every trial: never scored (NaN) and recorded in the output files, in every run.
    # They are hidden from the input only if training hid them (the run's hide_interpolated)
    trial_interpolations = load_interpolated_channels(session_dir / "epochs_manifest.csv")
    trial_interpolations_postpatch = load_interpolated_channels(settings.PATCH_DIR / "epochs_manifest.csv") if run_postpatch else None
    hide_interpolated = run_settings["hide_interpolated"]
    # model trained with masking: reconstruction hides every target channel with its cluster, classification runs the
    # occlusion passes as well
    neighbour_table = None
    if masked:
        neighbour_table = core_utils.build_neighbor_mask(n_neighbours=run_settings["n_neighbors"],
                                                         n_chans=settings.N_CHANNELS)
        task = "occlusion: every cluster of" if task_classify else "masked-channel reconstruction: every target channel hidden with its"
        print(f"Model trained with masking, {task} {run_settings['n_neighbors']} nearest electrodes")
    print("Interpolated channels are " + ("hidden from the input, as in training" if hide_interpolated else "visible inputs, as in training"))

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

        testing_dataset = build_test_dataset(task_classify, fold.testing_epochs, trial_interpolations, target_mode, target_stats)

        print("Setting up model with target mode:", target_mode)
        if target_mode == "autoencoder":
            model = EEGAutoencoder(**settings.AUTOENCODER_PARAMS)
        elif spectral:
            model = EEGSpectralDecoder(**settings.SPECTRAL_DECODER_PARAMS)
        elif target_mode == "classify":
            model = LocationDecoder(**settings.LOCATION_DECODER_PARAMS)
        else:
            raise ValueError(f"Invalid target mode: {target_mode}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        # set model to eval mode and move to device
        model.to(device)
        model.eval()
        fold_neighbour_table = torch.as_tensor(neighbour_table, device=device) if neighbour_table is not None else None

        if injected_splitters:
            # the fold's own (unseen) test trials with the injection, saved as the post-patching session of {model}_inj_{a}
            for attenuation, injected_splitter in injected_splitters.items():
                injected_path = settings.INFERENCE_DIR / f"{opts.model_name}_inj_{attenuation:g}" / str(i + 1)
                injected_path.mkdir(parents=True, exist_ok=True)  # the fold's file of an earlier run is overwritten
                print(f"Running injected test trials, attenuation {attenuation:g}...")
                injected_dataset = build_test_dataset(task_classify, injected_splitter.select(fold.keys["test"]),
                                                      trial_interpolations, target_mode, target_stats)
                run_reconstruction(model, injected_dataset, ch_names, device, injected_path / "postpatch_reconstruction.h5",
                                   valid_bins, band_indices, fold_neighbour_table, hide_interpolated, classify=task_classify)
            total_time += time.time() - start_time
            continue

        output_path = settings.INFERENCE_DIR / opts.model_name / str(i + 1)
        session_file = output_path / f"{session}_reconstruction.h5"
        shutil.rmtree(output_path, ignore_errors=True)  # delete existing output directory for this fold if it exists
        output_path.mkdir(parents=True, exist_ok=True)
        if session == "postpatch":
            # the test keys, as a baseline model's inference saves them for its post-patching test sets
            with open(output_path / "postpatch_keys.json", "w", encoding="utf-8") as f:
                json.dump({"test": fold.keys["test"]}, f)
        print(f"Running {session} reconstruction...")
        run_reconstruction(model, testing_dataset, ch_names, device, session_file, valid_bins, band_indices,
                           fold_neighbour_table, hide_interpolated, classify=task_classify)
        print(f"{session} reconstruction finished.")

        if run_postpatch:
            print("Running post-patch reconstruction...")
            postpatch_file = output_path / "postpatch_reconstruction.h5"
            with open(output_path / "postpatch_keys.json", "w", encoding="utf-8") as f:
                json.dump({"test": postpatch_folds[i].keys["test"]}, f)
            postpatch_dataset = build_test_dataset(task_classify, postpatch_folds[i].testing_epochs, trial_interpolations_postpatch,
                                                   target_mode, target_stats)
            run_reconstruction(model, postpatch_dataset, ch_names, device, postpatch_file, valid_bins, band_indices,
                               fold_neighbour_table, hide_interpolated, classify=task_classify)
            print("Post-patch reconstruction finished.")
        fold_time = time.time() - start_time
        print(f"Fold time: {str(datetime.timedelta(seconds=int(fold_time)))}")
        total_time += fold_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Inference finished for model {opts.model_name}. Elapsed time: {total_time_str}")


if __name__ == '__main__':
    opts = get_args()
    main(opts)
