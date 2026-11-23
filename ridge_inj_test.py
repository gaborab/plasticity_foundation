import argparse
import pathlib
import platform

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import utils as torch_utils

from core import settings, io
from core.datasets import KFoldSplitter, InjectionDataset
from core import utils as core_utils
from core.io import load_interpolated_channels
from core.plot_data_processing import create_topomap_info
from labram_inj_test import plot_sources, BATCH_SIZE, score_trials, summarize, summarize_split, plot_injection, print_summary

PLOT_ROOT = settings.PLOT_DIR / "ridge_inj_labram"
OUTPUT_ROOT = settings.BASE_DIR / "outputs" / "ridge_inj_labram"
ALPHAS = [1e-2, 1e-1, 1, 10, 100]

def collect(loader, return_type="inputs"):
    rows = []
    for batch in loader:

        x = batch[0]  # (batch, channels, times); adjust to what your loader returns
        rows.append(x.permute(0, 2, 1).reshape(-1, x.shape[1]).cpu().numpy())
    return np.concatenate(rows)

def stacked_patches(epochs, interpolations, patch_windows, epoch_window, fs):
    """
    Input patches of every trial of an epoch dict ({subject: {condition: [(id, (input, target)), ...]}}) cut to every
    window of patch_windows (n_trials * n_patches, n_chans, patch samples), and their allowed-channel masks
    (n_trials * n_patches, n_chans), False for the trial's interpolated channels (interpolations, None: none).
    """
    X, allowed = [], []
    for subject, conditions in epochs.items():
        for condition, epoch_list in conditions.items():
            for idx, (epoch, _) in epoch_list:
                trial_allowed = np.ones(epoch.shape[0], dtype=bool)
                trial_allowed[list((interpolations or {}).get((subject, condition, idx), []))] = False
                for window in patch_windows.values():
                    X.append(core_utils.cut_to_timewindow(epoch.float(), epoch_window, window, fs).numpy())
                    allowed.append(trial_allowed)
    return np.stack(X), np.stack(allowed)


def fit_ridge_models(train_epochs, val_epochs, interpolations, neighbour_table, ch_names, alphas=ALPHAS):
    """
    Fits one ridge model per target channel on the input patches of train_epochs (every window of settings.PATCH_WINDOWS,
    as LaBraM trained): the channel predicted from the channels visible when its cluster (neighbour_table[c]) is hidden,
    every time sample of the trials where it is not interpolated one row, the inputs standardized. Every relative alpha
    is scaled by the number of rows (the penalty relative to the standardized data, as RIDGE_ALPHA of ridge_leakage.py),
    the one with the best mean validation R² over the channels (val_epochs) is kept.
    Returns ({channel index: fitted pipeline}, sweep: train and validation R² of every alpha and channel).
    """
    X_train, allowed_train = stacked_patches(train_epochs, interpolations, settings.PATCH_WINDOWS, settings.EPOCH_WINDOW, settings.RSFREQ)
    X_val, allowed_val = stacked_patches(val_epochs, interpolations, settings.PATCH_WINDOWS, settings.EPOCH_WINDOW, settings.RSFREQ)
    models, sweep = {}, []
    for c in range(settings.N_CHANNELS):
        visible = ~neighbour_table[c]

        rows = X_train[allowed_train[:, c]].transpose(0, 2, 1).reshape(-1, settings.N_CHANNELS)
        rows_val = X_val[allowed_val[:, c]].transpose(0, 2, 1).reshape(-1, settings.N_CHANNELS)
        for rel_alpha in alphas:
            model = make_pipeline(StandardScaler(), Ridge(alpha=rel_alpha * len(rows))).fit(rows[:, visible], rows[:, c])
            models[rel_alpha, c] = model
            sweep.append({"alpha": rel_alpha, "channel": ch_names[c],
                          "train_r2": model.score(rows[:, visible], rows[:, c]),
                          "val_r2": model.score(rows_val[:, visible], rows_val[:, c])})
    sweep = pd.DataFrame(sweep)
    val_r2 = sweep.groupby("alpha")["val_r2"].mean()
    best_alpha = val_r2.idxmax()
    print(f"Ridge: relative alpha {best_alpha:g}, mean validation R² over channels {val_r2[best_alpha]:.3f}")
    return {c: models[best_alpha, c] for c in range(settings.N_CHANNELS)}, sweep


def predict_channels(models, x, neighbour_table):
    """Every channel c of x (..., channels, times) predicted by models[c] from the channels visible when its cluster (neighbour_table[c]) is hidden."""
    samples = np.moveaxis(x, -1, -2).reshape(-1, x.shape[-2])  # rows = time samples
    predicted = np.empty_like(x)
    for c, model in models.items():
        predicted[..., c, :] = model.predict(samples[:, ~neighbour_table[c]]).reshape(*x.shape[:-2], x.shape[-1])
    return predicted


def run_injection(models, dataset: InjectionDataset, gains, ch_names, bands, neighbour_table) -> pd.DataFrame:
    loader = torch_utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=torch_utils.data.SequentialSampler(dataset))
    window = settings.METRIC_WINDOWS["analysis"]["post"]
    rows = []
    for i, (inputs, targets, metadata) in enumerate(loader):
        predicted = predict_channels(models, inputs.numpy(), neighbour_table)  # (trials, versions, channels, times)
        output = torch.stack([torch.stack([dataset.wavelet_target(torch.from_numpy(version)) for version in trial])
                              for trial in predicted])  # dB, (trials, versions, channels, freqs, time bins)
        output = core_utils.cut_to_timewindow(output, dataset.patch_window, window, dataset.target_fs)
        targets = core_utils.cut_to_timewindow(targets, dataset.patch_window, window, dataset.target_fs)
        rows += score_trials(output, targets, metadata, gains, ch_names, bands)

    return pd.DataFrame(rows)

def get_args():
    """Parses the command line arguments of the injection test."""
    parser = argparse.ArgumentParser(description='Run the injection test with a trained model')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model architecture to use for inference')
    parser.add_argument('--injection', type=float, nargs="+", default=None,
                        help='gains of the injected source, e.g. --injection 0.1 0.25 0.5 (10/25/50 %% stronger after onset); '
                             'with --mixed_injection factors of all its gains (default 1 = the gains as listed)')
    parser.add_argument('--head_model', choices=["fsaverage", "sphere"], default="fsaverage",
                        help='head model of the injected source: fsaverage template (downloads ~200 MB on first use) or an offline 3-shell sphere')
    parser.add_argument('--mixed_injection', action='store_true', default=False,
                        help='inject the settings.MIXED_INJECTION_GAINS sources at once, each with its own gain. Needs --head_model fsaverage')
    parser.add_argument('--save_tag_name', type=str, default=None,)
    parser.add_argument('--n_folds', type=int, default=1,
                        help='number of folds to test (the first n), as labram_inj_test.py, whose runs test fold 1 by default')

    parser.add_argument('--split_seed', type=int, nargs='+', default=None,
                        help='also compare different trials, as two sessions are compared: per seed, half A original vs half B '
                             'injected and the null (half B original), stratified by subject and condition (summarize_split); '
                             'csvs injection_split{seed}[_null].csv, summary_split{seed}[_null].csv and topomaps in '
                             'split{seed}[_null]/. Works with --from_csv')
    opts = parser.parse_args()
    if opts.injection is None:
        if not opts.mixed_injection:
            parser.error("--injection is required without --mixed_injection")
        opts.injection = [1.0]  # the MIXED_INJECTION_GAINS as they are, unused with --from_csv (the gains are in the csv)
    opts.from_csv = False  # io.load_run_settings checks the target settings unless from_csv
    return opts

def main(opts):
    print(f"Running injection test with {opts.model_name} model")
    model_dir = settings.FINETUNE_OUTPUT_DIR / opts.model_name
    run_settings = io.load_run_settings(model_dir, opts, settings.SPECTRAL_DECODER_PARAMS["freq_window"], settings.TARGET_POOL)
    print(f"Run settings: {run_settings}")
    masked, k_folds = run_settings["masked"], run_settings["k_folds"]

    if platform.system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath  # checkpoints saved on Linux contain PosixPath objects

    # the baseline test sets are rebuilt from the keys saved during training
    print("Loading training split keys and building baseline inference folds...")
    splitter = KFoldSplitter(input_data=settings.FINETUNE_INPUT_EPOCHS_DIR,
                             target_data_dir=settings.FINETUNE_WAVELET_DIR,
                             target_mode="wavelet", seed=settings.SEED, k_folds=k_folds,
                             no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
    folds = splitter.split_from_keys(io.load_training_split_keys(model_dir, k_folds))[:opts.n_folds]

    # the injected sources and the channels where their net pattern is strongest
    topographies, relative_gains, description = core_utils.injected_sources(
        opts.head_model, settings.INJECTION_LABELS, settings.MIXED_INJECTION_GAINS if opts.mixed_injection else None,
        settings.INJECTION_SOURCE_DEPTH, settings.RSFREQ, settings.EGI_TO_1020_MAP, settings.CHORDER_STANDARD)
    names = list(topographies)
    G = np.stack([topographies[name] for name in names], axis=1)
    relative = np.array([relative_gains[name] for name in names])
    gain_sets = [gain * relative for gain in opts.injection]

    ch_names = settings.CHORDER_STANDARD
    # cut off "-REF" and "EEG " from channel names for compatibility with get_input_chans function
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]
    roi = [ch_names[c] for c in np.argsort(-np.abs(G @ relative))[:settings.INJECTION_N_ROI]]
    print(f"Injected source: {description}, strongest at {', '.join(roi)}")
    # one folder per injection, so single and mixed runs of the same model keep their own results
    injection_name = f"{'mixed' if opts.mixed_injection else 'single'}_{opts.head_model}"
    if opts.save_tag_name is not None:
        injection_name += opts.save_tag_name

    plot_dir = PLOT_ROOT / opts.model_name / injection_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    info = create_topomap_info()  # 10-20 labels at the positions of the recording HydroCel electrodes
    # name of a single source in the plot titles: its aparc labels, or the sphere-model dipole
    single_name = " + ".join(
        settings.INJECTION_LABELS) if opts.head_model == "fsaverage" else "occipital dipole (sphere model)"
    plot_sources(topographies, relative_gains, info, ch_names, plot_dir, source_names={"source": single_name})
    # every bin of the frequency window and the settings.FREQ_BANDS bands, as the bands of ridge_leakage.py
    bands = {f"all bands ({settings.FREQ_WINDOW_LABEL})": np.arange(len(settings.FREQS)), **settings.FREQ_BAND_INDICES}

    # masked-channel reconstruction: interpolated channels are never targets, a target is hidden with its cluster
    trial_interpolations = None
    neighbour_table = None
    if masked:
        trial_interpolations = load_interpolated_channels(settings.FINETUNE_DIR / "epochs_manifest.csv")
        neighbour_table = core_utils.build_neighbor_mask(n_neighbours=run_settings["n_neighbors"], n_chans=settings.N_CHANNELS)
        print(f"Masked-channel reconstruction: every target channel is hidden with its {run_settings['n_neighbors']} nearest electrodes"
              + (", interpolated channels are hidden from the input" if run_settings["hide_interpolated"] else ""))
    sweeps, all_trial_errors = [], []
    for i, fold in enumerate(folds):
        # fit model on training
        fold_dir = model_dir / str(i+1)
        # get training stats for the target normalization of the injected sources, as the model was trained on normalized targets


        models, sweep = fit_ridge_models(fold.training_epochs, fold.validation_epochs, trial_interpolations, neighbour_table, ch_names)
        sweep.insert(0, "fold", i + 1)
        sweeps.append(sweep)
        test = InjectionDataset(fold.testing_epochs, settings.EPOCH_WINDOW, settings.PATCH_WINDOWS["post"], settings.RSFREQ,
                                G, gain_sets, settings.INJECTION_ONSET_RAMP, settings.FREQS, settings.PADDING,
                                settings.WAVELET_PARAMS, pool=settings.TARGET_POOL, interpolations=trial_interpolations)


        trial_errors = run_injection(models, test, opts.injection, ch_names, bands, neighbour_table)

        trial_errors.insert(0, "fold", i + 1)
        all_trial_errors.append(trial_errors)

    out_dir = OUTPUT_ROOT / opts.model_name / injection_name
    trial_errors = pd.concat(all_trial_errors, ignore_index=True)
    per_channel, summary = summarize(trial_errors, roi)
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_errors.to_csv(out_dir / "trial_errors.csv", index=False)
    per_channel.to_csv(out_dir / "injection.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # the plot titles name the injected regions: the aparc labels (both hemispheres) with the gains actually applied
    if opts.mixed_injection:
        gain_label, gain_file = (lambda gain: f"mixed ×{gain:g}"), (lambda gain: f"mixed{gain:g}")
        change_description = lambda gain: (f"the {gain_label(gain)} injection ("
                                           + ", ".join(
                    f"{name} {100 * gain * g:+g} %" for name, g in relative_gains.items()) + ")")
    else:
        gain_label, gain_file = (lambda gain: f"{100 * gain:+g} %"), (lambda gain: f"gain{gain:g}")
        source_name = f"{single_name} source (both hemispheres)" if opts.head_model == "fsaverage" else single_name
        change_description = lambda gain: f"a {gain_label(gain)} gain of the {source_name}"
    error_description = f"Ridge masked reconstruction error change ({run_settings['n_neighbors']} neighbours hidden)"
    plot_injection(per_channel, info, ch_names, gain_file, change_description, error_description, plot_dir)

    print(f"\nInjection test of {opts.model_name}, {trial_errors.fold.nunique()} fold(s), source: {description}")
    print(f"Signal and error change in % on {', '.join(roi)}, median [quartiles] over subjects:")
    print_summary(summary, gain_label)

    # split halves (--split_seed): half A original vs half B injected, and the null (half B original), every seed with
    # its own csvs and topomaps (plot_dir/split{seed}, plot_dir/split{seed}_null)
    for seed in opts.split_seed or []:
        for null in (False, True):
            name = f"split{seed}" + ("_null" if null else "")
            split_channel, split_summary = summarize_split(trial_errors, roi, seed, null)
            split_channel.to_csv(out_dir / f"injection_{name}.csv", index=False)
            split_summary.to_csv(out_dir / f"summary_{name}.csv", index=False)
            split_dir = plot_dir / name
            split_dir.mkdir(parents=True, exist_ok=True)
            split_description = ((lambda
                                      gain: f"no injection (null test: half B uninjected relative to half A uninjected, split seed {seed})") if null
                                 else (lambda
                                           gain: f"{change_description(gain)} in half B (injected), relative to half A (uninjected), split seed {seed}"))
            plot_injection(split_channel, info, ch_names, gain_file, split_description, error_description, split_dir)
            print(
                f"\nSplit seed {seed}, {'null test (no injection): half B uninjected' if null else 'half B injected'} relative to "
                f"half A uninjected (B / A - 1), in % on "
                f"{', '.join(roi)}, median [quartiles] over subjects:")
            print_summary(split_summary, gain_label)
    print(f"Results saved to {out_dir}, topomaps to {plot_dir}")


    (OUTPUT_ROOT / opts.model_name).mkdir(parents=True, exist_ok=True)
    pd.concat(sweeps).to_csv(OUTPUT_ROOT / opts.model_name / "alpha_sweep.csv", index=False)





if __name__ == '__main__':
    opts = get_args()
    main(opts)