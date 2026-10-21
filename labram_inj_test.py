"""
Gain injection test of a trained masked-channel model, the LaBraM counterpart of the injection in ridge_leakage.py.

In a fold's held-out baseline test trials, the activity of a visual-cortex source is made (1 + gain) times stronger after
stimulus onset (core.utils.inject_sources). The wavelet targets of the post-stimulus patch are recomputed from the original
and the injected inputs (core.datasets.InjectionDataset), the model reconstructs both, and both are scored on the analysis
window (settings.POST_TARGET_WINDOW, 0 to 0.4 s), as ridge_leakage.py scores its injection, per band (all bins of the
frequency window and every settings.FREQ_BANDS band):
    error change  = log(SSE injected / SSE original) of the reconstruction, summed over a subject's trials, per channel;
                    the same bins are scored with and without the injection, so this is log(MSE injected / MSE original)
    signal change = log(power injected / power original) of the target (linear power of the dB target), the same way
per subject the mean over the settings.INJECTION_N_ROI channels where the injected pattern is strongest, then median and
quartiles over subjects. An error change much smaller than the signal change means the masked prediction absorbs the gain change.
--mixed_injection injects the settings.MIXED_INJECTION_GAINS sources at once (fsaverage), each --injection value scaling all gains.

Outputs (outputs/labram_injection/<model_name>/<injection>/, <injection> = {single,mixed}_{head model}): trial_errors.csv (one row per fold, trial, channel, band and gain),
injection.csv (per gain, band, subject and channel), summary.csv (per gain and band, median [quartiles] over subjects).
Topomaps (outputs/plots/labram_injection/<model_name>/<injection>/): injected_source_topographies.png and, per gain, one row of bands
of the median change over subjects per channel: topomaps_injection_signal_*.png and topomaps_injection_error_*.png.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from torch import utils as torch_utils

from core.architectures import EEGSpectralDecoder
from core.datasets import KFoldSplitter, InjectionDataset
from core.io import load_interpolated_channels
from core import settings, io
from core.io_models import TargetStats
from libs.labram import utils
from einops import rearrange
from core import utils as core_utils
from core.plot_data_processing import create_topomap_info
from core.plotting import plot_topomap_row, symmetric_limit
from run_inference import reconstruct_batch

OUTPUT_ROOT = settings.BASE_DIR / "outputs" / "labram_injection"
PLOT_ROOT = settings.PLOT_DIR / "labram_injection"
BATCH_SIZE = 32  # trials per batch, each with its original and every injected patch


def run_injection(model, dataset: InjectionDataset, gains: List[float], ch_names, device, bands: Dict[str, np.ndarray],
                  neighbour_table=None, hide_interpolated=False) -> pd.DataFrame:
    """
    Reconstructs every trial of dataset with and without every injection and returns one row per trial, allowed channel,
    band and gain: the summed squared reconstruction error and the summed linear power of the target on the analysis
    window of the post patch, over the band's frequency bins (bands: {band: frequency indices}), original and injected.
    Interpolated channels (metadata["allowed"] False) are left out, and hidden from the input if hide_interpolated.
    """
    loader = torch_utils.data.DataLoader(dataset, sampler=torch_utils.data.SequentialSampler(dataset), batch_size=BATCH_SIZE)
    window = settings.METRIC_WINDOWS["analysis"]["post"]
    rows = []
    with torch.no_grad():
        input_chans = utils.get_input_chans(ch_names)
        for i, (inputs, targets, metadata) in enumerate(loader):
            print(f"Reconstructing trials in batch {i + 1}/{len(loader)}")
            n_trials, n_versions = inputs.shape[:2]  # versions: original, then one per gain
            eeg = inputs.flatten(0, 1).float().to(device, non_blocking=True) / 100 # merge batch and versions
            eeg = rearrange(eeg, 'B N (A T) -> B N A T', T=200)
            # every version of a trial has the trial's interpolated channels
            interpolated = (~metadata["allowed"]).repeat_interleave(n_versions, dim=0).to(device) if hide_interpolated else None
            output = reconstruct_batch(model, eeg, input_chans, eeg.shape[1], neighbour_table, interpolated)
            output = dataset.target_stats.denormalize(output).cpu().unflatten(0, (n_trials, n_versions))  # back to dB, back to batch, versions

            output = core_utils.cut_to_timewindow(output, dataset.patch_window, window, dataset.target_fs)
            targets = core_utils.cut_to_timewindow(targets, dataset.patch_window, window, dataset.target_fs)
            # sums over time per frequency, then over every band's frequencies: (trials, versions, channels)
            sse_freqs = ((output - targets) ** 2).sum(-1).numpy()
            power_freqs = (10 ** (targets / 10)).sum(-1).numpy()  # linear power of the dB targets
            sse = {band: sse_freqs[..., indices].sum(-1) for band, indices in bands.items()}
            power = {band: power_freqs[..., indices].sum(-1) for band, indices in bands.items()}
            n_times = targets.shape[-1]

            allowed = metadata["allowed"].numpy()
            for j in range(n_trials):
                for c in np.flatnonzero(allowed[j]):
                    for band, indices in bands.items():
                        for g, gain in enumerate(gains):
                            rows.append({"subject": metadata["subject"][j], "condition": metadata["condition"][j],
                                         "id": int(metadata["id"][j]), "channel": ch_names[c], "band": band, "gain": gain,
                                         "n_bins": len(indices) * n_times,
                                         "sse_original": sse[band][j, 0, c], "sse_injected": sse[band][j, g + 1, c],
                                         "power_original": power[band][j, 0, c], "power_injected": power[band][j, g + 1, c]})
    return pd.DataFrame(rows)


def summarize(trial_errors: pd.DataFrame, roi: List[str]):
    """
    Returns (per gain, band, subject and channel: error and signal change as log ratios of the sums over the subject's
    trials, i.e. log(MSE injected / MSE original) and log(power injected / power original), as the bins are the same;
    per gain and band: median and quartiles over subjects of the subject's mean change over the roi channels, in %).
    """
    keys = ["gain", "band", "subject", "channel"]
    sums = trial_errors.groupby(keys, sort=False)[["sse_original", "sse_injected", "power_original", "power_injected"]].sum()
    per_channel = pd.DataFrame({"error_change": np.log(sums.sse_injected / sums.sse_original),
                                "signal_change": np.log(sums.power_injected / sums.power_original)}).reset_index()
    per_subject = per_channel[per_channel.channel.isin(roi)].groupby(["gain", "band", "subject"], sort=False)[["signal_change", "error_change"]].mean()
    summary = per_subject.groupby(["gain", "band"], sort=False).agg(["median", lambda v: v.quantile(0.25), lambda v: v.quantile(0.75)]).apply(pct)
    summary.columns = [f"{kind}_{stat}" for kind in ("signal_change", "error_change") for stat in ("median_pct", "q1_pct", "q3_pct")]
    summary["n_subjects"] = per_subject.groupby(["gain", "band"], sort=False).size()
    return per_channel, summary.reset_index()


def pct(log_ratio):
    """Converts a log ratio to a change in %."""
    return 100 * (np.exp(log_ratio) - 1)


def plot_injection(per_channel: pd.DataFrame, info, ch_names, gain_file, change_description, error_description, plot_dir: Path):
    """
    Topomaps of the median change over subjects per channel, one row of bands per gain and kind, as in ridge_leakage.py:
    target channel power change (topomaps_injection_signal_*) and reconstruction error change (topomaps_injection_error_*).
    """
    for gain, part in per_channel.groupby("gain", sort=False):
        n_subjects = part.subject.nunique()
        for kind, what in [("signal", "Target channel power change"), ("error", error_description)]:
            values = {band: pct(rows.groupby("channel")[f"{kind}_change"].median()) for band, rows in part.groupby("band", sort=False)}
            lim = symmetric_limit(values, default=1.0)
            plot_topomap_row(values, info, ch_names, (-lim, lim), "RdBu_r", "change (%)",
                             f"{what} after {change_description(gain)} (median over N={n_subjects} subjects)",
                             plot_dir / f"topomaps_injection_{kind}_{gain_file(gain)}.png")


def plot_sources(topographies: Dict[str, np.ndarray], relative_gains: Dict[str, float], info, ch_names, plot_dir: Path):
    """Topomaps of the injected sources (and their net pattern for a mixed injection), as in ridge_leakage.py."""
    values = {name: pd.Series(g, index=ch_names) for name, g in topographies.items()}
    if len(topographies) > 1:
        net = sum(relative_gains[name] * g for name, g in topographies.items())
        values["net change (injection 1)"] = pd.Series(net / np.abs(net).max(), index=ch_names)
        cosine, condition = core_utils.source_overlap(topographies)
        print(f"Overlap of the source topographies (cosine similarity):\n{cosine.round(2)}\nCondition number: {condition:.1f}")
    plot_topomap_row(values, info, ch_names, (-1, 1), "RdBu_r", "relative potential",
                     "Injected source topographies (23 model channels, average reference over the 65 HydroCel electrodes)",
                     plot_dir / "injected_source_topographies.png")


def load_run_settings(model_dir: Path) -> Dict[str, Any]:
    """
    Returns the settings of the training run the test has to match ({"masked", "n_neighbors", "k_folds", "hide_interpolated"})
    from the run's metadata.json, hide_interpolated False for runs that don't record it. Only wavelet models are supported.
    """
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        hyperparameters = json.load(f)["hyperparameters"]
    if hyperparameters.get("target_mode") != "wavelet":
        raise ValueError(f"Only wavelet models are supported, the run's target mode is {hyperparameters.get('target_mode')}")
    return {**{name: hyperparameters[name] for name in ("masked", "n_neighbors", "k_folds")},
            "hide_interpolated": hyperparameters.get("hide_interpolated", False)}


def load_target_stats(fold_dir: Path) -> Optional[TargetStats]:
    """Returns the target stats saved with a fold (target_stats.json), None if the fold has none."""
    stats_path = fold_dir / "target_stats.json"
    if not stats_path.exists():
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        return TargetStats.from_dict(json.load(f))


def get_args():
    """Parses the command line arguments of the injection test."""
    parser = argparse.ArgumentParser(description='Run the injection test with a trained model')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model architecture to use for inference')
    parser.add_argument('--injection', type=float, nargs="+", default=None,
                        help='gains of the injected source, e.g. --injection 0.1 0.25 0.5 (10/25/50 %% stronger after onset); '
                             'with --mixed_injection factors of all its gains (default 1 = the gains as listed)')
    parser.add_argument('--n_folds', type=int, default=1,
                        help='number of folds to test (the first n), every fold contains every subject')
    parser.add_argument('--head_model', choices=["fsaverage", "sphere"], default="fsaverage",
                        help='head model of the injected source: fsaverage template (downloads ~200 MB on first use) or an offline 3-shell sphere')
    parser.add_argument('--mixed_injection', action='store_true', default=False,
                        help='inject the settings.MIXED_INJECTION_GAINS sources at once, each with its own gain. Needs --head_model fsaverage')
    opts = parser.parse_args()
    if opts.injection is None:
        if not opts.mixed_injection:
            parser.error("--injection is required without --mixed_injection")
        opts.injection = [1.0]  # the MIXED_INJECTION_GAINS as they are
    return opts


def main(opts):
    print(f"Running injection test with {opts.model_name} model")
    model_dir = settings.FINETUNE_OUTPUT_DIR / opts.model_name
    run_settings = load_run_settings(model_dir)
    print(f"Run settings: {run_settings}")
    masked, k_folds = run_settings["masked"], run_settings["k_folds"]

    # the baseline test sets are rebuilt from the keys saved during training
    print("Loading training split keys and building baseline inference folds...")
    splitter = KFoldSplitter(input_data_dir=settings.FINETUNE_INPUT_EPOCHS_DIR, target_data_dir=settings.FINETUNE_WAVELET_DIR,
                             target_mode="wavelet", seed=settings.SEED, k_folds=k_folds, no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
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
    plot_dir = PLOT_ROOT / opts.model_name / injection_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    info = create_topomap_info()  # 10-20 labels at the positions of the recording HydroCel electrodes
    plot_sources(topographies, relative_gains, info, ch_names, plot_dir)
    # every bin of the frequency window and the settings.FREQ_BANDS bands, as the bands of ridge_leakage.py
    bands = {settings.FREQ_WINDOW_LABEL: np.arange(len(settings.FREQS)), **settings.FREQ_BAND_INDICES}

    # masked-channel reconstruction: interpolated channels are never targets, a target is hidden with its cluster
    trial_interpolations = None
    neighbour_table = None
    if masked:
        trial_interpolations = load_interpolated_channels(settings.FINETUNE_DIR / "epochs_manifest.csv")
        neighbour_table = core_utils.build_neighbor_mask(n_neighbours=run_settings["n_neighbors"], n_chans=settings.N_CHANNELS)
        print(f"Masked-channel reconstruction: every target channel is hidden with its {run_settings['n_neighbors']} nearest electrodes"
              + (", interpolated channels are hidden from the input" if run_settings["hide_interpolated"] else ""))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_trial_errors = []
    for i, fold in enumerate(folds):
        print("=" * 30)
        print(f"Running injection test for fold {i + 1}/{len(folds)}")
        fold_dir = model_dir / str(i + 1)
        checkpoint_path = fold_dir / "checkpoint-best_val.pth"
        print("using checkpoint:", checkpoint_path)
        if not checkpoint_path.exists():
            print(f"WARNING: no best checkpoint for fold {i + 1}, skipping it")
            continue

        # the model predicts normalized targets, the stats convert its outputs back to dB
        target_stats = load_target_stats(fold_dir)
        if target_stats is None:
            raise FileNotFoundError(f"Target stats file not found for fold {i + 1}: {fold_dir / 'target_stats.json'}")
        print(f"Target stats: {target_stats}")

        dataset = InjectionDataset(fold.testing_epochs, settings.EPOCH_WINDOW, settings.PATCH_WINDOWS["post"], settings.RSFREQ,
                                   G, gain_sets, settings.INJECTION_ONSET_RAMP, settings.FREQS, settings.PADDING,
                                   settings.WAVELET_PARAMS, pool=settings.TARGET_POOL, interpolations=trial_interpolations,
                                   target_stats=target_stats)

        model = EEGSpectralDecoder(**settings.SPECTRAL_DECODER_PARAMS)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        # set model to eval mode and move to device
        model.to(device)
        model.eval()
        fold_neighbour_table = torch.as_tensor(neighbour_table, device=device) if neighbour_table is not None else None

        trial_errors = run_injection(model, dataset, opts.injection, ch_names, device, bands, fold_neighbour_table,
                                     masked and run_settings["hide_interpolated"])
        trial_errors.insert(0, "fold", i + 1)
        all_trial_errors.append(trial_errors)

    trial_errors = pd.concat(all_trial_errors, ignore_index=True)
    per_channel, summary = summarize(trial_errors, roi)
    out_dir = OUTPUT_ROOT / opts.model_name / injection_name
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_errors.to_csv(out_dir / "trial_errors.csv", index=False)
    per_channel.to_csv(out_dir / "injection.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    if opts.mixed_injection:
        gain_label, gain_file = (lambda gain: f"mixed ×{gain:g}"), (lambda gain: f"mixed{gain:g}")
        change_description = lambda gain: f"the {gain_label(gain)} injection"
    else:
        gain_label, gain_file = (lambda gain: f"{100 * gain:+g} %"), (lambda gain: f"gain{gain:g}")
        change_description = lambda gain: f"a {gain_label(gain)} visual-cortex source gain"
    error_description = (f"Masked reconstruction error change ({run_settings['n_neighbors']} neighbours hidden)" if masked
                         else "Reconstruction error change")
    plot_injection(per_channel, info, ch_names, gain_file, change_description, error_description, plot_dir)

    print(f"\nInjection test of {opts.model_name}, {len(folds)} fold(s), source: {description}")
    print(f"Signal and error change in % on {', '.join(roi)}, median [quartiles] over subjects:")
    for _, row in summary.iterrows():
        print(f"  {gain_label(row.gain):>12} {row.band:16s} signal {row.signal_change_median_pct:+6.1f} % "
              f"[{row.signal_change_q1_pct:+.1f}, {row.signal_change_q3_pct:+.1f}], error {row.error_change_median_pct:+6.1f} % "
              f"[{row.error_change_q1_pct:+.1f}, {row.error_change_q3_pct:+.1f}] (N={int(row.n_subjects)} subjects)")
    print(f"Results saved to {out_dir}, topomaps to {plot_dir}")


if __name__ == '__main__':
    opts = get_args()
    main(opts)
