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
    signed error change = change of the mean signed error (output - target) of the dB targets, the part of the target's
                    dB level change (level change) the reconstruction doesn't follow: 0 if it follows, -level change if it
                    ignores it. Linear in the target, so unlike the squared error it is not changed by a change of the
                    targets' spread. Both are dB differences, i.e. log power ratios, shown in % like the others: the level
                    change is the change of the geometric mean power, the signed error change how much lower (< 0) or
                    higher the reconstructed power change is than the target's (e.g. -16.7 % if a +20 % change is ignored)
per subject the mean over the settings.INJECTION_N_ROI channels where the injected pattern is strongest, then median and
quartiles over subjects. An error change much smaller than the signal change means the masked prediction absorbs the gain change.
--mixed_injection injects the settings.MIXED_INJECTION_GAINS sources at once (fsaverage), each --injection value scaling all gains.
--predictor fixed_mean / refit_mean scores trial-mean predictors in place of the model, on the same trials, bins and plots
(run_mean_predictor): they bracket the model, fixed_mean ignores the injection and refit_mean follows its average change, so
an error change of the model like refit_mean's comes from the targets' spread in dB, not from the reconstruction.

Outputs (outputs/labram_injection/<model_name>/<injection>/, <injection> = {single,mixed}_{head model}): trial_errors.csv (one row per fold, trial, channel, band and gain),
injection.csv (per gain, band, subject and channel), summary.csv (per gain and band, median [quartiles] over subjects).
Topomaps (outputs/plots/labram_injection/<model_name>/<injection>/): injected_source_topographies.png and, per gain, one row of bands
of the median change over subjects per channel: topomaps_injection_signal_*.png, topomaps_injection_error_*.png and
topomaps_injection_signed_*.png.
--from_csv recomputes the summary, the topomaps and the printout from the trial_errors.csv of an earlier run with the same
--mixed_injection, --head_model, --save_tag_name and --predictor, without the model or the epochs.
--split_seed compares different trials as well, as two sessions are compared (summarize_split): half A of every
subject's trials of every condition original vs half B injected, and the null (half B original), per seed.
"""
import argparse
import json
import random
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
import platform
import pathlib

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
            output, _ = reconstruct_batch(model, eeg, input_chans, neighbour_table, interpolated)
            output = dataset.target_stats.denormalize(output).cpu().unflatten(0, (n_trials, n_versions))  # back to dB, back to batch, versions

            output = core_utils.cut_to_timewindow(output, dataset.patch_window, window, dataset.target_fs)
            targets = core_utils.cut_to_timewindow(targets, dataset.patch_window, window, dataset.target_fs)
            rows += score_trials(output, targets, metadata, gains, ch_names, bands)
    return pd.DataFrame(rows)


def run_mean_predictor(dataset: InjectionDataset, gains: List[float], ch_names, bands: Dict[str, np.ndarray],
                       refit=False) -> pd.DataFrame:
    """
    The trial-mean predictors that bracket the model, scored as run_injection scores the model: a subject's target
    channel is predicted by its mean dB target over the subject's trials in which the channel is not interpolated, per
    frequency and time bin, without looking at the input. The means are computed from the scored trials themselves.
    refit False: the mean of the original targets predicts every version, a predictor that ignores the injection.
    refit True: every version is predicted by its own mean, a predictor that follows the injection's average change
    exactly, so its error change is the change of the targets' spread around their mean.
    """
    window = settings.METRIC_WINDOWS["analysis"]["post"]
    subject_trials = {}
    for i, (_, metadata) in enumerate(dataset.samples):
        subject_trials.setdefault(metadata["subject"], []).append(i)
    rows = []
    for k, (subject, indices) in enumerate(subject_trials.items()):
        print(f"Scoring the trial-mean predictor on subject {subject} ({k + 1}/{len(subject_trials)})")
        loader = torch_utils.data.DataLoader(torch_utils.data.Subset(dataset, indices), batch_size=len(indices))
        _, targets, metadata = next(iter(loader))
        targets = core_utils.cut_to_timewindow(targets, dataset.patch_window, window, dataset.target_fs)
        # (versions, channels, freqs, time bins) mean over the trials in which the channel is not interpolated
        allowed = metadata["allowed"][:, None, :, None, None].float()
        means = (targets * allowed).sum(0) / allowed.sum(0).clamp(min=1)
        prediction = means if refit else means[:1].expand_as(means)
        rows += score_trials(prediction.expand_as(targets), targets, metadata, gains, ch_names, bands)
    return pd.DataFrame(rows)


def score_trials(output, targets, metadata, gains: List[float], ch_names, bands: Dict[str, np.ndarray]) -> List[Dict]:
    """
    One row per trial, allowed channel, band and gain of a batch, over the band's frequency bins, original (version 0)
    and injected (version g + 1): the summed squared error and the summed signed error (output - target) of output (dB,
    cut to the analysis window like targets, (trials, versions, channels, freqs, time bins)), the summed linear power
    of the target and the summed dB target (its level).
    """
    # sums over time per frequency, then over every band's frequencies: (trials, versions, channels)
    sse_freqs = ((output - targets) ** 2).sum(-1).numpy()
    signed_freqs = (output - targets).sum(-1).numpy()
    power_freqs = (10 ** (targets / 10)).sum(-1).numpy()  # linear power of the dB targets
    level_freqs = targets.sum(-1).numpy()
    sse = {band: sse_freqs[..., indices].sum(-1) for band, indices in bands.items()}
    signed = {band: signed_freqs[..., indices].sum(-1) for band, indices in bands.items()}
    power = {band: power_freqs[..., indices].sum(-1) for band, indices in bands.items()}
    level = {band: level_freqs[..., indices].sum(-1) for band, indices in bands.items()}
    n_times = targets.shape[-1]

    rows = []
    allowed = metadata["allowed"].numpy()
    for j in range(len(allowed)):
        for c in np.flatnonzero(allowed[j]):
            for band, indices in bands.items():
                for g, gain in enumerate(gains):
                    rows.append({"subject": metadata["subject"][j], "condition": metadata["condition"][j],
                                 "id": int(metadata["id"][j]), "channel": ch_names[c], "band": band, "gain": gain,
                                 "n_bins": len(indices) * n_times,
                                 "sse_original": sse[band][j, 0, c], "sse_injected": sse[band][j, g + 1, c],
                                 "signed_original": signed[band][j, 0, c], "signed_injected": signed[band][j, g + 1, c],
                                 "power_original": power[band][j, 0, c], "power_injected": power[band][j, g + 1, c],
                                 "level_original": level[band][j, 0, c], "level_injected": level[band][j, g + 1, c]})
    return rows


def summarize(trial_errors: pd.DataFrame, roi: List[str]):
    """
    Returns (per gain, band, subject and channel: error and signal change as log ratios of the sums over the subject's
    trials, i.e. log(MSE injected / MSE original) and log(power injected / power original), as the bins are the same,
    and the change of the mean signed error and of the mean dB target (level), dB differences as log ratios too;
    per gain and band: median and quartiles over subjects of the subject's mean change over the roi channels, in %).
    The signed error change is the level change the reconstruction doesn't follow: 0 if it follows the injected change,
    -level change if it ignores it. Unlike the squared error it doesn't change when only the targets' spread changes.
    """
    keys = ["gain", "band", "subject", "channel"]
    sums = trial_errors.groupby(keys, sort=False)[["sse_original", "sse_injected", "signed_original", "signed_injected", "power_original",
                                                   "power_injected", "level_original", "level_injected", "n_bins"]].sum()
    per_channel = pd.DataFrame({"error_change": np.log(sums.sse_injected / sums.sse_original),
                                "signal_change": np.log(sums.power_injected / sums.power_original),
                                "signed_error_change": db_to_log((sums.signed_injected - sums.signed_original) / sums.n_bins),
                                "level_change": db_to_log((sums.level_injected - sums.level_original) / sums.n_bins)}).reset_index()
    return per_channel, summarize_per_channel(per_channel, roi)


def split_half_b(trial_errors: pd.DataFrame, seed: int) -> pd.Series:
    """
    Random halves of the trials, stratified by subject and condition: True for the rows of half B. Every subject's trial
    ids of every condition are sorted, shuffled with random.Random(seed) and the second half is B, so the same seed
    gives the same halves in every run (and as the occlusion split in notebooks/decoding_injections.ipynb).
    """
    rng = random.Random(seed)
    half_b = set()
    trials = trial_errors[["subject", "condition", "id"]].drop_duplicates()
    for (subject, condition), group in trials.groupby(["subject", "condition"], sort=True):
        ids = sorted(group["id"])
        rng.shuffle(ids)
        half_b |= {(subject, condition, idx) for idx in ids[len(ids) // 2:]}
    return pd.Series(pd.MultiIndex.from_frame(trial_errors[["subject", "condition", "id"]]).isin(list(half_b)), index=trial_errors.index)


def summarize_split(trial_errors: pd.DataFrame, roi: List[str], seed: int, null: bool = False):
    """
    As summarize, but comparing different trials of the same session, as two sessions are compared: the original trials
    of half A (split_half_b with seed) with the injected trials of half B, or with null its original trials (the
    difference two halves of the same trials have by chance). The changes are those of the means per bin over the
    half's trials, as the halves can differ by a trial. Returns (per_channel, summary) like summarize.
    """
    half_b = split_half_b(trial_errors, seed)
    keys = ["gain", "band", "subject", "channel"]
    version = "original" if null else "injected"
    a = trial_errors[~half_b].groupby(keys, sort=False)[["sse_original", "signed_original", "power_original", "level_original", "n_bins"]].sum()
    b = trial_errors[half_b].groupby(keys, sort=False)[[f"sse_{version}", f"signed_{version}", f"power_{version}", f"level_{version}", "n_bins"]].sum()
    a, b = a.align(b, join="inner", axis=0)
    per_channel = pd.DataFrame({"error_change": np.log((b[f"sse_{version}"] / b.n_bins) / (a.sse_original / a.n_bins)),
                                "signal_change": np.log((b[f"power_{version}"] / b.n_bins) / (a.power_original / a.n_bins)),
                                "signed_error_change": db_to_log(b[f"signed_{version}"] / b.n_bins - a.signed_original / a.n_bins),
                                "level_change": db_to_log(b[f"level_{version}"] / b.n_bins - a.level_original / a.n_bins)}).reset_index()
    return per_channel, summarize_per_channel(per_channel, roi)


def summarize_per_channel(per_channel: pd.DataFrame, roi: List[str]) -> pd.DataFrame:
    """Per gain and band: median and quartiles over subjects (in %) of every subject's mean change over the roi channels."""
    per_subject = per_channel[per_channel.channel.isin(roi)].groupby(["gain", "band", "subject"], sort=False)[
        ["signal_change", "error_change", "level_change", "signed_error_change"]].mean()
    stats = ["median", lambda v: v.quantile(0.25), lambda v: v.quantile(0.75)]
    kinds = ["signal_change", "error_change", "level_change", "signed_error_change"]
    summary = per_subject[kinds].groupby(["gain", "band"], sort=False).agg(stats).apply(pct)
    summary.columns = [f"{kind}_{stat}" for kind in kinds for stat in ("median_pct", "q1_pct", "q3_pct")]
    summary["n_subjects"] = per_subject.groupby(["gain", "band"], sort=False).size()
    return summary.reset_index()


def pct(log_ratio):
    """Converts a log ratio to a change in %."""
    return 100 * (np.exp(log_ratio) - 1)


def db_to_log(db):
    """Converts a power difference in dB (of the dB targets, 10 * log10 of the power) to a log ratio."""
    return db * np.log(10) / 10


def plot_injection(per_channel: pd.DataFrame, info, ch_names, gain_file, change_description, error_description, plot_dir: Path):
    """
    Topomaps of the median change over subjects per channel, one row of bands per gain and kind, as in ridge_leakage.py:
    target channel power change (topomaps_injection_signal_*) and reconstruction error change (topomaps_injection_error_*)
    and the signed error change (topomaps_injection_signed_*), in %.
    """
    for gain, part in per_channel.groupby("gain", sort=False):
        n_subjects = part.subject.nunique()
        for kind, what in [("signal", "Target channel power change"), ("error", error_description)]:
            values = {band: pct(rows.groupby("channel")[f"{kind}_change"].median()) for band, rows in part.groupby("band", sort=False)}
            lim = symmetric_limit(values, default=1.0)
            plot_topomap_row(values, info, ch_names, (-lim, lim), "RdBu_r", "change (%)",
                             f"{what} after {change_description(gain)} (median over N={n_subjects} subjects)",
                             plot_dir / f"topomaps_injection_{kind}_{gain_file(gain)}.png")
        values = {band: pct(rows.groupby("channel")["signed_error_change"].median()) for band, rows in part.groupby("band", sort=False)}
        lim = symmetric_limit(values, default=1.0)
        plot_topomap_row(values, info, ch_names, (-lim, lim), "RdBu_r", "change (%)",
                         f"Signed error change (output - target) after {change_description(gain)} (median over N={n_subjects} subjects)",
                         plot_dir / f"topomaps_injection_signed_{gain_file(gain)}.png")


def plot_sources(topographies: Dict[str, np.ndarray], relative_gains: Dict[str, float], info, ch_names, plot_dir: Path,
                 source_names: Optional[Dict[str, str]] = None):
    """
    Topomaps of the injected sources (and their net pattern for a mixed injection), as in ridge_leakage.py.
    source_names: panel title of a source in place of its key, e.g. {"source": "pericalcarine"} for a single injection.
    """
    source_names = source_names or {}
    values = {source_names.get(name, name): pd.Series(g, index=ch_names) for name, g in topographies.items()}
    if len(topographies) > 1:
        net = sum(relative_gains[name] * g for name, g in topographies.items())
        values["net change (injection 1)"] = pd.Series(net / np.abs(net).max(), index=ch_names)
        cosine, condition = core_utils.source_overlap(topographies)
        print(f"Overlap of the source topographies (cosine similarity):\n{cosine.round(2)}\nCondition number: {condition:.1f}")
    plot_topomap_row(values, info, ch_names, (-1, 1), "RdBu_r", "relative potential",
                     "Injected source topographies (23 model channels, average reference over the 65 HydroCel electrodes)",
                     plot_dir / "injected_source_topographies.png")


def load_run_settings(model_dir: Path, opts) -> Dict[str, Any]:
    """
    Returns the settings of the training run the test has to match ({"masked", "n_neighbors", "k_folds", "hide_interpolated"})
    from the run's metadata.json, hide_interpolated False for runs that don't record it. Only wavelet models are supported.
    """
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        hyperparameters = json.load(f)["hyperparameters"]

    if hyperparameters.get("target_mode") is None:
        print(f"WARNING: target_mode is None in training run settings,using CLI provided target mode {opts.target_mode} as fallback.")
        hyperparameters["target_mode"] = opts.target_mode
    if hyperparameters.get("k_folds") is None:
        print(f"WARNING: k_folds is None in training run settings,using CLI provided k_folds {opts.k_folds} as fallback.")
        hyperparameters["k_folds"] = opts.k_folds

    if hyperparameters.get("target_mode") != "wavelet":
        raise ValueError(f"Only wavelet models are supported, the run's target mode is {hyperparameters.get('target_mode')}")
    # the targets are computed with the current settings: a different frequency window keeps the output shape, so the
    # model's rows would silently be compared with other frequencies. Runs that don't record these are not checked
    if not opts.from_csv:
        current = {"freq_window": list(settings.SPECTRAL_DECODER_PARAMS["freq_window"]), "target_pool": settings.TARGET_POOL}
        recorded = {name: hyperparameters[name] for name in current if name in hyperparameters}
        mismatched = {name: (value, current[name]) for name, value in recorded.items() if list(np.atleast_1d(value)) != list(np.atleast_1d(current[name]))}
        if mismatched:
            raise ValueError(f"The run was trained with other target settings than the current ones (trained, current): {mismatched}")
        if len(recorded) < len(current):
            print(f"WARNING: the run records no {', '.join(set(current) - set(recorded))}, the current settings are assumed to match")
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
    parser.add_argument('--target_mode', type=str, help='user-provided target mode to use as fallback if training did not log target_mode')
    parser.add_argument('--k_folds', type=int, default=6, help='user-provided number of folds for training as fallback')
    parser.add_argument('--save_tag_name', type=str, default=None,)
    parser.add_argument('--predictor', choices=["model", "fixed_mean", "refit_mean"], default="model",
                        help='what reconstructs the targets: the model, or a trial-mean predictor that brackets it (run_mean_predictor): '
                             'fixed_mean ignores the injection, refit_mean follows its average change. The predictors are '
                             'saved next to the model results, with _{predictor} added to the injection folder')
    parser.add_argument('--from_csv', action='store_true', default=False,
                        help='only recompute the summary and the topomaps from the trial_errors.csv of an earlier run in the same '
                             'injection folder (same --mixed_injection, --head_model, --save_tag_name and --predictor)')
    parser.add_argument('--split_seed', type=int, nargs='+', default=None,
                        help='also compare different trials, as two sessions are compared: per seed, half A original vs half B '
                             'injected and the null (half B original), stratified by subject and condition (summarize_split); '
                             'csvs injection_split{seed}[_null].csv, summary_split{seed}[_null].csv and topomaps in '
                             'split{seed}[_null]/. Works with --from_csv')
    opts = parser.parse_args()
    if opts.injection is None:
        if not opts.mixed_injection and not opts.from_csv:
            parser.error("--injection is required without --mixed_injection")
        opts.injection = [1.0]  # the MIXED_INJECTION_GAINS as they are, unused with --from_csv (the gains are in the csv)
    return opts


def main(opts):
    print(f"Running injection test with {opts.model_name} model")
    model_dir = settings.FINETUNE_OUTPUT_DIR / opts.model_name
    run_settings = load_run_settings(model_dir, opts)
    print(f"Run settings: {run_settings}")
    masked, k_folds = run_settings["masked"], run_settings["k_folds"]

    if platform.system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath  # checkpoints saved on Linux contain PosixPath objects


    # the baseline test sets are rebuilt from the keys saved during training (none with --from_csv, nothing is run)
    folds = []
    if not opts.from_csv:
        print("Loading training split keys and building baseline inference folds...")
        splitter = KFoldSplitter(input_data=settings.FINETUNE_INPUT_EPOCHS_DIR, target_data_dir=settings.FINETUNE_WAVELET_DIR,
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
    if opts.save_tag_name is not None:
        injection_name += opts.save_tag_name
    if opts.predictor != "model":
        injection_name += f"_{opts.predictor}"
    plot_dir = PLOT_ROOT / opts.model_name / injection_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    info = create_topomap_info()  # 10-20 labels at the positions of the recording HydroCel electrodes
    # name of a single source in the plot titles: its aparc labels, or the sphere-model dipole
    single_name = " + ".join(settings.INJECTION_LABELS) if opts.head_model == "fsaverage" else "occipital dipole (sphere model)"
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
        if opts.predictor != "model":
            trial_errors = run_mean_predictor(dataset, opts.injection, ch_names, bands, refit=opts.predictor == "refit_mean")
            trial_errors.insert(0, "fold", i + 1)
            all_trial_errors.append(trial_errors)
            continue

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

    out_dir = OUTPUT_ROOT / opts.model_name / injection_name
    if opts.from_csv:
        trial_errors = pd.read_csv(out_dir / "trial_errors.csv")
        # earlier runs named the band of all bins by its range only, e.g. "5-60 Hz"
        trial_errors["band"] = trial_errors.band.str.replace(r"^(\d+-\d+ Hz)$", r"all bands (\1)", regex=True)
        print(f"Loaded the trial errors of {trial_errors.fold.nunique()} fold(s) from {out_dir / 'trial_errors.csv'}")
    else:
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
                                           + ", ".join(f"{name} {100 * gain * g:+g} %" for name, g in relative_gains.items()) + ")")
    else:
        gain_label, gain_file = (lambda gain: f"{100 * gain:+g} %"), (lambda gain: f"gain{gain:g}")
        source_name = f"{single_name} source (both hemispheres)" if opts.head_model == "fsaverage" else single_name
        change_description = lambda gain: f"a {gain_label(gain)} gain of the {source_name}"
    error_description = (f"Masked reconstruction error change ({run_settings['n_neighbors']} neighbours hidden)" if masked
                         else "Reconstruction error change")
    if opts.predictor != "model":
        error_description = {"fixed_mean": "Error change of the trial mean of the original targets",
                             "refit_mean": "Error change of the trial mean refitted per version"}[opts.predictor]
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
            split_description = ((lambda gain: f"no injection (null test: half B uninjected relative to half A uninjected, split seed {seed})") if null
                                 else (lambda gain: f"{change_description(gain)} in half B (injected), relative to half A (uninjected), split seed {seed}"))
            plot_injection(split_channel, info, ch_names, gain_file, split_description, error_description, split_dir)
            print(f"\nSplit seed {seed}, {'null test (no injection): half B uninjected' if null else 'half B injected'} relative to "
                  f"half A uninjected (B / A - 1), in % on "
                  f"{', '.join(roi)}, median [quartiles] over subjects:")
            print_summary(split_summary, gain_label)
    print(f"Results saved to {out_dir}, topomaps to {plot_dir}")


def print_summary(summary: pd.DataFrame, gain_label):
    """Prints a summary (summarize, summarize_split) row by row: median [quartiles] over subjects of every change, in %."""
    for _, row in summary.iterrows():
        print(f"  {gain_label(row.gain):>12} {row.band:16s} signal {row.signal_change_median_pct:+6.1f} % "
              f"[{row.signal_change_q1_pct:+.1f}, {row.signal_change_q3_pct:+.1f}], error {row.error_change_median_pct:+6.1f} % "
              f"[{row.error_change_q1_pct:+.1f}, {row.error_change_q3_pct:+.1f}], level {row.level_change_median_pct:+6.1f} %, "
              f"signed error {row.signed_error_change_median_pct:+6.1f} % [{row.signed_error_change_q1_pct:+.1f}, "
              f"{row.signed_error_change_q3_pct:+.1f}] (N={int(row.n_subjects)} subjects)")


if __name__ == '__main__':
    opts = get_args()
    main(opts)
