import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import pywt
import torch
from sklearn.metrics import r2_score
from einops import rearrange
from core.io_models import *
import pandas as pd
import mne
from core import plot_data_processing
import zlib

def subject_paradigm(subject: str, thresh: int):
    return "no_press" if int(subject.removeprefix("STMD")) >= thresh else "press"

def morlet_cycles(freqs, max_cycles, max_reach):
    """
    Returns the number of Morlet wavelet cycles for each frequency: as many as possible up to max_cycles while the wavelet's
    3 SD reach (see wavelet_reach) stays within max_reach seconds, n(f) = min(max_cycles, 2*pi*f*max_reach/3).
    """
    return np.minimum(max_cycles, 2 * np.pi * np.asarray(freqs) * max_reach / 3)


def wavelet_time_sd(freqs, n_cycles):
    """
    Returns the temporal SD (s) of the Gaussian envelope of a Morlet wavelet with n_cycles at each frequency.
    The wavelet effectively uses data within ~3 SD of each time point.
    """
    return np.asarray(n_cycles) / (2 * np.pi * np.asarray(freqs))


def wavelet_reach(freqs, max_cycles, max_reach, n_sd: float = 3):
    """
    Returns how far (s) the wavelet of each frequency reaches from its time point: n_sd SDs of its Gaussian envelope.
    At 3 SD the envelope weight has dropped to ~1% of its peak.
    """
    return n_sd * wavelet_time_sd(freqs, morlet_cycles(freqs, max_cycles, max_reach))


def valid_wavelet_bins(freqs, patch_window, target_window, fs, max_cycles, max_reach, pool: int = 1):
    """
    Returns a (freqs, time) boolean mask over the samples of one patch (patch_window, sampled at fs from its start):
    True where the sample lies in target_window and the wavelet of that frequency reaches no further than the patch edges,
    i.e. the bin is computed from real data of the patch only, not from its reflect padding.
    pool: samples per bin of targets pooled with pool_power, a pooled bin is valid only if all its samples are.
    """
    n_samples = round((patch_window[1] - patch_window[0]) * fs)
    times = patch_window[0] + np.arange(n_samples) / fs
    reach = wavelet_reach(freqs, max_cycles, max_reach)[:, None]
    in_target = (times >= target_window[0] - 1e-9) & (times <= target_window[1] + 1e-9)
    inside_patch = (times - reach >= patch_window[0] - 1e-9) & (times + reach <= patch_window[1] + 1e-9)
    valid = in_target[None, :] & inside_patch
    return valid.reshape(len(freqs), -1, pool).all(axis=-1)


def pool_power(mag, pool: int):
    """
    Pools wavelet magnitudes (..., time) into bins of pool consecutive samples: the magnitude of the mean power of each bin,
    sqrt(mean(mag^2)), so 20*log10 of a pooled bin is the bin's mean power in dB. The time length has to be a multiple of pool.
    """
    if pool == 1:
        return mag
    if mag.shape[-1] % pool:
        raise ValueError(f"Time length {mag.shape[-1]} is not a multiple of the pooling factor {pool}")
    power = mag.reshape(*mag.shape[:-1], -1, pool) ** 2
    return np.sqrt(power.mean(axis=-1))


def calc_wavelet(tensor_data, fs, freqs, pad, max_cycles, max_reach):
    """
    Calculates the complex Morlet wavelet magnitudes of an EEG epoch tensor (channels, time) at freqs and returns them as (freqs, channels, time) along with freqs.
    The number of cycles per frequency follows morlet_cycles (pywt cmorB-1.0 has 2*pi*sqrt(B/2) cycles).
    All wavelets give the same magnitude for a sinusoid, while broadband activity gets ~sqrt(max_cycles / n) times larger magnitudes
    from the wider frequency band of a wavelet with n < max_cycles cycles.
    """
    np_data = tensor_data.numpy()
    # Pad the data to avoid edge effects in the wavelet transform
    if pad > 0:
        np_data = np.pad(np_data, pad_width=((0, 0), (pad, pad)), mode='reflect')

    # Use fewer cycles for low frequencies to keep their wavelets within max_reach
    cycles = morlet_cycles(freqs, max_cycles, max_reach)
    mag = np.empty((len(freqs), *np_data.shape))
    for n in np.unique(cycles):
        idx = np.where(cycles == n)[0]
        wavelet = f"cmor{2 * (n / (2 * np.pi)) ** 2}-1.0"
        # normalize frequencies by sampling frequency
        scales = pywt.frequency2scale(wavelet, freqs[idx] / fs)
        coeffs, _ = pywt.cwt(np_data, scales, wavelet)
        mag[idx] = np.abs(coeffs)

    # Remove the padding from the magnitude array
    if pad > 0:
        mag = mag[:, :, pad:-pad]
    return mag, freqs

def normalize_wavelet(mags, by=Optional[Path], save_path: Optional[Path] = None):
    """
    Normalizes the wavelet magnitudes by the mean and std of the training set, or by the mean and std of the current data if no path is provided.
    Expects mags to be (freqs, channels, time).
    Returns normalized mags and the mean and std used for normalization.
    """
    if by is not None:
        # Load mean and std from file
        stats = np.load(by)
        mean = stats['mean']
        std = stats['std']
    else:
        # Calculate mean and std from current data
        mean = np.mean(mags, axis=(1, 2), keepdims=True)
        std = np.std(mags, axis=(1, 2), keepdims=True)

    normalized_mags = (mags - mean) / (std + 1e-8)  # Add small value to avoid division by zero
    return normalized_mags, mean, std

def cut_to_timewindow(tensor, original_window: Tuple[float, float], new_window: Tuple[float, float], fs:int):
    """
        Cuts the time dimension of a tensor based on the sampling frequency.
        Expects time to be the last dimension.
    """

    from_index =  round((new_window[0] - original_window[0]) * fs)
    to_index = round((new_window[1] - original_window[0]) * fs)

    if from_index < 0 or to_index > tensor.shape[-1]:
        raise ValueError(f"Requested window {new_window} is outside original window {original_window} range.")

    return tensor[..., from_index:to_index]


def cut_to_size(tensor, window_size: int):
    """
    Cuts the input tensor to be divisible by the specified size along the time dimension.
    Assumes the time dimension is the last dimension of the tensor.
    """
    time_points = tensor.shape[-1]
    excess = time_points % window_size
    if excess > 0:
        return tensor[..., :-excess]
    return tensor

def add_prefix_dict_keys(dict, prefix, separator='.'):
    """
    Adds a prefix to all keys in the input dictionary.
    """

    return {f"{prefix}{separator}{key}": value for key, value in dict.items()}

def format_model_input(trial, device):
    """
    Formats single trial tensor input for model inference by adding a batch dimension, rearranging to match the expected input shape of (batch_size, n_chans, n_areas, time_points), and normalizing the values by dividing by 100.
    """
    test_input = trial.unsqueeze(0)
    test_input = rearrange(test_input, 'B N (A T) -> B N A T', T=200)
    test_input = test_input.float().to(device, non_blocking=True) / 100
    return test_input

def reconstruct_model_output(checkpoint_path: Path, model, input, input_chans, **kwargs):
    """
    Reconstructs the model output from a given checkpoint and input tensor, reshaping it according to `n_chans`, `n_freqs`, and `window_size` parameters.

    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        recon = model.get_unflattened_output(x=input, input_chans=input_chans).squeeze(0)



    epoch_idx = checkpoint["epoch"]
    return recon, epoch_idx

def calculate_r2(y_true, y_pred):
    """
    Calculates the R-squared metric for the given true and predicted values, flattened over all dimensions (1D to 3D arrays).
    NaN-aware: only positions where both values are finite are used. Returns NaN if fewer than 2 such positions remain.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")
    if y_true.ndim > 3 or y_pred.ndim > 3:
        raise ValueError(f"Expected y_true and y_pred to be 1D or 2D or 3D arrays, but got shapes {y_true.shape} and {y_pred.shape}.")

    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return np.nan
    return r2_score(y_true[finite], y_pred[finite])


def calculate_bandwise_metrics(y_true, y_pred, signal_freq_stds, recon_freq_stds, band_indices: Dict[str, np.ndarray], eps: float = 1e-8):
    """
    Calculates R-squared, MAE, STD and NMAE metrics for each frequency band by indexing into the frequency dimension of y_true and y_pred.
    Expects band_indices as {band name: indices of the band's frequency bins}, e.g. settings.FREQ_BAND_INDICES.
    Expects y_true and y_pred to be 2D or 3D arrays, where the last dimension corresponds to time points and the second-to-last dimension corresponds to frequencies.
    Expects freq_stds to be a 1D array of standard deviations for each frequency.
    NaN-aware: NaN bins (bins outside the metric window, interpolated channels) are left out, a band without any finite bin gets NaN metrics.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")
    if y_true.ndim not in [2, 3] or y_pred.ndim not in [2, 3]:
        raise ValueError(f"Expected y_true and y_pred to be 2D or 3D arrays, but got shapes {y_true.shape} and {y_pred.shape}.")
    r2_bandwise = {}
    mae_bandwise = {}
    signal_std_bandwise = {}
    recon_std_bandwise = {}
    signal_mean_bandwise = {}
    recon_mean_bandwise = {}
    nmae_bandwise = {}
    for name, indices in band_indices.items():
        if indices.size == 0:
            print(f"WARNING: No frequencies found in band {name}. Skipping metric calculation for this band.")
            continue
        if y_true.ndim == 2:
            band_true = y_true[indices, :].flatten()
            band_pred = y_pred[indices, :].flatten()
        else:
            band_true = y_true[:, indices, :].flatten()
            band_pred = y_pred[:, indices, :].flatten()
        mae = np.nanmean(np.abs(band_true - band_pred))
        r2_bandwise[name] = calculate_r2(band_true, band_pred)
        mae_bandwise[name] = mae
        signal_std_bandwise[name] = np.nanmean(signal_freq_stds[indices])
        recon_std_bandwise[name] = np.nanmean(recon_freq_stds[indices])
        signal_mean_bandwise[name] = np.nanmean(band_true)
        recon_mean_bandwise[name] = np.nanmean(band_pred)
        nmae_bandwise[name] = mae / signal_std_bandwise[name] + eps

    return r2_bandwise, mae_bandwise, signal_std_bandwise, recon_std_bandwise, nmae_bandwise, signal_mean_bandwise, recon_mean_bandwise


def get_inference_metrics(trial_output, trial_target, band_indices: Optional[Dict[str, np.ndarray]] = None, ch_prefix: str = 'ch_', eps: float = 1e-8) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Calculates inference metrics (e.g., R-squared, MAE) for a single trial's output and target data. Expects trial_output and trial_target to be 3D arrays, where the last dimension corresponds to time points.
    band_indices: {band name: indices of the band's frequency bins} for spectral targets, None skips the band-wise metrics.
    NaN-aware: a bin that is NaN in the output or the target (bins outside the metric window, interpolated channels in
    masked-channel reconstruction) is left out of every metric. Metrics without any finite bin, e.g. those of an
    interpolated channel, are NaN.
    level_error: per channel |mean target - mean output| over the window's bins (and over the band's bins for
    level_error_bandwise), i.e. the error of the predicted mean level, in which the unpredictable bin-to-bin fluctuation
    averages out. The trial-level value is the mean over the channels, so errors of different channels do not cancel.
    Returns a dictionary containing the calculated metrics.
    """
    spectral = band_indices is not None
    # leave a bin out of both arrays if it is missing in either, so every metric uses the same bins
    finite = np.isfinite(trial_output) & np.isfinite(trial_target)
    trial_output = np.where(finite, trial_output, np.nan)
    trial_target = np.where(finite, trial_target, np.nan)

    # all-NaN slices (interpolated channels, frequencies without bins in the window) give NaN, without RuntimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        signal_stds = np.nanstd(trial_target, axis=-1)
        recon_stds = np.nanstd(trial_output, axis=-1)
        global_std = np.nanmean(signal_stds)
        recon_std = np.nanmean(recon_stds)

        # global metrics
        g_metrics = Metrics()
        g_metrics.r2 = calculate_r2(trial_target, trial_output)
        g_metrics.mae = np.nanmean(np.abs(trial_target - trial_output))
        g_metrics.nmae = g_metrics.mae / global_std + eps
        g_metrics.signal_volatility = global_std
        g_metrics.signal_mean = np.nanmean(trial_target).item()
        g_metrics.recon_mean = np.nanmean(trial_output).item()
        g_metrics.recon_volatility = recon_std

        # global band-wise metrics
        if spectral:
            signal_freq_stds = np.nanmean(signal_stds, axis=0)
            recon_freq_stds = np.nanmean(recon_stds, axis=0)
            g_metrics.r2_bandwise, g_metrics.mae_bandwise, g_metrics.signal_volatility_bandwise, g_metrics.recon_volatility_bandwise, g_metrics.nmae_bandwise, g_metrics.signal_mean_bandwise, g_metrics.recon_mean_bandwise = (
                calculate_bandwise_metrics(trial_target, trial_output, signal_freq_stds, recon_freq_stds, band_indices=band_indices, eps=eps)
            )
        g_metrics = g_metrics.to_dict()

        ch_metrics_dict = {}
        level_errors, level_errors_bandwise = [], []  # of every channel, averaged into the trial-level metric
        signal_channel_stds = np.nanmean(signal_stds, axis=1) if spectral else signal_stds
        recon_channel_stds = np.nanmean(recon_stds, axis=1) if spectral else recon_stds
        # channel-wise metrics
        for i in range(trial_target.shape[0]):
            ch_key = f'{ch_prefix}{i}'
            ch_metrics = Metrics()
            ch_metrics.r2 = calculate_r2(trial_target[i], trial_output[i])
            ch_metrics.mae = np.nanmean(np.abs(trial_target[i] - trial_output[i]))

            ch_metrics.signal_volatility = signal_channel_stds[i]
            ch_metrics.recon_volatility = recon_channel_stds[i]
            ch_metrics.nmae = ch_metrics.mae / (signal_channel_stds[i] + eps)

            ch_metrics.signal_mean = np.nanmean(trial_target[i]).item()
            ch_metrics.recon_mean = np.nanmean(trial_output[i]).item()
            ch_metrics.level_error = abs(ch_metrics.signal_mean - ch_metrics.recon_mean)
            level_errors.append(ch_metrics.level_error)

            # channel-wise band-wise metrics
            if spectral:
                signal_channel_freq_stds = signal_stds[i, :]
                recon_channel_freq_stds = recon_stds[i, :]
                ch_metrics.r2_bandwise, ch_metrics.mae_bandwise, ch_metrics.signal_volatility_bandwise, ch_metrics.recon_volatility_bandwise, ch_metrics.nmae_bandwise, ch_metrics.signal_mean_bandwise, ch_metrics.recon_mean_bandwise = (
                    calculate_bandwise_metrics(trial_target[i], trial_output[i], signal_channel_freq_stds, recon_channel_freq_stds, band_indices=band_indices, eps=eps)
                )
                ch_metrics.level_error_bandwise = {band: abs(ch_metrics.signal_mean_bandwise[band] - ch_metrics.recon_mean_bandwise[band])
                                                   for band in ch_metrics.signal_mean_bandwise}
                level_errors_bandwise.append(ch_metrics.level_error_bandwise)

            ch_metrics_dict[ch_key] = ch_metrics.to_dict()

        g_metrics["level_error"] = np.nanmean(level_errors)
        if spectral:
            g_metrics["level_error_bandwise"] = {band: np.nanmean([errors[band] for errors in level_errors_bandwise])
                                                 for band in g_metrics["signal_mean_bandwise"]}
    return g_metrics, ch_metrics_dict


def avg_metrics_over_folds(model_data: Dict[str, List[Dict]], metrics: List[str]):
    """
    Averages specified metrics over multiple folds of cross-validation. Expects a list of dictionaries containing the metrics for each fold, and the name of the metrics to average.
    Returns the average value of the specified metric across all folds.
    """
    averaged_metrics = {}
    for metric in metrics:
        metric_values = []
        for fold, fold_data in model_data.items():
            fold_values = []
            for entry in fold_data:
                if metric in entry:
                    fold_values.append(entry[metric])
            metric_values.append(fold_values)
        if len(metric_values) > 0:
            averaged_metrics[metric] = np.mean(metric_values, axis=0)
        else:
            print(f"WARNING: No values found for metric {metric} across folds.")
            averaged_metrics[metric] = None
    return averaged_metrics

def get_subjects_by_eye_dominance(side: str, eye_dom_csv_path: Path):
    """
    Reads eye dominance information from a CSV file and returns a list of subject IDs with the specified eye dominance ('left' or 'right').
    Expects the CSV to have columns 'subject_id' and 'eye_dominance'.
    """
    eye_dom = pd.read_csv(eye_dom_csv_path)

    return eye_dom[eye_dom["eye_dominance"] == side]["subject_id"].tolist()

def nearest_electrodes(info, n_neighbours):
    """{channel index: indices of its n_neighbours nearest electrodes}, Euclidean distance of the electrode positions in info."""
    pos = np.array([ch["loc"][:3] for ch in info["chs"]])
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return {c: [j for j in np.argsort(dist[c]) if j != c][:n_neighbours] for c in range(len(pos))}

def build_neighbor_mask(n_neighbours, n_chans):
    """
    Builds the (n_chans, n_chans) boolean cluster table of masked-channel reconstruction: row c is True for channel c
    and its n_neighbours nearest electrodes (HydroCel-65 positions), the channels hidden when c is the target.
    With n_neighbours = 0 only the target channel itself is hidden.
    """
    info = plot_data_processing.create_topomap_info()
    neighbors = nearest_electrodes(info, n_neighbours)
    mask = np.zeros((n_chans, n_chans), dtype=bool)
    for ch, ch_neighbors in neighbors.items():
        mask[ch, [ch, *ch_neighbors]] = True
    return mask


def balanced_eval_channels(allowed_masks, n_channels):
    """
    Returns the evaluation target channels of every sample, n_channels allowed channels each, so that every channel is
    evaluated about equally often although interpolation makes some channels allowed in far fewer samples than others.
    allowed_masks: one (n_chans,) boolean mask of allowed target channels per sample, in dataset order.
    Greedy and deterministic: going through the samples in order, every sample takes the allowed channels that have been
    evaluated least so far. A channel that is rarely allowed thus catches up in the samples where it is allowed.
    """
    if len(allowed_masks) == 0:
        return []
    n_chans = len(allowed_masks[0])
    times_evaluated = np.zeros(n_chans, dtype=int)  # how often every channel has been chosen so far
    eval_channels = []
    for sample_idx, allowed in enumerate(allowed_masks):
        candidates = np.flatnonzero(np.asarray(allowed)).tolist()
        if len(candidates) < n_channels:
            raise ValueError(f"Sample {sample_idx} has {len(candidates)} allowed channels, fewer than the {n_channels} evaluation channels")

        def priority(ch):
            # 1. the channels evaluated least so far come first
            # 2. among channels evaluated equally often, start at channel sample_idx and go up (wrapping around),
            #    so that ties don't always go to the lowest channel indices
            return times_evaluated[ch], (ch - sample_idx) % n_chans

        chosen = sorted(sorted(candidates, key=priority)[:n_channels])
        times_evaluated[chosen] += 1
        eval_channels.append(np.array(chosen))
    return eval_channels



def channel_permutation(key, allowed, seed):
    """
    Returns a fixed random order of the allowed target channels of one sample, seeded by the sample's key,
    so it is the same in every run, fold and dataset order.
    """
    rng = np.random.default_rng([seed, zlib.crc32(repr(key).encode())])
    return rng.permutation(np.flatnonzero(allowed))







# ---------------------------------------------------------------- gain injection
def hydrocel_info(fs: int):
    """mne.Info of all 65 HydroCel-65 electrodes (E1 ... E64 and the reference Cz) with the standard montage, and the montage."""
    montage = mne.channels.make_standard_montage("GSN-HydroCel-65_1.0")
    info = mne.create_info(montage.ch_names, fs, "eeg")
    info.set_montage(montage)
    return info, montage


def model_channel_topography(gain, info, egi_to_1020_map: Dict[str, str], channel_order: List[str]):
    """
    Average-references a 65-channel topography (as in the preprocessing), picks the model channels in channel_order
    (e.g. settings.CHORDER_STANDARD, mapped to EGI electrodes with egi_to_1020_map) and scales it to a maximum absolute value of 1.
    """
    gain = gain - gain.mean()
    egi = {v: k for k, v in egi_to_1020_map.items()}
    montage_name = lambda ch: "Cz" if egi[ch] == "EEG VREF" else "E" + egi[ch].split(" ")[-1]
    g = np.array([gain[info.ch_names.index(montage_name(ch))] for ch in channel_order])
    return g / np.abs(g).max()


def fsaverage_label_gains(labels: List[str], fs: int):
    """
    Scalp field of every given aparc label (both hemispheres) on the MNE fsaverage template: the sum over the label's sources,
    oriented normal to the cortex with equal amplitude, through the 3-layer fsaverage BEM, HydroCel-65 electrodes aligned
    with the template by their fiducials (trans='fsaverage'). Folded cortex partly cancels itself, as in real EEG.
    Returns ({label: 65-channel field}, {label: number of sources}, info of the 65 channels).
    Downloads fsaverage (~200 MB) into the MNE data folder on the first call.
    """
    fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    info, _ = hydrocel_info(fs)
    fwd = mne.make_forward_solution(info, trans="fsaverage", src=fs_dir / "bem" / "fsaverage-ico-5-src.fif",
                                    bem=fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif", meg=False, eeg=True, verbose=False)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    annotation = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir=fs_dir.parent, verbose=False)
    fields, counts = {}, {}
    for name in labels:
        offset, columns = 0, []
        for hemi, src in zip(["lh", "rh"], fwd["src"]):
            vertices = np.concatenate([lab.vertices for lab in annotation if lab.name == f"{name}-{hemi}"] or [[]])
            columns.append(offset + np.where(np.isin(src["vertno"], vertices))[0])
            offset += len(src["vertno"])
        columns = np.concatenate(columns)
        if len(columns) == 0:
            raise ValueError(f"No fsaverage source in the aparc label {name!r}")
        fields[name], counts[name] = fwd["sol"]["data"][:, columns].sum(axis=1), len(columns)
    return fields, counts, info


def sphere_topography(depth: float, fs: int, egi_to_1020_map: Dict[str, str], channel_order: List[str]):
    """
    Scalp topography (model channels, max |g| = 1) of a radial dipole in occipital cortex, at depth (fraction of the head
    radius) towards the midpoint of O1 (E35) and O2 (E39), 3-shell sphere model fitted to the HydroCel-65 electrodes,
    and a description of it. Needs no download.
    """
    info, montage = hydrocel_info(fs)
    sphere = mne.make_sphere_model("auto", "auto", info, verbose=False)
    pos = montage.get_positions()["ch_pos"]
    r0 = np.asarray(sphere["r0"])
    direction = (pos["E35"] + pos["E39"]) / 2 - r0
    direction /= np.linalg.norm(direction)
    dipole = mne.Dipole(times=[0.0], pos=[r0 + depth * sphere.radius * direction], amplitude=[1e-8],
                        ori=[direction], gof=[100.0])
    fwd, _ = mne.make_forward_dipole(dipole, sphere, info, trans=None, verbose=False)
    return model_channel_topography(fwd["sol"]["data"][:, 0], info, egi_to_1020_map, channel_order), \
        f"radial dipole at {depth:g} of the head radius towards O1/O2, 3-shell sphere model"


def injected_sources(head_model: str, labels: List[str], mixed_gains: Optional[Dict[str, float]], depth: float, fs: int,
                     egi_to_1020_map: Dict[str, str], channel_order: List[str]):
    """
    The sources of a gain injection: ({name: topography (model channels, max |g| = 1)}, {name: gain per unit of injection},
    description). mixed_gains ({aparc label: gain}, needs head_model "fsaverage"): every label is a separate source with its
    own gain. Otherwise one source: the given aparc labels together (head_model "fsaverage") or the sphere-model dipole at
    depth ("sphere"), with gain 1, so the injection value is its gain.
    """
    if mixed_gains is not None:
        if head_model != "fsaverage":
            raise ValueError("A mixed injection needs the fsaverage head model: its sources are aparc labels")
        fields, counts, info65 = fsaverage_label_gains(list(mixed_gains), fs)
        topographies = {name: model_channel_topography(field, info65, egi_to_1020_map, channel_order) for name, field in fields.items()}
        description = "; ".join(f"{name} {100 * gain:+g} % ({counts[name]} sources)" for name, gain in mixed_gains.items())
        return topographies, dict(mixed_gains), f"mixed injection: {description} (aparc, both hemispheres, fsaverage BEM)"
    if head_model == "fsaverage":
        fields, counts, info65 = fsaverage_label_gains(labels, fs)
        g = model_channel_topography(sum(fields.values()), info65, egi_to_1020_map, channel_order)
        description = f"{', '.join(labels)} (aparc, both hemispheres, {sum(counts.values())} sources normal to the cortex), fsaverage BEM"
    else:
        g, description = sphere_topography(depth, fs, egi_to_1020_map, channel_order)
    return {"source": g}, {"source": 1.0}, description


def source_overlap(topographies: Dict[str, np.ndarray]):
    """
    How distinguishable injected sources are on the model channels: pairwise cosine similarity of their topographies
    (1 = identical shape, -1 = identical but inverted) and the condition number of their matrix (largest / smallest
    singular value; large values mean their activity cannot be told apart and the joint estimate is unstable).
    """
    names = list(topographies)
    G = np.stack([topographies[name] for name in names], axis=1)
    unit = G / np.linalg.norm(G, axis=0)
    singular = np.linalg.svd(G, compute_uv=False)
    return pd.DataFrame(unit.T @ unit, index=names, columns=names), singular[0] / singular[-1]


def inject_sources(X, G, gains, times, ramp_duration: float):
    """
    Returns a copy of the trials X (..., channels, time) in which the injected sources are stronger after stimulus onset.
    G: (channels, n_sources) topographies, gains: (n_sources,) relative gain of every source, times: (time,) in s, onset at 0.
    The sources' activity is estimated jointly by least squares, S = pinv(G) x, so activity shared by overlapping
    topographies is not counted twice; source k then gets gains[k] * G[:, k] * S_k(t) added, ramped in with a raised
    cosine over ramp_duration s after onset.
    """
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip(times / ramp_duration, 0, 1))  # 0 before onset, 1 from ramp_duration s on
    sources = np.einsum("kc,...ct->...kt", np.linalg.pinv(G), X)
    return X + np.einsum("ck,k,...kt->...ct", G, np.asarray(gains), sources * ramp)
