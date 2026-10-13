from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import pywt
import torch
from sklearn.metrics import r2_score
from einops import rearrange
from core.io_models import *
import pandas as pd

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


def valid_wavelet_bins(freqs, patch_window, target_window, fs, max_cycles, max_reach):
    """
    Returns a (freqs, time) boolean mask over the samples of one patch (patch_window, sampled at fs from its start):
    True where the sample lies in target_window and the wavelet of that frequency reaches no further than the patch edges,
    i.e. the bin is computed from real data of the patch only, not from its reflect padding.
    """
    n_samples = round((patch_window[1] - patch_window[0]) * fs)
    times = patch_window[0] + np.arange(n_samples) / fs
    reach = wavelet_reach(freqs, max_cycles, max_reach)[:, None]
    in_target = (times >= target_window[0] - 1e-9) & (times <= target_window[1] + 1e-9)
    inside_patch = (times - reach >= patch_window[0] - 1e-9) & (times + reach <= patch_window[1] + 1e-9)
    return in_target[None, :] & inside_patch


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

def cut_to_timewindow(tensor, original_window: Tuple[int, int], new_window: Tuple[int, int], fs:int):
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
    Calculates the R-squared metric for the given true and predicted values of single-channel data.
    The function expects either 1D or 2D arrays for y_true and y_pred, where the last dimension corresponds to time points. If the input is 2D, it will be flattened before calculation.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")
    if y_true.ndim > 3 or y_pred.ndim > 3:
        raise ValueError(f"Expected y_true and y_pred to be 1D or 2D or 3D arrays, but got shapes {y_true.shape} and {y_pred.shape}.")

    if y_true.ndim >= 2:
        # Flatten the arrays if they are 2D
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

    r2 = r2_score(y_true, y_pred)
    return r2


def calculate_bandwise_metrics(y_true, y_pred, signal_freq_stds, recon_freq_stds, band_indices: Dict[str, np.ndarray], eps: float = 1e-8):
    """
    Calculates R-squared, MAE, STD and NMAE metrics for each frequency band by indexing into the frequency dimension of y_true and y_pred.
    Expects band_indices as {band name: indices of the band's frequency bins}, e.g. settings.FREQ_BAND_INDICES.
    Expects y_true and y_pred to be 2D or 3D arrays, where the last dimension corresponds to time points and the second-to-last dimension corresponds to frequencies.
    Expects freq_stds to be a 1D array of standard deviations for each frequency.
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
        elif y_true.ndim == 3:
            band_true = y_true[:, indices, :].flatten()
            band_pred = y_pred[:, indices, :].flatten()
        else:
            print("WARNING: Expected 2D or 3D arrays for y_true and y_pred when calculating bandwise metrics, but got arrays with shape:", y_true.shape)
            return None, None, None, None, None, None, None
        r2 = r2_score(band_true, band_pred)
        mae = np.mean(np.abs(band_true - band_pred))
        r2_bandwise[name] = r2
        mae_bandwise[name] = mae
        signal_std_bandwise[name] = np.mean(signal_freq_stds[indices])
        recon_std_bandwise[name] = np.mean(recon_freq_stds[indices])
        signal_mean_bandwise[name] = np.mean(band_true)
        recon_mean_bandwise[name] = np.mean(band_pred)
        nmae_bandwise[name] = mae / signal_std_bandwise[name] + eps

    return r2_bandwise, mae_bandwise, signal_std_bandwise, recon_std_bandwise, nmae_bandwise, signal_mean_bandwise, recon_mean_bandwise


def get_inference_metrics(trial_output, trial_target, band_indices: Optional[Dict[str, np.ndarray]] = None, ch_prefix: str = 'ch_', eps: float = 1e-8) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Calculates inference metrics (e.g., R-squared, MAE) for a single trial's output and target data. Expects trial_output and trial_target to be 3D arrays, where the last dimension corresponds to time points.
    band_indices: {band name: indices of the band's frequency bins} for spectral targets, None skips the band-wise metrics.
    Returns a dictionary containing the calculated metrics.
    """
    spectral = band_indices is not None

    signal_stds = np.std(trial_target, axis=-1)
    recon_stds = np.std(trial_output, axis=-1)
    global_std = np.mean(signal_stds)
    recon_std = np.mean(recon_stds)

    # global metrics
    g_metrics = Metrics()
    g_metrics.r2 = calculate_r2(trial_target, trial_output)
    g_metrics.mae = np.mean(np.abs(trial_target - trial_output))
    g_metrics.nmae = g_metrics.mae / global_std + eps
    g_metrics.signal_volatility = global_std
    g_metrics.signal_mean = np.mean(trial_target).item()
    g_metrics.recon_mean = np.mean(trial_output).item()
    g_metrics.recon_volatility = recon_std

    # global band-wise metrics
    if spectral:
        signal_freq_stds = np.mean(signal_stds, axis=0)
        recon_freq_stds = np.mean(recon_stds, axis=0)
        g_metrics.r2_bandwise, g_metrics.mae_bandwise, g_metrics.signal_volatility_bandwise, g_metrics.recon_volatility_bandwise, g_metrics.nmae_bandwise, g_metrics.signal_mean_bandwise, g_metrics.recon_mean_bandwise = (
            calculate_bandwise_metrics(trial_target, trial_output, signal_freq_stds, recon_freq_stds, band_indices=band_indices, eps=eps)
        )
    g_metrics = g_metrics.to_dict()

    ch_metrics_dict = {}
    signal_channel_stds = np.mean(signal_stds, axis=1) if spectral else signal_stds
    recon_channel_stds = np.mean(recon_stds, axis=1) if spectral else recon_stds
    # channel-wise metrics
    for i in range(trial_target.shape[0]):
        ch_key = f'{ch_prefix}{i}'
        ch_metrics = Metrics()
        ch_metrics.r2 = calculate_r2(trial_target[i], trial_output[i])
        ch_metrics.mae = np.mean(np.abs(trial_target[i] - trial_output[i]))

        ch_metrics.signal_volatility = signal_channel_stds[i]
        ch_metrics.recon_volatility = recon_channel_stds[i]
        ch_metrics.nmae = ch_metrics.mae / (signal_channel_stds[i] + eps)

        ch_metrics.signal_mean = np.mean(trial_target[i]).item()
        ch_metrics.recon_mean = np.mean(trial_output[i]).item()



        # channel-wise band-wise metrics
        if spectral:
            signal_channel_freq_stds = signal_stds[i, :]
            recon_channel_freq_stds = recon_stds[i, :]
            ch_metrics.r2_bandwise, ch_metrics.mae_bandwise, ch_metrics.signal_volatility_bandwise, ch_metrics.recon_volatility_bandwise, ch_metrics.nmae_bandwise, ch_metrics.signal_mean_bandwise, ch_metrics.recon_mean_bandwise = (
                calculate_bandwise_metrics(trial_target[i], trial_output[i], signal_channel_freq_stds, recon_channel_freq_stds, band_indices=band_indices, eps=eps)
            )

        ch_metrics_dict[ch_key] = ch_metrics.to_dict()
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





