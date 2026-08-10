from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import pywt
import torch
from sklearn.metrics import r2_score
from einops import rearrange
from core.io_models import *

def calc_wavelet(tensor_data, fs, freq_window, n_freqs, wavelet, **kwargs):
    """
    Calculates the wavelet coefficients for a given EEG epoch tensor and returns the magnitude of the coefficients along with the corresponding frequencies in a geometric scale within the specified frequency window.
    """
    freqs = np.geomspace(freq_window[0], freq_window[1], n_freqs)
    # normalize frequencies by sampling frequency
    freqs_norm = freqs / fs

    scales = pywt.frequency2scale(wavelet, freqs_norm)

    coeffs, _ = pywt.cwt(tensor_data.numpy(), scales, wavelet)
    mag = np.abs(coeffs)
    return mag, freqs



def cut_to_timewindow(tensor, original_window: Tuple[int, int], new_window: Tuple[int, int], fs:int):
    """
        Cuts the time dimension of a tensor based on the sampling frequency.
        Expects time to be the last dimension.
    """

    from_index =  int((new_window[0] - original_window[0]) * fs)
    to_index = int((new_window[1] - original_window[0]) * fs)

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


def calculate_bandwise_metrics(y_true, y_pred, freq_stds, freqs, freq_bands, eps: float = 1e-8):
    """
    Calculates R-squared, MAE, STD and NMAE metrics for each frequency band defined in freq_bands by indexing into the frequency dimension of y_true and y_pred.
    Expects y_true and y_pred to be 2D or 3D arrays, where the last dimension corresponds to time points and the second-to-last dimension corresponds to frequencies.
    Expects freq_stds to be a 1D array of standard deviations for each frequency.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")
    if y_true.ndim not in [2, 3] or y_pred.ndim not in [2, 3]:
        raise ValueError(f"Expected y_true and y_pred to be 2D or 3D arrays, but got shapes {y_true.shape} and {y_pred.shape}.")
    r2_bandwise = {}
    mae_bandwise = {}
    std_bandwise = {}
    nmae_bandwise = {}
    for name, band in freq_bands.items():
        low, high = band
        band_indices = np.where((freqs >= low) & (freqs <= high))[0]
        if band_indices.size == 0:
            print(
                f"WARNING: No frequencies found in band {name} ({low}-{high} Hz). Skipping R2 calculation for this band.")
            continue
        if y_true.ndim == 2:
            band_true = y_true[band_indices, :].flatten()
            band_pred = y_pred[band_indices, :].flatten()
        elif y_true.ndim == 3:
            band_true = y_true[:, band_indices, :].flatten()
            band_pred = y_pred[:, band_indices, :].flatten()
        else:
            print("WARNING: Expected 2D or 3D arrays for y_true and y_pred when calculating bandwise metrics, but got arrays with shape:", y_true.shape)
            return {}, {}
        r2 = r2_score(band_true, band_pred)
        mae = np.mean(np.abs(band_true - band_pred))
        r2_bandwise[name] = r2
        mae_bandwise[name] = mae
        std_bandwise[name] = np.mean(freq_stds[band_indices])
        nmae_bandwise[name] = mae / std_bandwise[name] + eps

    return r2_bandwise, mae_bandwise, std_bandwise, nmae_bandwise


def get_inference_metrics(trial_output, trial_target, freqs: Optional[Any] = None, freq_bands: Optional[Any]= None, ch_prefix: str = 'ch_', eps: float = 1e-8) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Calculates inference metrics (e.g., R-squared, MAE) for a single trial's output and target data. Expects trial_output and trial_target to be 3D arrays, where the last dimension corresponds to time points.
    Returns a dictionary containing the calculated metrics.
    """
    spectral = freqs is not None and freq_bands is not None

    stds = np.std(trial_target, axis=-1)
    global_std = np.mean(stds)


    print("debug: std shape: ", stds.shape) # TODO remove debug

    # global metrics
    g_metrics = Metrics()
    g_metrics.r2 = calculate_r2(trial_target, trial_output)
    g_metrics.mae = np.mean(np.abs(trial_target - trial_output))
    g_metrics.nmae = g_metrics.mae / global_std + eps
    g_metrics.std = global_std

    # global band-wise metrics
    if spectral:
        freq_stds = np.mean(stds, axis=0)
        g_metrics.r2_bandwise, g_metrics.mae_bandwise, g_metrics.std_bandwise, g_metrics.nmae_bandwise = (
            calculate_bandwise_metrics(trial_target, trial_output, freq_stds, freqs=freqs, freq_bands=freq_bands, eps=eps)
        )
    g_metrics = g_metrics.to_dict()

    ch_metrics_dict = {}
    channel_stds = np.mean(stds, axis=1) if spectral else stds
    # channel-wise metrics
    for i in range(trial_target.shape[0]):
        ch_key = f'{ch_prefix}{i}'
        ch_metrics = Metrics()
        ch_metrics.r2 = calculate_r2(trial_target[i], trial_output[i])
        ch_metrics.mae = np.mean(np.abs(trial_target[i] - trial_output[i]))

        ch_metrics.std = channel_stds[i]
        ch_metrics.nmae = ch_metrics.mae / channel_stds[i]


        # channel-wise band-wise metrics
        if spectral:
            channel_freq_stds = stds[i, :]
            ch_metrics.r2_bandwise, ch_metrics.mae_bandwise, ch_metrics.std_bandwise, ch_metrics.nmae_bandwise = (
                calculate_bandwise_metrics(trial_target[i], trial_output[i], channel_freq_stds, freqs=freqs, freq_bands=freq_bands, eps=eps)
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





