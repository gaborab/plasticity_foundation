from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
import pywt
import torch
from sklearn.metrics import r2_score


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

def calc_superlet(tensor_data, fs, freq_window, n_freqs, n_cycles, **kwargs):
    """
    Placeholder function for calculating superlet coefficients. The actual implementation would depend on the specific superlet algorithm being used.
    """

def vars_from_filename(stem, marker):
    """
    Extracts the numeric ID from a filename based on a specified marker. For example, if the filename is "subject_condition_epoch5.pt" and the marker is "_epoch", this function will return 5.
    """

    # Extract epoch ID
    pos = stem.rfind(marker)
    if pos == -1:
        print("WARNING: marker not found in filename:", stem)
    id = stem[pos + len(marker):]  # part after "_epoch"
    if not id.isdigit():
        print("WARNING: epoch id non numeric in filename:", stem)
    id = int(id)

    # Extract subject and condition from filename
    prefix = stem[:pos]  # part before "_epoch"
    parts = prefix.split('_', 1)
    if len(parts) < 2:
        print("WARNING: unexpected filename:", stem)
    subject, condition = parts[0], parts[1]

    return pos, id, subject, condition

def vars_to_filename(subject, condition, id, marker, ext):
    """
    Constructs a filename from the given subject, condition, ID, marker, and extension. For example, if subject is "subject1", condition is "conditionA", id is 5, marker is "_epoch", and ext is ".pt", this function will return "subject1_conditionA_epoch5.pt".
    """
    filename = f"{subject}_{condition}_{marker}{str(id)}{ext}"
    return filename

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
    if y_true.ndim > 2 or y_pred.ndim > 2:
        raise ValueError(f"Expected y_true and y_pred to be 1D or 2D arrays, but got shapes {y_true.shape} and {y_pred.shape}.")

    if y_true.ndim == 2:
        # Flatten the arrays if they are 2D (e.g., for wavelet targets)
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

    r2 = r2_score(y_true, y_pred)
    return r2

def calculate_bandwise_r2(y_true, y_pred, freqs, freq_bands):
    """
    Calculates the R-squared metric for each specified frequency band given the true and predicted values of spectral data.
    The function expects y_true and y_pred to be 2D arrays with shape (n_freqs, time_points) or 3D arrays with shape (n_chans, n_freqs, time_points), and freqs to be a 1D array of frequency values corresponding to the second dimension of y_true and y_pred.
    Freq_bands should be a dict of tuples specifying the frequency ranges for each band (e.g., [(1, 4), (4, 8), (8, 12), (12, 30), (30, 50)]).
    Returns a dictionary mapping each frequency band to its corresponding R-squared value.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")
    if y_true.ndim not in [2, 3] or y_pred.ndim not in [2, 3]:
        raise ValueError(f"Expected y_true and y_pred to be 2D or 3D arrays, but got shapes {y_true.shape} and {y_pred.shape}.")
    r2_bandwise = {}
    for name, band in freq_bands.items():
        low, high = band
        band_indices = np.where((freqs >= low) & (freqs <= high))[0]
        if band_indices.size == 0:
            print(f"WARNING: No frequencies found in band {name} ({low}-{high} Hz). Skipping R2 calculation for this band.")
            continue
        if y_true.ndim == 3:
            band_true = y_true[:, band_indices, :].flatten()
            band_pred = y_pred[:, band_indices, :].flatten()
        else:
            band_true = y_true[band_indices, :].flatten()
            band_pred = y_pred[band_indices, :].flatten()

        r2 = r2_score(band_true, band_pred)
        r2_bandwise[name] = r2
    return r2_bandwise


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





