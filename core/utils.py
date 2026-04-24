import numpy as np
import pywt



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

def cut_to_timewindow(tensor, window_size: int):
    """
    Cuts the input tensor to be divisible by the specified patch size along the time dimension.
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