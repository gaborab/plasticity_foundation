"""Band-passed trial arrays for the sliding decoder, and the planted changes of the Task C checks."""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import butter, hilbert, sosfiltfilt


def band_file_name(band: str) -> str:
    """The short name of a DECODING_BANDS band in file names: its first word, e.g. "theta" for "theta (5-8Hz)"."""
    return band.split(" ")[0]


def band_pass_trials(X: np.ndarray, fs: float, bands: Dict[str, Tuple[Optional[float], float]], decim: int = 1,
                     envelope: bool = False, order: int = 4) -> np.ndarray:
    """
    Band-passes trials X (n_trials, n_chans, n_times) into every band and returns (n_trials, n_chans, n_bands, n_times // decim).
    bands: {name: (low, high)} in Hz, low None for a low-pass only (broadband). Zero-phase Butterworth (sosfiltfilt) of
    the given order, with reflection padding of the window itself, so the filter sees no data outside the window (no
    activity from after its end, e.g. button presses, leaks in). The edges of the window are less reliable.
    decim: keep every decim-th sample after filtering, the bands have to stay below the new Nyquist frequency.
    envelope: the Hilbert amplitude of every band-passed signal instead of the signal itself (low-pass bands stay signals).
    """
    out = np.empty((X.shape[0], X.shape[1], len(bands), X.shape[2] // decim), dtype=np.float32)
    for b, (low, high) in enumerate(bands.values()):
        if low is None:
            sos = butter(order, high, btype="lowpass", fs=fs, output="sos")
        else:
            sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, X, axis=-1, padtype="even", padlen=X.shape[-1] - 1)
        if envelope and low is not None:
            filtered = np.abs(hilbert(filtered, axis=-1))
        out[:, :, b] = filtered[..., ::decim][..., :out.shape[-1]]
    return out


def stack_class_samples(datasets) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Stacks the samples of one or more ClassFinetuneDatasets: inputs (n_trials, n_chans, n_times), class ids (n_trials,)
    and the trials' metadata, in dataset order.
    """
    samples = [sample for dataset in datasets for sample in dataset.samples]
    if not samples:
        raise ValueError("No trials to stack")
    X = np.stack([x.numpy() for x, _, _ in samples]).astype(np.float64)
    y = np.array([label for _, label, _ in samples])
    return X, y, [metadata for _, _, metadata in samples]


def decoding_times(window: Tuple[float, float], fs: float, decim: int, n_times: int) -> np.ndarray:
    """Times in s of the decimated samples of a window (the time of each sample's start)."""
    return window[0] + np.arange(n_times) * decim / fs

# PLANTED CHANGES (handoff Task C), on epochs {subject: {condition: [(id, epoch (n_chans, n_times)), ...]}}. They return
# a new dict with new tensors for the planted trials, the other trials are the same objects: the input stays unchanged
Epochs = Dict[str, Dict[str, List[Tuple[int, Any]]]]


def plant_delay(epochs: Epochs, conditions: Sequence[str], delay: float, fs: float) -> Epochs:
    """
    Delays the epochs of conditions by delay s: every sample moves round(delay * fs) samples later, zeros fill the start
    and the end is dropped. Cutting a window afterwards equals cutting it delay s earlier (handoff C3), as long as the
    window stays clear of the epoch's start and end.
    """
    shift = round(delay * fs)
    delayed = {}
    for subject, subject_epochs in epochs.items():
        delayed[subject] = {}
        for condition, trials in subject_epochs.items():
            if condition not in conditions:
                delayed[subject][condition] = trials
                continue
            delayed[subject][condition] = []
            for (idx, trial) in trials:
                shifted = torch.zeros_like(trial)
                shifted[:, shift:] = trial[:, :trial.shape[-1] - shift]
                delayed[subject][condition].append((idx, shifted))
    return delayed


def plant_attenuation(epochs: Epochs, conditions: Sequence[str], attenuation: float, channel_mask: Sequence[bool],
                      epoch_window: Tuple[float, float], fs: float, window: Tuple[float, float]) -> Epochs:
    """
    Attenuates the evoked response of conditions (handoff C4): subtracts attenuation x the subject's mean epoch of the
    condition (calc_class_mean, computed from these epochs) on the channels of channel_mask, within window (s, epoch
    times from epoch_window sampled at fs).
    """
    attenuated = {}
    cond_templates = {cond: calc_class_mean(epochs, cond, channel_mask, epoch_window, fs, window) for cond in conditions}
    for subject, subject_epochs in epochs.items():
        attenuated[subject] = {}
        for condition, trials in subject_epochs.items():
            if condition in conditions:
                template = cond_templates[condition][subject]
                attenuated[subject][condition] = [(idx, trial - attenuation * template) for (idx, trial) in trials]
            else:
                attenuated[subject][condition] = trials
    return attenuated


def calc_class_mean(epochs: Epochs, condition: str, channel_mask: Sequence[bool], epoch_window: Tuple[float, float],
                    fs: float, window: Tuple[float, float]) -> Dict[str, torch.Tensor]:
    """
    Every subject's mean epoch of condition, the subject's class-average evoked response (n_chans, n_times), zero outside
    the channels of channel_mask and outside window (s). Subjects without trials of the condition are left out.
    """
    templates = {}
    for subject, subject_epochs in epochs.items():
        trials = subject_epochs.get(condition, [])
        if not trials:
            continue
        mean = torch.stack([epoch for _, epoch in trials]).mean(dim=0)
        times = epoch_window[0] + torch.arange(mean.shape[-1]) / fs
        in_window = (times >= window[0] - 1e-9) & (times <= window[1] + 1e-9)
        mask = torch.as_tensor(channel_mask, dtype=torch.bool)[:, None] & in_window[None, :]
        templates[subject] = mean * mask
    return templates