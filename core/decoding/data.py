"""Band-passed trial arrays for the sliding decoder."""
from typing import Dict, List, Optional, Tuple

import numpy as np
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
