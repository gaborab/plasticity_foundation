from dataclasses import dataclass, fields, asdict
from functools import cached_property
from typing import Dict, Any, Optional, List, Tuple

from core import settings
import h5py
import numpy as np
import json

@dataclass
class Metrics:
    @classmethod
    def from_dict(cls, dict: Dict) -> 'Metrics':
        metrics_keys = {f.name for f in fields(Metrics)}
        filtered_dict = {k: v for k, v in dict.items() if k in metrics_keys}
        return cls(**filtered_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


    r2: float = None
    r2_bandwise: Dict[str, float] = None
    mae: float = None
    mae_bandwise: Dict[str, float] = None
    signal_volatility: float = None
    signal_volatility_bandwise: Dict[str, float] = None
    nmae: float = None
    nmae_bandwise: Dict[str, float] = None
    corrected_mae: float = None
    corrected_mae_bandwise: Dict[str, float] = None
    signal_mean: float = None
    signal_mean_bandwise: Dict[str, float] = None
    recon_mean: float = None
    recon_mean_bandwise: Dict[str, float] = None
    recon_volatility: float = None
    recon_volatility_bandwise: Dict[str, float] = None


def ch_metrics_from_json(ch_json: str) -> Dict[str, Metrics]:
    raw_dict = json.loads(ch_json)
    ret_metrics = {}
    for ch_name, metrics_data in raw_dict.items():
        ret_metrics[ch_name] = Metrics.from_dict(metrics_data)
    return ret_metrics

def _metrics_field(window: str, channel: bool = False) -> cached_property:
    """Returns a cached property that parses the trial-level (or channel-level) metrics of window from the trial's HDF5 attributes."""
    key = f"{window}_channel_metrics_json" if channel else f"{window}_metrics_json"

    def getter(self):
        raw = self.trial_data.attrs[key]
        return ch_metrics_from_json(raw) if channel else Metrics.from_dict(json.loads(raw))

    return cached_property(getter)

@dataclass
class Trial:
    """
    Lazy view on one trial of an inference HDF5 file.
    The metrics of every time window are available as {window}_metrics / {window}_channel_metrics, parsed on first access and cached.
    metrics / channel_metrics return the metrics of the trial's window, so analyses reading them use the window selected when loading.
    If loaded with a time series file, time_series(metric) reads the trial's per-band time series from it on every call (not cached).
    """
    trial_data: h5py.Group
    subject: str
    condition: str
    idx: int
    window: str = "full"  # key of settings.METRIC_WINDOWS, selects which stored metrics metrics / channel_metrics return
    time_series_data: Optional[h5py.Group] = None  # group of this trial in the time series file, None if loaded without one

    def __post_init__(self):
        if self.window not in settings.METRIC_WINDOWS:
            raise ValueError(f"Unknown window '{self.window}', expected one of {list(settings.METRIC_WINDOWS)}")

    @property
    def target_data(self) -> np.ndarray:
        return self.trial_data["target"][()]
    @property
    def recon_data(self) -> np.ndarray:
        return self.trial_data["output"][()]

    @property
    def metrics(self) -> Metrics:
        """Trial-level metrics of the selected window, the same cached object as {window}_metrics (e.g. analysis_metrics)."""
        return getattr(self, f"{self.window}_metrics")

    @property
    def channel_metrics(self) -> Dict[str, Metrics]:
        """Channel-level metrics {"ch_0": Metrics, ...} of the selected window, the same cached object as {window}_channel_metrics."""
        return getattr(self, f"{self.window}_channel_metrics")

    def time_series(self, metric: str, bands: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Returns the time series of metric as {band: (n_chans, n_timepoints) array}, including "all bands".
        Available metrics are the ones saved in the time series file (keys of plot_data_processing.TIME_SERIES_METRICS: mae, error, target, recon).
        bands: only read these bands (None reads all of them).
        Reads from the file on every call, the result is not cached.
        """
        if self.time_series_data is None:
            raise ValueError("Time series data not available for this trial. Please load the trial with time series data.")
        if metric not in self.time_series_data:
            raise ValueError(f"Metric '{metric}' not found in time series data. Available metrics: {list(self.time_series_data.keys())}")
        metric_group = self.time_series_data[metric]
        if bands is None:
            bands = list(metric_group.keys())
        missing = [band for band in bands if band not in metric_group]
        if missing:
            raise ValueError(f"Bands {missing} not found in time series data. Available bands: {list(metric_group.keys())}")
        return {band: metric_group[band][()] for band in bands}


    full_metrics = _metrics_field("full")
    full_channel_metrics = _metrics_field("full", channel=True)
    analysis_metrics = _metrics_field("analysis")
    analysis_channel_metrics = _metrics_field("analysis", channel=True)
    reference_metrics = _metrics_field("reference")
    reference_channel_metrics = _metrics_field("reference", channel=True)

@dataclass
class TargetStats:
    """Mean and std the dB targets are normalized with, fitted on the training targets of one fold."""
    mu: float
    sigma: float

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TargetStats":
        return cls(mu=data["mu"], sigma=data["sigma"])

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def fit(cls, targets_data, eps: float = 1e-8) -> "TargetStats":
        """
        Fits the mean and std of all masked-in target values, pooled over trials, channels, freqs and time.
        targets_data yields (target, mask, metadata) per sample, see FinetuneDataset.masked_targets. Only the channels
        in metadata["allowed"] are used, so interpolated channels, which are never targets, are left out.
        """
        n, total, total_sq = 0, 0.0, 0.0
        for target, mask, meta in targets_data:
            values = target[meta["allowed"]][:, mask].double()  # (channels, freqs, time) -> (allowed channels, n_valid_bins)
            n += values.numel()
            total += values.sum().item()
            total_sq += (values ** 2).sum().item()
        mu = total / n
        return cls(mu=mu, sigma=(total_sq / n - mu ** 2) ** 0.5 + eps)

    def normalize(self, data):
        """Normalizes a dB tensor or array."""
        return (data - self.mu) / self.sigma

    def denormalize(self, data):
        """Converts a normalized tensor or array, e.g. a model output, back to dB."""
        return (data * self.sigma) + self.mu