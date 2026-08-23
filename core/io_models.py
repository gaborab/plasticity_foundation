from dataclasses import dataclass, fields, asdict
from functools import cached_property
from typing import Dict, Any
from core.io_utils import nested_group_to_dict
import h5py
import numpy as np

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
    std: float = None
    std_bandwise: Dict[str, float] = None
    nmae: float = None
    nmae_bandwise: Dict[str, float] = None

@dataclass
class Trial:
    trial_data: h5py.Group
    subject: str
    condition: str
    idx: int

    @property
    def target_data(self) -> np.ndarray:
        return self.trial_data["target"][()]
    @property
    def recon_data(self) -> np.ndarray:
        return self.trial_data["output"][()]
    @cached_property
    def channel_metrics(self) -> Dict[str, Metrics]:
        ch_group = self.trial_data["channel_metrics"]
        ret_metrics = {}
        for ch_name, metrics_data in ch_group.items():
            metrics_dict = nested_group_to_dict(metrics_data)
            ret_metrics[ch_name] = Metrics.from_dict(metrics_dict)
        return ret_metrics
    @cached_property
    def metrics(self) -> Metrics:
        m = self.trial_data["metrics"]
        metrics_dict = nested_group_to_dict(m)
        return Metrics.from_dict(metrics_dict)