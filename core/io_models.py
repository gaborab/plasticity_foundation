from dataclasses import dataclass, fields, asdict
from functools import cached_property
from typing import Dict, Any
from core.io_utils import nested_group_to_dict
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
        ch_json = self.trial_data.attrs["channel_metrics_json"]
        raw_dict = json.loads(ch_json)
        ret_metrics = {}
        for ch_name, metrics_data in raw_dict.items():
            ret_metrics[ch_name] = Metrics.from_dict(metrics_data)
        return ret_metrics
    @cached_property
    def metrics(self) -> Metrics:
        m_json = self.trial_data.attrs["metrics_json"]
        raw_dict = json.loads(m_json)
        return Metrics.from_dict(raw_dict)