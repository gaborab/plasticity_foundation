# python
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

from core import io_utils
from core.io_models import Trial, Metrics
import numpy as np
import torch
import json
import h5py

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class EpochIO:
    """
    Simple utility for saving/loading epoch tensors to/from a directory.

    Filename convention: {subject}_{condition}_epoch{index}.pt
    Example: STMD1_left_epoch1.pt
    """

    def __init__(self, ext: Optional[str] = '.pt'):
        self.ext = ext

    def save(self, tensor_data: Any, directory: Path, subject: str, condition: str, epoch_id: int, saved_type = 'epoch') -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        filename = io_utils.vars_to_filename(subject, condition, epoch_id, saved_type, self.ext)
        target = directory / filename
        torch.save(tensor_data.clone(), target)
        return target

    def load_specific(self, directory: Path, subject: str, cond: str, idx: str, marker: str) -> Optional[Any]:
        """
        Load all epoch files in `directory` for a specific `subject`, `cond`, `marker` and `id` matching the filename convention.
        """
        filename = io_utils.vars_to_filename(subject, cond, idx, marker=marker, ext=self.ext)
        # find matching file in directory
        file = directory / filename

        if not file.exists():
            print(f"WARNING: File not found for filename: {filename} in directory: {directory}")
        try:
            return torch.load(file)
        except Exception:
            print(f"WARNING: Failed to load file: {file.name}")
            return None

    def load_all_epochs(self, directory: Path) -> Dict[str, Dict[str, List[tuple[int, Any]]]]:
        """
        Load all epoch files in `directory` matching the filename convention.
        Returns: {subject: {condition: [1: epoch_tensor1, ...]}}
        """
        epochs = {}
        if not directory.exists():
            print("WARNING: Directory does not exist, unable to load epochs:", directory)
            return epochs

        for file in directory.glob(f"*{self.ext}"):
            pos, id, subject, condition = io_utils.vars_from_filename(file.stem, marker="_epoch")

            # load tensor
            try:
                data = torch.load(file)
            except Exception:
                # skip corrupted/unreadable file
                print("WARNING: Failed to load file, skipping:", file.name)
                continue

            subj_dict = epochs.setdefault(subject, {})
            cond_list = subj_dict.setdefault(condition, [])
            cond_list.append((id, data))  # store with ID for potential sorting

        # sort each condition list by epoch_id
        for subj, conds in epochs.items():
            for cond, pairs in conds.items():
                pairs.sort(key=lambda t: t[0])


        return epochs

    def load_subject(self, directory: Path, subject: str) -> Optional[Dict[str, List[Any]]]:
        all_epochs = self.load_all_epochs(directory)
        return all_epochs.get(subject)


def load_model_log_data(directory: Path, model_name: str, folds=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads the log data for a given model and list of folds from the specified directory.
    Expects a folder named `model_name` to exist and contain subfolders for each fold
    with one log file each named log.txt.
    """

    if folds is None:
        folds = []
    all_logs = {}
    model_folder = directory / model_name

    if not model_folder.exists():
        print(f"WARNING: Model folder not found for model name: {model_name} in directory: {directory}")
        return all_logs

    # Determine which folds to process
    keep_all = len(folds) == 0

    for fold_dir in model_folder.iterdir():
        if not fold_dir.is_dir():
            continue

        if not keep_all and fold_dir.name not in folds:
            continue

        log_file = fold_dir / "log.txt"
        if not log_file.exists():
            print(f"WARNING: Log file not found in {fold_dir.name}")
            continue

        fold_data = []
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        fold_data.append(json.loads(line))
            all_logs[fold_dir.name] = fold_data
        except Exception as e:
            print(f"WARNING: Failed to parse log file in {fold_dir.name}: {e}")

    return all_logs


def save_dict_to_hdf(data: Dict, group: h5py.Group):
    for key, value in data.items():
        if isinstance(value, Dict):
            subgroup = group.create_group(key)
            save_dict_to_hdf(value, subgroup)
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value, compression="gzip")
        elif not hasattr(value, "__len__") and not isinstance(value, str):
            value = float(value)
            group.create_dataset(key, data=value)
        else:
            print(f"Found unexpected type {type(value)} for {key}")
            print("WARNING: value is not a numpy ndarray or dict, converting to JSON as fallback")
            value_str = json.dumps(value, cls=NumpyEncoder)
            group.create_dataset(key, data=value_str)



def save_inference_trial_hdf(subject: str, condition: str, idx, h5file: h5py.File, trial_target, trial_output, window_metrics: Dict[str, Tuple[Metrics, Dict[str, Metrics]]]):
    trial_path = f"{subject}/{condition}/{idx}"
    trial_group = h5file.create_group(trial_path)
    trial_group.create_dataset("target", data=trial_target, compression="gzip")
    trial_group.create_dataset("output", data=trial_output, compression="gzip")

    for name, (metrics, ch_metrics) in window_metrics.items():
        trial_group.attrs[f"{name}_metrics_json"] = json.dumps(metrics, cls=NumpyEncoder)
        trial_group.attrs[f"{name}_channel_metrics_json"] = json.dumps(ch_metrics, cls=NumpyEncoder)

def load_inference_data(h5file_path: Path, channel_mode: bool, subjects: str = None, conditions: str = None, idxs = None, results: List[str]=None, metrics: List[str]=None, channels: List[str]=None, bands: List[str] = None) -> Any:

    try:
        data = h5py.File(h5file_path, "r")
    except:
        print(f"ERROR: Failed to open HDF file at path: {h5file_path}")
        return None

    # assuming file structure is {subject}/{condition}/{idx}/ with datasets "target", "output", and group "metrics" containing the metrics as datasets
    if results is not None and "metrics" not in results:
        results.append("metrics")

    master_keys_channel = [subjects, conditions, idxs, results, channels, metrics, bands]
    master_keys_global = [subjects, conditions, idxs, results, metrics, bands]

    master_keys = master_keys_channel if channel_mode else master_keys_global
    loaded_data = {}
    loaded_data = extract_data(data, loaded_data, master_keys, depth=0, channel_mode=channel_mode)

    data.close()

    return loaded_data



def extract_data(group: h5py.File | h5py.Group, ret: Dict, get: List[List[str]], depth, channel_mode: bool) -> Dict:
    if depth >= len(get):
        print("WARNING: Depth exceeds length of get list, returning current data")
        return ret
    level_keys = get[depth]

    if level_keys is None:
        level_keys = list(group.keys())
        # in case of metrics, decide whether to collect channel-wise metrics or global metrics based on channel_mode
        if group.name.split("/")[-1] == "metrics":
            if channel_mode:
                level_keys = [key for key in level_keys if "ch_" in key]
            else:
                level_keys = [key for key in level_keys if "ch_" not in key]

    for key in level_keys:
        if key not in group:
            print(f"WARNING: Key {key} not found in group {group.name}")
            continue
        item = group[key]

        # recurse if group, otherwise save data
        if isinstance(item, h5py.Group):
            if depth + 1 < len(get):
                extract_data(item, ret, get, depth + 1, channel_mode=channel_mode)
        elif isinstance(item, h5py.Dataset):
            ret[item.name] = item[()]
    return ret

def load_inference_trials(h5file_path: Path, subjects: List[str] = None, conditions: List[str] = None, idxs: List[int] = None, exclude_subjects: List[str] = None) -> List[Trial]:
    print("Loading inference trials...")
    try:
        data = h5py.File(h5file_path, "r")
    except:
        print(f"ERROR: Failed to open HDF file at path: {h5file_path}")
        return []

    print(f"Filtering for subjects: {', '.join(subjects) if subjects is not None else 'all'}")
    print(f"Filtering for conditions: {', '.join(conditions) if conditions is not None else 'all'}")
    print(f"Filtering for idxs: {', '.join(map(str,idxs)) if idxs is not None else 'all'}")
    all_trials = []
    for subject, subj_group in data.items():
        if subjects is not None and subject not in subjects:
            continue
        if exclude_subjects is not None and subject in exclude_subjects:
            continue
        for condition, cond_group in subj_group.items():
            if conditions is not None and condition not in conditions:
                continue
            for idx, trial_group in cond_group.items():
                if idxs is not None and idx not in idxs:
                    continue
                trial = Trial(trial_group, subject, condition, idx)
                all_trials.append(trial)
    return all_trials


def save_report(rows, filename: str, save_dir: Path):
    if not rows:
        print("No data to save in report.")
        return
    path = save_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} rows to {path}")


def save_figure(name_input: str, save_dir: Path, fig, default_name="untitled"):
    filename = name_input.strip() or default_name
    if not filename.endswith(".png"):
        filename += ".png"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")

def get_stored_subject_names(data_dir: Path) -> List[str]:
    """
    Returns a list of subject names based on the directory contents of the data_dir.
    Assumes that each subject has a dedicated subdirectory in data_dir.
    """
    if not data_dir.exists():
        print(f"WARNING: Data directory does not exist: {data_dir}")
        return []

    subject_names = [d.name for d in data_dir.iterdir() if d.is_dir()]
    return subject_names
