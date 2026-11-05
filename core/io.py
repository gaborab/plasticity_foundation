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

    Filename convention: {subject}_{condition}_{saved type}{index}.pt, the saved type is the marker the files are loaded by.
    Example: STMD1_left_input1.pt (input epoch, marker "_input"), STMD1_left_wavelet1.pt (wavelet target, marker "_wavelet")
    """

    def __init__(self, ext: Optional[str] = '.pt'):
        self.ext = ext

    def save(self, tensor_data: Any, directory: Path, subject: str, condition: str, epoch_id: int,
             saved_type='epoch') -> Path:
        """
        Saves a copy of tensor_data to directory as {subject}_{condition}_{saved_type}{epoch_id}{ext} and returns its path.
        """
        directory.mkdir(parents=True, exist_ok=True)
        filename = io_utils.vars_to_filename(subject, condition, epoch_id, saved_type, self.ext)
        target = directory / filename
        torch.save(tensor_data.clone(), target)
        return target

    def load_all_epochs(self, directory: Path, marker: str = '_epoch') -> Dict[str, Dict[str, List[tuple[int, Any]]]]:
        """
        Load all epoch files in `directory` matching the filename convention, `marker` is the part before the epoch id, e.g. "_input".
        Unreadable files are skipped with a warning.
        Returns: {subject: {condition: [(id1, epoch_tensor1), ...]}} sorted by id
        """
        epochs = {}
        if not directory.exists():
            print("WARNING: Directory does not exist, unable to load epochs:", directory)
            return epochs

        # sorted, as glob order depends on the OS and file system
        for file in sorted(directory.glob(f"*{self.ext}")):
            pos, id, subject, condition = io_utils.vars_from_filename(file.stem, marker=marker)

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

    def load_subject(self, directory: Path, subject: str, marker: str = '_epoch') -> Optional[
        Dict[str, List[tuple[int, Any]]]]:
        """
        Returns the epochs of one subject in `directory` as {condition: [(id1, epoch_tensor1), ...]}, None if there are none.
        """
        all_epochs = self.load_all_epochs(directory, marker=marker)
        return all_epochs.get(subject)

    def load_all_input_target_data(self, input_dir: Path, target_dir: Optional[Path], target_mode: str) -> Dict[
        str, Dict[str, List[tuple[int, Tuple[Any, Any]]]]]:
        """
        Loads the input epochs in `input_dir` and pairs each with the target of the same (subject, condition, id) in `target_dir`.
        target_mode selects the target files by their marker, e.g. "wavelet" loads {subject}_{condition}_wavelet{id}.pt,
        "autoencoder" uses every input as its own target and does not read `target_dir`.
        Inputs without a target are skipped with a warning.
        Returns: {subject: {condition: [(id1, (input_tensor1, target_tensor1)), ...]}} sorted by id, the (id, data) format of load_all_epochs.
        """
        input_data = self.load_all_epochs(input_dir, marker="_input")
        if target_mode == "autoencoder":
            return {subject: {cond: [(idx, (epoch, epoch)) for idx, epoch in pairs] for cond, pairs in conds.items()}
                    for subject, conds in input_data.items()}

        target_data = self.load_all_epochs(target_dir, marker=f"_{target_mode}")
        targets = {(subject, cond, idx): target for subject, conds in target_data.items()
                   for cond, pairs in conds.items() for idx, target in pairs}

        data = {}
        for subject, conds in input_data.items():
            for cond, pairs in conds.items():
                for idx, input_tensor in pairs:
                    target_tensor = targets.get((subject, cond, idx))
                    if target_tensor is None:
                        print(f"WARNING: Missing {target_mode} target for {subject}, {cond}, epoch {idx}, skipping")
                        continue
                    data.setdefault(subject, {}).setdefault(cond, []).append((idx, (input_tensor, target_tensor)))
        return data


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


def save_time_series_metrics_hdf(subject: str, condition: str, idx: int, patch: str, h5file: h5py.File,
                                 time_series_metrics: Dict[str, Dict[str, np.ndarray]]):
    """
    Saves the time series metrics of one patch of a trial as {subject}/{condition}/{idx}/{patch}/{metric}/{band}, one (channels, time) dataset per band,
    the layout of the inference files (see save_inference_trial_hdf).
    Expects time_series_metrics as {metric: {band: (channels, time) array}}, e.g. {"mae": calc_metric_time_series(trial, "mae", band_indices)}.
    """
    trial_path = f"{subject}/{condition}/{idx}/{patch}"
    trial_group = h5file.create_group(trial_path)

    for metric_name, metric_bands in time_series_metrics.items():
        metric_group = trial_group.create_group(metric_name)
        for band, metric_array in metric_bands.items():
            metric_group.create_dataset(band, data=metric_array)


def save_inference_trial_hdf(subject: str, condition: str, idx: int, h5file: h5py.File, trial_target, trial_output,
                             window_metrics: Dict[str, Tuple[Metrics, Dict[str, Metrics]]], patch: str = None,
                             trial_occlusion_output=None):
    trial_path = f"{subject}/{condition}/{str(idx)}/{patch}" if patch is not None else f"{subject}/{condition}/{str(idx)}"
    trial_group = h5file.create_group(trial_path)
    # a class id is a scalar, which h5py can't compress
    trial_group.create_dataset("target", data=trial_target, compression="gzip" if np.ndim(trial_target) else None)
    trial_group.create_dataset("output", data=trial_output, compression="gzip")
    if trial_occlusion_output is not None:
        trial_group.create_dataset("occlusion", data=trial_occlusion_output, compression="gzip")

    for name, (metrics, ch_metrics) in window_metrics.items():
        trial_group.attrs[f"{name}_metrics_json"] = json.dumps(metrics, cls=NumpyEncoder)
        trial_group.attrs[f"{name}_channel_metrics_json"] = json.dumps(ch_metrics, cls=NumpyEncoder)


def load_training_split_keys(path: Path, k_folds: int, filename: str = "keys.json") -> List[
    Dict[str, List[Tuple[str, str, int]]]]:
    """
    Loads the keys.json of every fold of a training run in path (folders 1..k_folds) and returns one
    {"train": [...], "val": [...], "test": [...]} dict of (subject, condition, id) keys per fold.
    JSON stores the keys as lists, they are converted back to tuples so they can be compared with the epoch keys.
    """
    result = []
    for i in range(k_folds):
        with open(path / str(i + 1) / filename, "r", encoding="utf-8") as f:
            fold_keys = json.load(f)
        result.append({split: [tuple(key) for key in keys] for split, keys in fold_keys.items()})
    return result


def load_inference_trials(h5file_path: Path, subjects: List[str] = None, conditions: List[str] = None,
                          idxs: List[int] | List[str] = None, exclude_subjects: List[str] = None, window: str = "full",
                          time_series_path: Optional[Path] = None, stim_patch: Optional[str] = "post") -> List[Trial]:
    """
    Loads the trials of an inference HDF5 file as lazy Trial objects, keeping the file open while they are in use.
    subjects, conditions, idxs: only keep these (None keeps all). idxs are compared with the HDF5 keys, so pass them as strings.
    exclude_subjects: drop these subjects.
    window: key of settings.METRIC_WINDOWS, the time window whose metrics trial.metrics / trial.channel_metrics return.
    time_series_path: optional HDF5 file of time series metrics (see save_time_series_metrics_hdf). Its matching trial groups are attached
    as trial.time_series_data, so trial.time_series(metric) can read them. Trials missing from it are kept without time series data.
    stim_patch: the patch of every trial to load, "pre" or "post" (also the patch of the attached time series).
    None for files without a patch level (class runs, {subject}/{condition}/{id}): the trial group itself is loaded.
    """
    print("Loading inference trials...")
    try:
        data = h5py.File(h5file_path, "r")
    except:
        print(f"ERROR: Failed to open HDF file at path: {h5file_path}")
        return []
    ts_data = None
    if time_series_path is not None:
        try:
            ts_data = h5py.File(time_series_path, "r")
        except:
            print(f"ERROR: Failed to open time series HDF file at path: {time_series_path}")
            data.close()
            return []

    print(f"Filtering for subjects: {', '.join(subjects) if subjects is not None else 'all'}")
    print(f"Filtering for conditions: {', '.join(conditions) if conditions is not None else 'all'}")
    print(f"Filtering for idxs: {', '.join(map(str, idxs)) if idxs is not None else 'all'}")
    all_trials = []
    n_missing_ts = 0
    for subject, subj_group in data.items():
        if subjects is not None and subject not in subjects:
            continue
        if exclude_subjects is not None and subject in exclude_subjects:
            continue
        for condition, cond_group in subj_group.items():
            if conditions is not None and condition not in conditions:
                continue
            for idx, trial_group in cond_group.items():
                if idxs is not None and idx not in {str(i) for i in idxs}:
                    continue
                trial = Trial(trial_group[stim_patch] if stim_patch is not None else trial_group, subject, condition, idx, window=window)

                if ts_data is not None:
                    try:
                        trial.time_series_data = ts_data[f"{subject}/{condition}/{idx}/{stim_patch}"]
                    except KeyError:
                        n_missing_ts += 1

                all_trials.append(trial)
    if n_missing_ts > 0:
        print(f"WARNING: {n_missing_ts}/{len(all_trials)} trials have no time series data in {time_series_path}")
    return all_trials


def load_interpolated_channels(manifest_path: Path) -> Dict[Tuple[str, str, int], List[int]]:
    """
    Returns the interpolated channels of every trial in an epochs_manifest.csv written by make_dataset.py as
    {(subject, stimulus, id): sorted channel indices}, the indices into the channel order of the saved epochs (CHORDER_STANDARD).
    A trial's interpolated channels are the union of the channels interpolated for its whole recording (rec_interpolated)
    and those autoreject interpolated in the epoch itself (ar_interpolated). Trials without interpolation get an empty list.
    """
    manifest = pd.read_csv(manifest_path, dtype={"rec_interpolated": str, "ar_interpolated": str},
                           keep_default_na=False)
    interpolated = {}
    for row in manifest.itertuples(index=False):
        channels = {int(ch) for column in (row.rec_interpolated, row.ar_interpolated) for ch in column.split(';') if ch}
        interpolated[(row.subject, row.stimulus, int(row.id))] = sorted(channels)
    return interpolated


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
    if not filename.endswith(".svg"):
        filename += ".svg"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_path = save_dir / filename
        fig.savefig(save_path, bbox_inches='tight')
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


def load_run_settings(model_dir: Path, opts) -> Dict[str, Any]:
    """
    Returns the settings of the training run the test has to match ({"masked", "n_neighbors", "k_folds", "hide_interpolated"})
    from the run's metadata.json, hide_interpolated False for runs that don't record it. Only wavelet models are supported.
    """
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        hyperparameters = json.load(f)["hyperparameters"]

    if hyperparameters.get("target_mode") is None:
        print(f"WARNING: target_mode is None in training run settings,using CLI provided target mode {opts.target_mode} as fallback.")
        hyperparameters["target_mode"] = opts.target_mode
    if hyperparameters.get("k_folds") is None:
        print(f"WARNING: k_folds is None in training run settings,using CLI provided k_folds {opts.k_folds} as fallback.")
        hyperparameters["k_folds"] = opts.k_folds

    if hyperparameters.get("target_mode") != "wavelet":
        raise ValueError(f"Only wavelet models are supported, the run's target mode is {hyperparameters.get('target_mode')}")
    # the targets are computed with the current settings: a different frequency window keeps the output shape, so the
    # model's rows would silently be compared with other frequencies. Runs that don't record these are not checked
    if not opts.from_csv:
        current = {"freq_window": list(settings.SPECTRAL_DECODER_PARAMS["freq_window"]), "target_pool": settings.TARGET_POOL}
        recorded = {name: hyperparameters[name] for name in current if name in hyperparameters}
        mismatched = {name: (value, current[name]) for name, value in recorded.items() if list(np.atleast_1d(value)) != list(np.atleast_1d(current[name]))}
        if mismatched:
            raise ValueError(f"The run was trained with other target settings than the current ones (trained, current): {mismatched}")
        if len(recorded) < len(current):
            print(f"WARNING: the run records no {', '.join(set(current) - set(recorded))}, the current settings are assumed to match")
    return {**{name: hyperparameters[name] for name in ("masked", "n_neighbors", "k_folds")},
            "hide_interpolated": hyperparameters.get("hide_interpolated", False)}
