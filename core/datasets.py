"""Class for handling the dataset for fine-tuning.
This class is designed to work with the epochs generated from the preprocessing step and saved to disk.
It provides a structure for loading and accessing training, validating and testing epochs for fine-tuning the model head.
"""
import random
from pathlib import Path

from core import utils
from core.io import EpochIO
from typing import Dict, List, Any, Optional, Tuple
import torch
import numpy as np

from core.io_models import TargetStats


class FinetuneDataset(torch.utils.data.Dataset):
    """
    This class can be implemented to wrap a list of epoch tensors into a PyTorch Dataset for use in DataLoaders during fine-tuning.
    Expects input dictionaries in the format of EpochIO.load_all_input_target_data: {subject: {condition: [(id1, (input_tensor1, target_tensor1)), ...]}}
    Every trial is split into one sample per patch, e.g. a pre- and a post-stimulus sample, with the patch name in metadata["patch"].
    epoch_window: (start, end) in s of the saved input epochs and targets, the time axis is their last dimension.
    patch_windows: {patch name: (start, end) in s}, the windows the inputs and targets are cut to, each within epoch_window.
    fs: sampling frequency of the inputs and targets.
    masks: {patch name: (freqs, time) boolean mask} of the target bins the loss and metrics use in each patch, see utils.valid_wavelet_bins.
    Without masks every target bin is used.
    target_stats: normalizes the targets returned by __getitem__ if normalize is set, samples keep the dB targets.
    interpolations: {(subject, condition, id): interpolated channel indices}, see io.load_interpolated_channels. Interpolated
    channels are never target channels (metadata["allowed"] is False for them). Without it every channel is allowed.
    seed: seed of the per-sample channel permutations that select the training target channel of every epoch.
    n_eval_channels: number of evaluation target channels per sample, see utils.balanced_eval_channels.
    """

    def __init__(self, epochs_dict: Dict[str, Dict[str, List[tuple[int, Tuple[Any, Any]]]]], epoch_window: Tuple[float, float],
                 patch_windows: Dict[str, Tuple[float, float]], fs: int, target_mode='wavelet', log: bool = True, normalize: bool = True,
                 masks: Optional[Dict[str, np.ndarray]] = None, target_stats: Optional[TargetStats] = None,
                 interpolations: Optional[Dict[Tuple[str, str, int], List[int]]] = None, seed: int = 0, n_eval_channels: int = 3):
        if target_mode not in ('autoencoder', 'wavelet'):
            raise ValueError(f"Unknown target mode '{target_mode}', expected 'autoencoder' or 'wavelet'")
        self.samples = []
        self.perms = []
        self.eval_channels = []
        self.epoch = 0
        self.subjects = list(epochs_dict.keys())
        self.masks = None if masks is None else {patch: torch.as_tensor(mask, dtype=torch.bool) for patch, mask in masks.items()}
        self.normalize = normalize
        self.target_stats = target_stats
        for subj, conds in epochs_dict.items():
            for cond, epoch_list in conds.items():
                for idx, (input_tensor, target_tensor) in epoch_list:
                    # build interpolation mask for channel masking
                    if interpolations is not None:
                        allowed = torch.ones(input_tensor.shape[0], dtype=torch.bool)
                        for ch in interpolations.get((subj, cond, idx), []):
                            allowed[ch] = False
                    else:
                        allowed = torch.ones(input_tensor.shape[0], dtype=torch.bool)

                    metadata = {
                        "subject": subj,
                        "condition": cond,
                        "id": idx,
                        "allowed": allowed,
                    }

                    epoch_data = input_tensor.clone().detach()
                    target_data = None
                    if target_mode == 'wavelet':
                        target_data = target_tensor.clone().detach().float()  # targets are stored as float16
                        # log scale wavelet magnitudes
                        if log: target_data = 20*torch.log10(target_data + 1e-7) # add small value to avoid log(0)

                    for patch, patch_window in patch_windows.items():
                        patch_input = utils.cut_to_timewindow(epoch_data, epoch_window, patch_window, fs)
                        patch_target = patch_input if target_mode == 'autoencoder' else utils.cut_to_timewindow(target_data, epoch_window, patch_window, fs)
                        self.samples.append((patch_input, patch_target, {**metadata, "patch": patch}))
                        self.perms.append(utils.channel_permutation((subj, cond, idx, patch), allowed, seed=seed))
        # fixed evaluation target channels, every channel evaluated about equally often over the dataset
        all_allowed = [metadata["allowed"] for _, _, metadata in self.samples]
        self.eval_channels = [torch.as_tensor(chs) for chs in
                              utils.balanced_eval_channels(all_allowed, n_eval_channels)]

    @property
    def target_dim(self) -> int:
        """Returns the number of elements of one sample's target (flattened size), 0 for an empty dataset."""
        if len(self.samples) == 0:
            return 0
        return self.samples[0][1].numel()

    def __len__(self):
        """Returns the number of samples, one per patch of every trial."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns (input, target, metadata, mask, target_ch, eval_chs) of one patch sample:
        mask is a boolean tensor of the target's shape marking the bins the loss and metrics use,
        target_ch the training target channel of the current epoch (see set_epoch), from the sample's fixed channel permutation,
        eval_chs the fixed evaluation target channels of the sample.
        """
        input_tensor, target_tensor, metadata = self.samples[idx]
        if self.normalize and self.target_stats is not None:
            target_tensor = self.target_stats.normalize(target_tensor)

        # include all target bins if no masks are provided (autoencoding), otherwise use the mask of the patch
        if self.masks is None:
            mask = torch.ones_like(target_tensor, dtype=torch.bool)
        else:
            # (freqs, time) mask of the patch, repeated over channels
            mask = self.masks[metadata["patch"]].expand_as(target_tensor)
        perm = self.perms[idx]
        target_ch = int(perm[self.epoch % len(perm)])
        return input_tensor, target_tensor, metadata, mask, target_ch, self.eval_channels[idx]

    def masked_targets(self):
        """Yields (dB target, mask, metadata) of every sample, with the mask of its patch."""
        for _, target, metadata in self.samples:
            yield target, self.masks[metadata["patch"]], metadata

    def set_epoch(self, epoch: int):
        """Sets the epoch number for the dataset, used to change the channel permutation every epoch."""
        self.epoch = epoch


class InjectionDataset(torch.utils.data.Dataset):
    """
    One patch of every trial with and without gain injections, for the injection test of masked-channel reconstruction.
    Expects input dictionaries in the format of EpochIO.load_all_input_target_data, only the input epochs are used: the
    targets are recomputed from the (injected) inputs, so the uninjected and injected targets go through the same code.
    epoch_window: (start, end) in s of the saved input epochs, stimulus onset at 0. patch_window: the patch to return, e.g. the post patch.
    topographies: (n_chans, n_sources) topographies of the injected sources, gain_sets: one (n_sources,) gain vector per
    injection; the sources get stronger after onset, ramped in over ramp_duration s (utils.inject_sources).
    freqs, pad, wavelet_params: wavelet transform of the patch as in make_dataset.patch_wavelet_target (utils.calc_wavelet).
    interpolations: {(subject, condition, id): interpolated channel indices}, metadata["allowed"] is False for them.
    target_stats: the fold's target normalization, converts the model outputs back to dB (the targets are returned in dB).
    __getitem__ returns (inputs, targets, metadata): inputs (1 + n_injections, n_chans, patch samples) and dB targets
    (1 + n_injections, n_chans, n_freqs, patch samples), index 0 uninjected and index i + 1 injected with gain_sets[i].
    """

    def __init__(self, epochs_dict: Dict[str, Dict[str, List[tuple[int, Tuple[Any, Any]]]]], epoch_window: Tuple[float, float],
                 patch_window: Tuple[float, float], fs: int, topographies: np.ndarray, gain_sets: List[np.ndarray], ramp_duration: float,
                 freqs: np.ndarray, pad: int, wavelet_params: Dict[str, float],
                 interpolations: Optional[Dict[Tuple[str, str, int], List[int]]] = None, target_stats: Optional[TargetStats] = None):
        self.epoch_window, self.patch_window, self.fs = epoch_window, patch_window, fs
        self.topographies, self.gain_sets, self.ramp_duration = topographies, gain_sets, ramp_duration
        self.freqs, self.pad, self.wavelet_params = freqs, pad, wavelet_params
        self.target_stats = target_stats
        self.samples = []
        for subj, conds in epochs_dict.items():
            for cond, epoch_list in conds.items():
                for idx, (input_tensor, _) in epoch_list:
                    allowed = torch.ones(input_tensor.shape[0], dtype=torch.bool)
                    if interpolations is not None:
                        allowed[interpolations.get((subj, cond, idx), [])] = False
                    metadata = {"subject": subj, "condition": cond, "id": idx, "allowed": allowed}
                    self.samples.append((input_tensor.float().numpy(), metadata))

    def __len__(self):
        """Returns the number of trials."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (inputs, targets, metadata) of one trial's patch, uninjected first, then one per gain set."""
        epoch, metadata = self.samples[idx]
        times = self.epoch_window[0] + np.arange(epoch.shape[-1]) / self.fs
        epochs = [epoch] + [utils.inject_sources(epoch, self.topographies, gains, times, self.ramp_duration).astype(np.float32)
                            for gains in self.gain_sets]
        inputs, targets = [], []
        for x in epochs:
            patch = utils.cut_to_timewindow(torch.from_numpy(x), self.epoch_window, self.patch_window, self.fs)
            mag, _ = utils.calc_wavelet(patch, fs=self.fs, freqs=self.freqs, pad=self.pad, **self.wavelet_params)
            # (freqs, channels, time) -> (channels, freqs, time) in dB, as make_dataset and FinetuneDataset
            targets.append(20 * torch.log10(torch.tensor(mag, dtype=torch.float32).permute(1, 0, 2) + 1e-7))
            inputs.append(patch)
        return torch.stack(inputs), torch.stack(targets), metadata


class FinetuneSplits:
    """
    This class can be implemented to hold the training, validation, and testing epochs for a single fold of cross-validation.
    The epoch dicts use the {subject: {condition: [(id, (input, target)), ...]}} format of FinetuneDataset.
    keys: {"train": [...], "val": [...], "test": [...]}, the (subject, condition, id) keys of the epochs in every split,
    saved with the fold so inference can rebuild the same split.
    """

    def __init__(self, training_dict: Dict, validation_dict: Dict, testing_dict: Dict, keys: Dict[str, List[Tuple[str, str, int]]] = None):
        self.training_epochs = training_dict
        self.validation_epochs = validation_dict
        self.testing_epochs = testing_dict
        self.target_stats: Optional[TargetStats] = None  # set by initialize_datasets
        self.keys = keys  # store the keys for each split

    def describe(self) -> str:
        """
        Returns a summary of the fold: trials, subjects and conditions of every split. Subjects are listed by name for the
        validation and testing splits of a by-subject split. For a within-subject split, the (subject, condition) groups
        missing from a split are listed instead.
        """
        splits = {"train": self.training_epochs, "val": self.validation_epochs, "test": self.testing_epochs}
        groups = {name: {(subject, cond) for subject, conds in epochs.items() for cond in conds} for name, epochs in splits.items()}
        within_subjects = bool(set(self.training_epochs) & set(self.testing_epochs))
        lines = []
        for name, epochs in splits.items():
            trials_per_subject = {subject: sum(len(pairs) for pairs in conds.values()) for subject, conds in epochs.items()}
            n_conditions = len({cond for _, cond in groups[name]})
            line = f"  {name:5s}: {sum(trials_per_subject.values())} trials, {len(trials_per_subject)} subjects, {n_conditions} conditions"
            if trials_per_subject:
                line += f", {min(trials_per_subject.values())}-{max(trials_per_subject.values())} trials per subject"
            if not within_subjects and name != "train":
                line += f": {', '.join(sorted(epochs))}"
            lines.append(line)
        if within_subjects:
            all_groups = set().union(*groups.values())
            for name in splits:
                missing = sorted(all_groups - groups[name])
                if missing:
                    lines.append(f"  WARNING: {len(missing)} (subject, condition) groups have no {name} trials: "
                                 + ", ".join(f"{subject}/{cond}" for subject, cond in missing))
        return "\n".join(lines)

    def initialize_datasets(self, patch_params: Dict[str, Any], target_mode='wavelet', log=True, normalize=True,
                            masks: Optional[Dict[str, np.ndarray]] = None, target_stats: Optional[TargetStats] = None,
                            interpolations: Optional[Dict[Tuple[str, str, int], List[int]]] = None, seed: int = 0,
                            n_eval_channels: int = 3) -> Tuple[FinetuneDataset, FinetuneDataset, FinetuneDataset]:
        """
        Builds the training, validation and testing FinetuneDataset of this fold.
        patch_params: the time window arguments of FinetuneDataset, {"epoch_window": ..., "patch_windows": ..., "fs": ...}.
        target_stats: the stats all three datasets normalize their targets with. If None, they are fitted on the masked
        training targets of the allowed channels (wavelet mode with masks only). The stats used are stored in self.target_stats.
        interpolations, seed, n_eval_channels: see FinetuneDataset.
        """
        datasets = [FinetuneDataset(epochs, **patch_params, target_mode=target_mode, log=log, normalize=normalize, masks=masks,
                                    interpolations=interpolations, seed=seed, n_eval_channels=n_eval_channels)
                    for epochs in (self.training_epochs, self.validation_epochs, self.testing_epochs)]
        if normalize and target_stats is None and target_mode == 'wavelet':
            if masks is None:
                print("WARNING: no masks given, unable to fit target stats, targets are not normalized")
            else:
                target_stats = TargetStats.fit(datasets[0].masked_targets())
        self.target_stats = target_stats
        for dataset in datasets:
            dataset.target_stats = target_stats

        training_dataset, validation_dataset, testing_dataset = datasets
        return training_dataset, validation_dataset, testing_dataset

class KFoldSplitter:
    """
    Build a list of `FinetuneSplits` instances, one per fold.
    The constructor loads the input epochs of `input_data_dir` paired with their targets in `target_data_dir` using `EpochIO`.
    target_mode: "wavelet" loads the wavelet targets, "autoencoder" uses every input as its own target and does not read target_data_dir.
    seed: seed of the shuffling, the same seed and data give the same folds.
    no_press_threshold: first subject number of the paradigm without button presses, split_by_subject stratifies by paradigm.
    """

    def __init__(self, input_data_dir: Path, target_data_dir: Optional[Path], target_mode: str, seed: int, k_folds: int = 6, no_press_threshold: int = 59):
        io_handler = EpochIO(ext='.pt')
        # loads data in format {subject: {condition: [(id1, (input_tensor1, target_tensor1)), ...]}}

        self.input_target_data = io_handler.load_all_input_target_data(input_dir=input_data_dir, target_dir=target_data_dir, target_mode=target_mode)
        self.k_folds = k_folds
        self.seed = seed
        self.no_press_threshold = no_press_threshold

    def split_by_subject(self) -> List[FinetuneSplits]:
        """
        This method can be implemented to split the dataset into K folds for cross-validation based on subjects, ensuring that all epochs from a given subject are in the same fold.
        Used for fine-tuning generalizable models.
        Returns: a list of `FinetuneSplits` instances, one per fold.
        """
        subject_paradigm = utils.subject_paradigm
        rng = random.Random(self.seed)
        # Split subjects into K folds, stratify by button press paradigm
        groups = {}
        for subject in sorted(self.input_target_data.keys()):
            button_press = subject_paradigm(subject, thresh=self.no_press_threshold)
            groups.setdefault(button_press, []).append(subject)
        grouped_shuffle_subjects = []
        for paradigm, p_subjects in groups.items():
            rng.shuffle(p_subjects)
            grouped_shuffle_subjects.extend(p_subjects)

        folds = [grouped_shuffle_subjects[i::self.k_folds] for i in range(self.k_folds)]
        # Iterate over folds to create train/val/test splits for each fold
        splits = []
        for i in range(self.k_folds):
            test_subj = folds[i]
            train_val_subj = [s for s in grouped_shuffle_subjects if s not in test_subj]
            val_subj = []
            for paradigm in groups.keys():
                paradigm_subj = [s for s in train_val_subj if subject_paradigm(s, thresh=self.no_press_threshold) == paradigm]
                rng.shuffle(paradigm_subj)
                val_subj.extend(paradigm_subj[:len(paradigm_subj)//5])  # 20% for validation
            train_subj = [s for s in train_val_subj if s not in val_subj]
            train = {subject: self.input_target_data[subject] for subject in train_subj}
            val = {subject: self.input_target_data[subject] for subject in val_subj}
            test = {subject: self.input_target_data[subject] for subject in test_subj}
            keys = {"train": self._keys_of(train), "val": self._keys_of(val), "test": self._keys_of(test)}
            splits.append(FinetuneSplits(training_dict=train, validation_dict=val, testing_dict=test, keys=keys))

        return splits

    def split_within_subjects(self) -> List[FinetuneSplits]:
        """
        Splits the epochs of every subject into K folds for cross-validation within subjects, so every subject is in the training, validation and testing data of each fold.
        Epochs are identified by (subject, condition, id) keys. Split by slicing like split_by_subject, stratified by (subject, condition) group:
        the keys are shuffled within every group and concatenated, so slicing every K-th key puts ~1/K of every subject's every condition into each fold.
        In fold i, fold i is the test set and 20% of every group's remaining epochs is the validation set.
        Used for fine-tuning personalized models.
        Returns: a list of `FinetuneSplits` instances, one per fold.
        """
        rng = random.Random(self.seed)
        # Shuffle the epoch keys within every (subject, condition) group
        groups = {}
        for subject in sorted(self.input_target_data):
            for cond in sorted(self.input_target_data[subject]):
                # collect subject, cond, idx as identifiers grouped by (subject, cond) for stratification
                groups[(subject, cond)] = [(subject, cond, idx) for idx, _ in self.input_target_data[subject][cond]]
        grouped_shuffle_keys = []
        for group_keys in groups.values():
            rng.shuffle(group_keys)
            grouped_shuffle_keys.extend(group_keys)

        folds = [grouped_shuffle_keys[i::self.k_folds] for i in range(self.k_folds)]
        # Iterate over folds to create train/val/test splits for each fold
        splits = []
        for i in range(self.k_folds):
            test_keys = folds[i]
            test_set = set(test_keys)
            val_keys = []
            for group_keys in groups.values():
                group_train_val = [key for key in group_keys if key not in test_set]
                rng.shuffle(group_train_val)
                val_keys.extend(group_train_val[:len(group_train_val)//5])  # 20% for validation
            val_set = set(val_keys)
            train_keys = [key for key in grouped_shuffle_keys if key not in test_set and key not in val_set]
            keys = {"train": train_keys, "val": val_keys, "test": test_keys}
            splits.append(FinetuneSplits(training_dict=self._select(train_keys), validation_dict=self._select(val_keys),
                                         testing_dict=self._select(test_keys), keys=keys))
        return splits

    def split_from_keys(self, keys: List[Dict[str, List[Tuple[str, str, int]]]]) -> List[FinetuneSplits]:
        """
        Rebuilds the FinetuneSplits of a training run from its saved keys (io.load_training_split_keys), one per fold.
        keys: one {"train": [...], "val": [...], "test": [...]} dict of (subject, condition, id) keys per fold.
        Warns if saved keys are missing from the loaded data, e.g. after the dataset was rebuilt.
        """
        loaded = set(self._keys_of(self.input_target_data))
        splits = []
        for i, fold in enumerate(keys):
            missing = {split: len(set(split_keys) - loaded) for split, split_keys in fold.items()}
            if any(missing.values()):
                print(f"WARNING: fold {i + 1} has saved keys missing from the loaded data: {missing}")
            splits.append(FinetuneSplits(training_dict=self._select(fold["train"]), validation_dict=self._select(fold["val"]),
                                         testing_dict=self._select(fold["test"]), keys=fold))
        return splits

    @staticmethod
    def _keys_of(epochs: Dict[str, Dict[str, List[tuple[int, Any]]]]) -> List[Tuple[str, str, int]]:
        """
        Returns the (subject, condition, id) keys of all epochs in the {subject: {condition: [(id, (input, target)), ...]}} format.
        """
        return [(subject, cond, idx) for subject, conds in epochs.items() for cond, epoch_list in conds.items() for idx, _ in epoch_list]

    def _select(self, keys: List[Tuple[str, str, Any]]) -> Dict[str, Dict[str, List[tuple[int, Any]]]]:
        """
        Returns the epochs with the given (subject, condition, id) keys in the {subject: {condition: [(id, (input, target)), ...]}} format of FinetuneDataset.
        """
        keys = set(keys)
        selected = {}
        for subject, conds in self.input_target_data.items():
            for cond, epoch_list in conds.items():
                kept = [(idx, epoch) for idx, epoch in epoch_list if (subject, cond, idx) in keys]
                if kept:
                    selected.setdefault(subject, {})[cond] = kept
        return selected

