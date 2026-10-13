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


class FinetuneDataset(torch.utils.data.Dataset):
    """
    This class can be implemented to wrap a list of epoch tensors into a PyTorch Dataset for use in DataLoaders during fine-tuning.
    Expects input dictionaries in the format: {subject: {condition: [epoch_tensor1, epoch_tensor2, ...]}}
    """

    def __init__(self, epochs_dict: Dict[str, Dict[str, List[tuple[int, Any]]]], window_size: int, target_mode='wavelet', wavelet_dir: Optional[Path] = None, superlet_dir: Optional[Path] = None, log: bool = True, normalize: bool = True):
        self.samples = []
        self.subjects = list(epochs_dict.keys())
        for subj, conds in epochs_dict.items():
            for cond, epoch_list in conds.items():
                for idx, epoch in epoch_list:
                    metadata = {
                        "subject": subj,
                        "condition": cond,
                        "id": idx
                    }

                    # tensor_data = torch.tensor(epoch, dtype=torch.float32)
                    epoch_data = epoch.clone().detach()
                    # cut last time point for matching LaBraM expectations
                    epoch_data = utils.cut_to_size(epoch_data, window_size)

                    if target_mode == 'autoencoder':
                        self.samples.append((epoch_data, epoch_data, metadata))

                    if target_mode == 'wavelet':
                        if wavelet_dir is None:
                            print("WARNING: wavelet_dir not provided, unable to load wavelet targets for spectral decoding.")
                            break
                        io_handler = EpochIO(ext='.pt')

                        mags = io_handler.load_specific(directory=wavelet_dir, subject=subj, cond=cond,
                                                        idx=str(idx), marker="wavelet")
                        if mags is None:
                            print(f"WARNING: wavelet target not found for subject: {subj}, condition: {cond}, id: {idx}")
                            continue
                        mags_data = mags.clone().detach()
                        mags_data = utils.cut_to_size(mags_data, window_size)
                        # log scale wavelet magnitudes
                        if log: mags_data = 20*torch.log10(mags_data + 1e-7) # add small value to avoid log(0)
                        if normalize: mags_data = (mags_data - mags_data.mean()) / mags_data.std()
                        self.samples.append((epoch_data, mags_data, metadata))


    @property
    def target_dim(self) -> int:
        if len(self.samples) == 0:
            return 0
        # Returns the total number of elements in the target tensor (Flattened size)
        return self.samples[0][1].numel()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class FinetuneSplits:
    """
    This class can be implemented to hold the training, validation, and testing epochs for a single fold of cross-validation.

    """

    def __init__(self, training_dict: Dict, validation_dict: Dict, testing_dict: Dict):
        self.training_epochs = training_dict
        self.validation_epochs = validation_dict
        self.testing_epochs = testing_dict

    def initialize_datasets(self, target_mode='wavelet', window_size: int = 200, wavelet_dir: Optional[Path] = None, superlet_dir: Optional[Path] = None, log=True, normalize=True):
        training_dataset = FinetuneDataset(self.training_epochs, target_mode=target_mode, wavelet_dir=wavelet_dir, superlet_dir=superlet_dir, window_size = window_size)
        validation_dataset = FinetuneDataset(self.validation_epochs, target_mode=target_mode, wavelet_dir=wavelet_dir, superlet_dir=superlet_dir, window_size = window_size)
        testing_dataset = FinetuneDataset(self.testing_epochs, target_mode=target_mode, wavelet_dir=wavelet_dir, superlet_dir=superlet_dir, window_size=window_size, log=log, normalize=normalize)

        return training_dataset, validation_dataset, testing_dataset


class KFoldSplitter:
    """
    Build a list of `FinetuneSplit` instances, one per fold.
    If `all_epochs` is None the constructor loads from disk using `EpochIO`
    and `settings.FINETUNE_DIR`.
    """

    def __init__(self, input_data_dir: Optional[Path], target_data_dir: Optional[Path], seed: int, input_data: Optional[Dict[str, Dict[str, list]]] = None, k_folds: int = 6, no_press_threshold: int = 59):
        if input_data is None:
            io_handler = EpochIO(ext='.pt')
            # loads data in format {subject: {condition: [(id1, epoch_tensor1), (id2, epoch_tensor2), ...]}}
            if input_data_dir is None:
                print("WARNING: data_dir not provided, unable to load epochs for KFoldSplitter.")
                self.all_data = {}

            input_data = io_handler.load_all_epochs(input_data_dir)
            target_data = io_handler.load_all_epochs(target_data_dir)
        self.all_data = input_data
        self.k_folds = k_folds
        self.seed = seed
        self.no_press_threshold = no_press_threshold

    def split_by_subject(self) -> List[FinetuneSplits]:
        """
        This method can be implemented to split the dataset into K folds for cross-validation based on subjects, ensuring that all epochs from a given subject are in the same fold.
        Used for fine-tuning generalizable models.
        Returns: a list of `FinetuneDataset` instances, one per fold.
        """
        subject_paradigm = utils.subject_paradigm
        rng = random.Random(self.seed)
        # Split subjects into K folds, stratify by button press paradigm
        groups = {}
        for subject in sorted(self.all_data.keys()):
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
            train = {subject: self.all_data[subject] for subject in train_subj}
            val = {subject: self.all_data[subject] for subject in val_subj}
            test = {subject: self.all_data[subject] for subject in test_subj}
            splits.append(FinetuneSplits(training_dict=train, validation_dict=val, testing_dict=test))

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
        for subject in sorted(self.all_data):
            for cond in sorted(self.all_data[subject]):
                # collect subject, cond, idx as identifiers grouped by (subject, cond) for stratification
                groups[(subject, cond)] = [(subject, cond, idx) for idx, _ in self.all_data[subject][cond]]
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
            splits.append(FinetuneSplits(training_dict=self._select(train_keys), validation_dict=self._select(val_keys),
                                         testing_dict=self._select(test_keys)))
        return splits

    def _select(self, keys: List[Tuple[str, str, Any]]) -> Dict[str, Dict[str, List[tuple[int, Any]]]]:
        """
        Returns the epochs with the given (subject, condition, id) keys in the {subject: {condition: [(id, epoch), ...]}} format of FinetuneDataset.
        """
        keys = set(keys)
        selected = {}
        for subject, conds in self.all_data.items():
            for cond, epoch_list in conds.items():
                kept = [(idx, epoch) for idx, epoch in epoch_list if (subject, cond, idx) in keys]
                if kept:
                    selected.setdefault(subject, {})[cond] = kept
        return selected

