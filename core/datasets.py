"""Class for handling the dataset for fine-tuning.
This class is designed to work with the epochs generated from the preprocessing step and saved to disk.
It provides a structure for loading and accessing training, validating and testing epochs for fine-tuning the model head.
"""
import random
from pathlib import Path

from core import utils
from core.epoch_io import EpochIO
from typing import Dict, List, Any, Optional
import torch


class FinetuneDataset(torch.utils.data.Dataset):
    """
    This class can be implemented to wrap a list of epoch tensors into a PyTorch Dataset for use in DataLoaders during fine-tuning.
    Expects input dictionaries in the format: {subject: {condition: [epoch_tensor1, epoch_tensor2, ...]}}
    """

    def __init__(self, epochs_dict: Dict[str, Dict[str, List[tuple[int, Any]]]], window_size: int, target_mode='wavelet', wavelet_dir: Optional[Path] = None, superlet_dir: Optional[Path] = None):
        self.samples = []
        self.subjects = list(epochs_dict.keys())
        for subj, conds in epochs_dict.items():
            for cond, epoch_list in conds.items():
                for idx, epoch in epoch_list:
                    # tensor_data = torch.tensor(epoch, dtype=torch.float32)
                    epoch_data = epoch.clone().detach()
                    # cut last time point for matching LaBraM expectations
                    epoch_data = utils.cut_to_size(epoch_data, window_size)

                    if target_mode == 'autoencoder':
                        self.samples.append((epoch_data, epoch_data))

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
                        self.samples.append((epoch_data, mags_data))


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

    def initialize_datasets(self, target_mode='wavelet', window_size: int = 200, wavelet_dir: Optional[Path] = None, superlet_dir: Optional[Path] = None):
        training_dataset = FinetuneDataset(self.training_epochs, target_mode=target_mode, wavelet_dir=wavelet_dir, superlet_dir=superlet_dir, window_size = window_size)
        validation_dataset = FinetuneDataset(self.validation_epochs, target_mode=target_mode, wavelet_dir=wavelet_dir, superlet_dir=superlet_dir, window_size = window_size)
        testing_dataset = FinetuneDataset(self.testing_epochs, target_mode=target_mode, wavelet_dir=wavelet_dir, superlet_dir=superlet_dir, window_size=window_size)

        return training_dataset, validation_dataset, testing_dataset


class KFoldSplitter:
    """
    Build a list of `FinetuneSplit` instances, one per fold.
    If `all_epochs` is None the constructor loads from disk using `EpochIO`
    and `settings.FINETUNE_DIR`.
    """

    def __init__(self, data_dir: Path, seed: int, all_data: Optional[Dict[str, Dict[str, list]]] = None, k_folds: int = 6):
        if all_data is None:
            io_handler = EpochIO(ext='.pt')
            # loads data in format {subject: {condition: [(id1, epoch_tensor1), (id2, epoch_tensor2), ...]}}
            all_data = io_handler.load_all_epochs(data_dir)
        self.all_data = all_data
        self.k_folds = k_folds
        self.seed = seed

    def split_by_subject(self) -> List[FinetuneSplits]:
        """
        This method can be implemented to split the dataset into K folds for cross-validation based on subjects, ensuring that all epochs from a given subject are in the same fold.
        Used for fine-tuning generalizable models.
        Returns: a list of `FinetuneDataset` instances, one per fold.
        """
        subjects = list(self.all_data.keys())
        rng = random.Random(self.seed)
        rng.shuffle(subjects)
        # Split subjects into K folds
        folds = [subjects[i::self.k_folds] for i in range(self.k_folds)]
        # Iterate over folds to create train/val/test splits for each fold
        splits = []
        for i in range(self.k_folds):
            test_subj = folds[i]
            train_val_subj = [s for s in subjects if s not in test_subj]
            rng.shuffle(train_val_subj)
            split_idx = len(train_val_subj) // 5
            val_subj = train_val_subj[:split_idx]  # 20% for validation
            train_subj = train_val_subj[split_idx:]  # 80% for training
            train = {subject: self.all_data[subject] for subject in train_subj}
            val = {subject: self.all_data[subject] for subject in val_subj}
            test = {subject: self.all_data[subject] for subject in test_subj}
            splits.append(FinetuneSplits(training_dict=train, validation_dict=val, testing_dict=test))

        return splits
