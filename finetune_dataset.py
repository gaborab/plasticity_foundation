"""Class for handling the dataset for fine-tuning.
This class is designed to work with the epochs generated from the preprocessing step and saved to disk.
It provides a structure for loading and accessing training, validating and testing epochs for fine-tuning the model head.
"""
from pathlib import Path
import random
import settings
from epoch_io import EpochIO
from typing import Dict, List, Any, Optional
import torch

class FinetuneLoader(torch.utils.data.Dataset):
    """
    This class can be implemented to wrap a list of epoch tensors into a PyTorch Dataset for use in DataLoaders during fine-tuning.
    """
    def __init__(self, epochs_dict: Dict[str, Dict[str, List[Any]]]):
        self.samples = []
        for subj, conds in epochs_dict.items():
            for cond, data in conds.items():
                self.samples.extend((data, subj, cond))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.epochs_dict[idx]

class FinetuneSplits:
    """
    This class can be implemented to hold the training, validation, and testing epochs for a single fold of cross-validation. It provides a method `get_datasets()` to return these epochs as tensors for model input.
    """
    def __init__(self, training_epochs: FinetuneLoader, validation_epochs: FinetuneLoader, testing_epochs=FinetuneLoader):
        self.training_epochs = training_epochs
        self.validation_epochs = validation_epochs
        self.testing_epochs = testing_epochs

    def get_datasets(self) -> Dict[str, FinetuneLoader]:
        """This method can be implemented to return the training, validation, and testing epochs as torch datasets for model input.
        Keys: "train", "val", "test"
        """

        return {
            "train": self.training_epochs,
            "val": self.validation_epochs,
            "test": self.testing_epochs
        }


class KFoldSplitter:
    """
    Build a list of `FinetuneDataset` instances, one per fold.
    If `all_epochs` is None the constructor loads from disk using `EpochIO`
    and `settings.FINETUNE_DIR`.
    """

    def __init__(self, all_epochs: Optional[Dict[str, Dict[str, list]]] = None, k_folds: int = 6, seed: int = settings.SEED):
        if all_epochs is None:
            io_handler = EpochIO(ext='.pt')
            all_epochs = io_handler.load_all(settings.FINETUNE_DIR)
        self.all_epochs = all_epochs
        self.k_folds = k_folds
        self.seed = seed

    def split_by_subject(self) -> List[FinetuneSplits]:
        """
        This method can be implemented to split the dataset into K folds for cross-validation based on subjects, ensuring that all epochs from a given subject are in the same fold.
        Used for fine-tuning generalizable models.
        Returns: a list of `FinetuneDataset` instances, one per fold.
        """
        subjects = list(self.all_epochs.keys())
        rng = random.Random(self.seed)
        rng.shuffle(subjects)
        # Split subjects into K folds
        folds = [subjects[i::self.k_folds] for i in range(self.k_folds)]
        # Iterate over folds to create train/val/test splits for each fold
        datasets = []
        for i in range(self.k_folds):
            test_subj = folds[i]
            train_val_subj = [s for s in subjects if s not in test_subj]
            rng.shuffle(train_val_subj)
            val_subj = train_val_subj[:len(train_val_subj) // 5]  # 20% for validation
            train_subj = train_val_subj[len(train_val_subj) // 5:]  # 80% for training
            train = {subject: self.all_epochs[subject] for subject in train_subj}
            val = {subject: self.all_epochs[subject] for subject in val_subj}
            test = {subject: self.all_epochs[subject] for subject in test_subj}
            datasets.append(FinetuneSplits(training_epochs=train, validation_epochs=val, testing_epochs=test))

        return datasets

    def split_by_epoch(self, dataset: FinetuneSplits):
        # This method can be implemented to split the dataset into K folds for cross-validation
        pass
