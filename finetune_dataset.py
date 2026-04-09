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
        self.subjects = list(epochs_dict.keys())
        for subj, conds in epochs_dict.items():
            for cond, epoch_list in conds.items():
                for epoch in epoch_list:
                    tensor_data = torch.tensor(epoch, dtype=torch.float32)
                    self.samples.append((tensor_data, tensor_data))
               # self.samples.extend((data, subj, cond))
    #TODO replace regression with autoencoder
    #TODO does run_general_finetuning even have any specific logic for generalizing or is it all happening here?

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
    This class can be implemented to hold the training, validation, and testing epochs as torch dataset loaders for a single fold of cross-validation. It provides a method `get_datasets()` to return these epochs as tensors for model input.
    Expects input dictionaries in the format: {subject: {condition: [epoch_tensor1, epoch_tensor2, ...]}}
    """

    def __init__(self, training_dict: Dict, validation_dict: Dict, testing_dict: Dict):
        self.training_epochs = FinetuneLoader(training_dict)
        self.validation_epochs = FinetuneLoader(validation_dict)
        self.testing_epochs = FinetuneLoader(testing_dict)

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
    Build a list of `FinetuneSplit` instances, one per fold.
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
            split_idx = len(train_val_subj) // 5  # 20% for validation
            val_subj = train_val_subj[:split_idx]  # 20% for validation
            train_subj = train_val_subj[split_idx:]  # 80% for training
            train = {subject: self.all_epochs[subject] for subject in train_subj}
            val = {subject: self.all_epochs[subject] for subject in val_subj}
            test = {subject: self.all_epochs[subject] for subject in test_subj}
            datasets.append(FinetuneSplits(training_dict=train, validation_dict=val, testing_dict=test))

        return datasets

    def split_by_epoch(self, dataset: FinetuneSplits):
        # This method can be implemented to split the dataset into K folds for cross-validation
        pass
