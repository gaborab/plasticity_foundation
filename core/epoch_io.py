# python
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch

from core import utils


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
        filename = utils.vars_to_filename(subject, condition, epoch_id, saved_type, self.ext)
        target = directory / filename
        torch.save(tensor_data, target)
        return target

    def load_specific(self, directory: Path, subject: str, cond: str, idx: str, marker: str) -> Optional[Any]:
        """
        Load all epoch files in `directory` for a specific `subject`, `cond`, `marker` and `id` matching the filename convention.
        """
        filename = utils.vars_to_filename(subject, cond, idx, marker=marker, ext=self.ext)
        # find matching file in directory
        file = next(directory.glob(filename), None)

        if not file or not file.exists():
            print(f"WARNING: File not found for filename: {filename} in directory: {directory}")

        if file and file.exists():
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
            pos, id, subject, condition = utils.vars_from_filename(file.stem, marker="_epoch")

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
