# python
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch

class EpochIO:
    """
    Simple utility for saving/loading epoch tensors to a directory.

    Filename convention: {subject}_{condition}_epoch{index}.pt
    Example: STMD1_left_epoch1.pt
    """

    def __init__(self, ext: Optional[str] = '.pt'):
        self.ext = ext

    def save(self, epoch_tensor: Any, directory: Path, subject: str, condition: str, epoch_id: int) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{subject}_{condition}_epoch{epoch_id}{self.ext}"
        target = directory / filename
        torch.save(epoch_tensor, target)
        return target

    def load_all(self, directory: Path) -> Dict[str, Dict[str, List[Any]]]:
        """
        Load all epoch files in `directory` matching the filename convention.
        Returns: {subject: {condition: [epoch_tensor, ...]}}
        """
        epochs = {}
        if not directory.exists():
            print("WARNING: Directory does not exist, unable to load epochs:", directory)
            return epochs

        for file in directory.glob(f"*{self.ext}"):
            stem = file.stem

            # Extract epoch ID
            marker = "_epoch"
            pos = stem.rfind(marker)
            if pos == -1:
                print("WARNING: Skipping file with unexpected name format:", file.name)
                continue
            id = stem[pos + len(marker):]  # part after "_epoch"
            if not id.isdigit():
                print("WARNING: Skipping file with non-numeric epoch ID:", file.name)
                continue
            id = int(id)

            # Extract subject and condition from filename
            prefix = stem[:pos] # part before "_epoch"
            parts = prefix.split('_', 1)
            if len(parts) < 2:
                print("WARNING: Skipping file with unexpected name format:", file.name)
                continue
            subject, condition = parts[0], parts[1]
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

        # sort each condition list by epoch_id and keep only the data in order
        for subj, conds in epochs.items():
            for cond, pairs in list(conds.items()):
                pairs.sort(key=lambda t: t[0])
                conds[cond] = [data for (_id, data) in pairs]

        return epochs

    def load_subject(self, directory: Path, subject: str) -> Optional[Dict[str, List[Any]]]:
        all_epochs = self.load_all(directory)
        return all_epochs.get(subject)
