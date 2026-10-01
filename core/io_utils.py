import json
import hashlib
import h5py
from typing import Dict, Any

def vars_from_filename(stem, marker):
    """
    Extracts the numeric ID from a filename based on a specified marker. For example, if the filename is "subject_condition_epoch5.pt" and the marker is "_epoch", this function will return 5.
    """

    # Extract epoch ID
    pos = stem.rfind(marker)
    if pos == -1:
        print("WARNING: marker not found in filename:", stem)
    id = stem[pos + len(marker):]  # part after "_epoch"
    if not id.isdigit():
        print("WARNING: epoch id non numeric in filename:", stem)
    id = int(id)

    # Extract subject and condition from filename
    prefix = stem[:pos]  # part before "_epoch"
    parts = prefix.split('_', 1)
    if len(parts) < 2:
        print("WARNING: unexpected filename:", stem)
    subject, condition = parts[0], parts[1]

    return pos, id, subject, condition

def vars_to_filename(subject, condition, id, marker, ext):
    """
    Constructs a filename from the given subject, condition, ID, marker, and extension. For example, if subject is "subject1", condition is "conditionA", id is 5, marker is "_epoch", and ext is ".pt", this function will return "subject1_conditionA_epoch5.pt".
    """
    filename = f"{subject}_{condition}_{marker}{str(id)}{ext}"
    return filename

def group_to_dict(group: h5py.Group) -> Dict[str, Any]:
    return {key: value for key, value in group.items()}

def nested_group_to_dict(group: h5py.Group) -> Dict[str, Any]:
    """
    Recursively converts an h5py.Group to a flat dictionary.
    Expects no duplicate keys in the nested structure!
    """
    ret_dict = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            ret_dict[key] = nested_group_to_dict(value)
        else:
            if key in ret_dict:
                print(f"WARNING: key {key} already exists in nested group, overriding value: {ret_dict[key]}")
            ret_dict[key] = value[()]
    return ret_dict


def generate_hash(input_dict: dict, return_length: int = 10) -> str:
    """
    Generates a SHA256 hash for the given input dictionary.
    """

    return hashlib.sha256(json.dumps(input_dict).encode()).hexdigest()[:return_length]
