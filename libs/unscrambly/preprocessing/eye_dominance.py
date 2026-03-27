import pandas as pd
import warnings

def load_eye_dominance(csv_path: str, subject_col: str = 'subject_id', 
                       dominance_col: str = 'eye_dominance') -> dict:
    """
    Load eye dominance information from a CSV file.
    
    Returns:
        Dictionary mapping subject_id -> eye dominance ('left' or 'right')
    """    
    df = pd.read_csv(csv_path)
    
    if subject_col not in df.columns:
        raise ValueError(f"Column '{subject_col}' not found in CSV. Available columns: {df.columns.tolist()}")
    if dominance_col not in df.columns:
        raise ValueError(f"Column '{dominance_col}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Normalize eye dominance values to lowercase
    df[dominance_col] = df[dominance_col].str.lower().str.strip()
    
    # Validate values
    valid_values = {'left', 'right'}
    invalid = df[~df[dominance_col].isin(valid_values)][dominance_col].unique()
    if len(invalid) > 0:
        warnings.warn(f"Invalid eye dominance values found: {invalid}. Expected 'left' or 'right'.")
    
    dominance_dict = dict(zip(df[subject_col].astype(str), df[dominance_col]))
    
    print(f"Loaded eye dominance for {len(dominance_dict)} subjects:")
    left_count = sum(1 for v in dominance_dict.values() if v == 'left')
    right_count = sum(1 for v in dominance_dict.values() if v == 'right')
    print(f"  Left-eye dominant: {left_count}")
    print(f"  Right-eye dominant: {right_count}")
    
    return dominance_dict


def swap_left_right_for_subject(epochs_dict: dict) -> dict:
    """
    Swap 'left' and 'right' arrays in an epochs dictionary.
    Purpose: for left ocular dominance subjects, switch mismatch sides for accurate comparison.
    
    Handles both plain keys ('left', 'right') from split_by='sides'
    and suffixed keys ('left_1', 'right_1', ...) from split_by='both'.
    """
    swapped_dict = epochs_dict.copy()
    swapped = False

    # Exact keys (split_by='sides')
    if 'left' in epochs_dict and 'right' in epochs_dict:
        swapped_dict['left'], swapped_dict['right'] = epochs_dict['right'], epochs_dict['left']
        swapped = True

    # Suffixed keys (split_by='both'), e.g. left_1 <-> right_1
    left_keys = {k for k in epochs_dict if k.startswith('left_')}
    right_keys = {k for k in epochs_dict if k.startswith('right_')}
    for lk in left_keys:
        suffix = lk[len('left'):]          # e.g. '_1'
        rk = 'right' + suffix              # e.g. 'right_1'
        if rk in right_keys:
            swapped_dict[lk], swapped_dict[rk] = epochs_dict[rk], epochs_dict[lk]
            swapped = True

    if not swapped:
        return epochs_dict
    return swapped_dict


def apply_eye_dominance_remapping(subject_epochs: dict, eye_dominance: dict) -> dict:
    """
    Apply eye dominance remapping to all subjects' epochs.
    For left-eye dominant subjects, swap left/right triggers.
    
    Parameters:
        subject_epochs: Nested dict {subject: {condition: {trigger: array}}}
        eye_dominance: Dict mapping subject_id -> 'left' or 'right'
    
    Returns:
        Remapped subject_epochs dictionary
    """
    remapped = {}
    
    for subject_name, conditions in subject_epochs.items():
        # Try to find subject in eye dominance dict (handle case variations)
        dominance = None
        for key in eye_dominance:
            if key.lower() == subject_name.lower():
                dominance = eye_dominance[key]
                break
        
        if dominance is None:
            print(f"  WARNING: No eye dominance info for {subject_name}, keeping original left/right")
            remapped[subject_name] = conditions
            continue
        
        if dominance == 'left':
            # Left-eye dominant: swap left/right
            print(f"  {subject_name}: Left-eye dominant - swapping left/right")
            remapped[subject_name] = {}
            for condition_name, triggers in conditions.items():
                remapped[subject_name][condition_name] = swap_left_right_for_subject(triggers)
        else:
            # Right-eye dominant: keep as is
            remapped[subject_name] = conditions
    
    return remapped