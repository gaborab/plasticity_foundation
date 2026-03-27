import numpy as np
import mne
import os
import pickle
from pathlib import Path

from .file_io import extract_edf_from_session as edf_from_dir
from .file_io import extract_pkl_from_directory as pkl_from_dir

# Function to select bad ICA components manually
def select_ica(file_path, n_components):
    '''
    Function to view recordings and select bad components to store in dictionary.
    Selected components are removed in preprocessing.py.

    Parameters:
    file_path (str): Path to the input file (EDF file).
    n_components (int): Number of components for ICA.
    
    Returns:
    selected_ica (dict): Key: filepath (str); value: number of ICA components [0] & selected indices [1] (tuple).
    '''

    # Open file
    file_path = Path(file_path)
    if file_path.suffix == '.edf':
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.notch_filter(50, fir_design='firwin', verbose=False)
    
    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter='auto', verbose=False)
    print('Fitting ICA...')
    ica.fit(raw, verbose=False)

    # Calculate variance using the mixing matrix
    mixing_matrix = ica.mixing_matrix_
    # Square and sum each column, then normalize
    component_importance = np.sum(mixing_matrix**2, axis=0)
    importance_percent = 100 * component_importance / np.sum(component_importance)

    # Info: components are automatically sorted in descending order by MNE (0: max var - n: min var)

    print(f'\n--- Component importance (based on mixing matrix) ---')
    for i in range(n_components):
        print(f'Component {i}: {importance_percent[i]:.2f}%')
    print('------------------------------------------------------')

    # Plot ICA components for visual inspection
    print(f"Displaying ICA components for '{os.path.basename(file_path.stem)}'... Close the plot when you're done reviewing.")
    ica.plot_sources(raw, block=True)

    # After plot is closed, prompt user to enter components to remove
    while True:
        try:
            print('\nAfter reviewing the components, please enter the indices of bad components to remove.')
            components_str = input('Enter bad components as comma-separated integers (e.g. "0, 1, 5"). If none found, press enter: ')

            # Parse components
            if components_str.strip():
                components_with_duplicates = [int(comp.strip()) for comp in components_str.split(',') if comp.strip()]
                # Sort and remove duplicates
                components = sorted(list(set(components_with_duplicates)))
                
                if len(components) < len(components_with_duplicates):
                    print(f'Note: Duplicate entries were removed. Using components: {components}')
            else:
                components = []

            # Validate components
            if any(comp < 0 or comp >= n_components for comp in components):
                print(f'ERROR: Component indices must be between 0 and {n_components-1}. Try again.')
                continue
            
            break

        except ValueError:
            print('ERROR: Please enter valid integers separated by commas. Try again.')
    
    # Create and return dictionary
    result_dict = {file_path : (n_components, components)}

    return result_dict

def directory_ica(directory_path, n_components, edf_paths=None):
    '''
    Function to iterate over directory and specify bad ICA components manually (using select_ica function).

    Directory structure: specified directory > recording directory > EDF file. Each folder within the specified directory is treated as a recording
    directory only if an EDF file is found in it.

    To track selected ICA components, a pickle file is created/updated in the directory of each individual recording.

    Parameters:
    directory_path (str): Path to input directory to iterate over (e.g. recordings of a subject, pooled recordings etc.).
    n_components (int): Number of components for ICA.
    edf_paths (list): A list of the edf paths in the directory (optional).
    '''

    if not directory_path:
        print('ERROR: Directory path is not defined.')
        return

    if not n_components:
        print('ERROR: Number of ICA components is not defined.')
        return
    
    # Collect all EDF paths (if not provided)
    if not edf_paths:
        edf_paths = edf_from_dir(directory_path)
    # Ensure all paths are Path objects
    edf_paths = [Path(p) for p in edf_paths]

    # Dictionary to track selected ICA components (+ no. of components used); key: edf absolute path
    all_ica = {}    

    # Collect all pickle paths
    pkl_list = pkl_from_dir(directory_path, 'ica_selection')

    # Iterate over pickle files
    for pkl_path_str in pkl_list:
        try:
            pkl_path = Path(pkl_path_str)
            with open(pkl_path, 'rb') as pkl:
                # Read PKL file to retrieve EDF file name, selected ICA components
                ica_dict = pickle.load(pkl)
                edf_filename = next(iter(ica_dict))
                ica_data = next(iter(ica_dict.values())) # Tuple: [0] - n_components, [1] - selected bad components
                # Construct absolute path to EDF (PKL path + EDF filename)
                edf_absolute_path = pkl_path.with_name(edf_filename)
                # Record existing selection to dictionary
                all_ica[edf_absolute_path] = ica_data
        except Exception as e:
            print(f'ERROR: Failed reading pickle file. \n<{e}>. \nFile: "{pkl_path_str}".')
    
    for edf in edf_paths:
        if edf not in all_ica:
            all_ica[edf] = None

    # --- INFO BLOCK ---
    print('--- ICA INFO ---')
    print(f'{len(edf_paths)} recordings (EDF files) found.')

    reviewed = {}
    not_reviewed = {}

    # Iterate over dictionary & determine if recording has entry
    for key, value in all_ica.items():
        # Block 1: (new) recordings without existing entry
        if value == None:
            not_reviewed[key] = value
        # Block 2: recordings with existing entry
        else:
            reviewed[key] = value
    
    # Index to refer to particular recordings
    index = 0
    index_dict = {}
    
    num_reviewed = len(reviewed.keys())
    num_new = len(not_reviewed.keys())
    
    # Block 1: new (no entries found)
    if num_new > 0:
        print('-- New files --')
        for key, value in not_reviewed.items():
            print(f'[{index}] File: "{key.stem}"; not reviewed.')
            index_dict[index] = key
            index += 1

    # Block 2: reviewed (entries found)
    if num_reviewed > 0:
        print('-- Already reviewed files --')
        for key, value in reviewed.items():
            print(f'[{index}] File: "{key.stem}"; {len(value[1])} components selected from {value[0]} total. Components: {value[1]}.')
            index_dict[index] = key
            index += 1
    print()
    
    # Check if all files have same n_components as what function is called with -> if yes, accept no user input
    overwrite_mode = False
    reviewed_entries = [entry for entry in all_ica.keys() if all_ica[entry] is not None]
    if reviewed_entries and not all(all_ica[entry][0] == n_components for entry in reviewed_entries):
        overwrite_mode = True

    print('To quit without making any changes, enter Q.\n')
    if num_reviewed == 0:
        print('No ICA data found for any recordings. Press Enter to review all.\n')
    elif num_new == 0:
        print('ICA data found for all recordings. Press Enter to keep existing selection or enter indices of files to review (e.g. 1,2 or 3) or enter O to review all.\n')
    else:
        print('To only review new recordings, press Enter.\n'
              'To enter Overwrite-mode (review and overwrite all recordings), enter O.\n'
              'To review only select recordings, enter corresponding indices (e.g. 1,2 or 3).')

    # --- USER INPUT HANDLING (by modifying edf_paths list) ---
    if overwrite_mode:
        print('WARNING: Inconsistent number of ICA components detected across previous entries. All files will be reviewed due to inconsistent component numbers.')
    else:
        while True:
            user_input = input('Your choice: ').strip()
            max_index = index - 1

            # Quit
            if user_input.lower() == 'q':
                print('Quitting without changes.')
                return all_ica

            # Overwrite mode
            if user_input.lower() == 'o':
                print('Overwrite mode selected. All files will be reviewed.')
                break

            # Enter (no input)
            if not user_input:
                if num_new > 0:
                    print('Reviewing new files.')
                    edf_paths = list(not_reviewed.keys())
                else:
                    print('Keeping existing selection.')
                    return all_ica
                break

            # Indices
            try:
                selected_indices = [int(i.strip()) for i in user_input.split(',') if i.strip()]

                if not selected_indices:
                    print('ERROR: No indices provided. Please try again.') # E.g. just a comma
                    continue

                # Validate indices
                invalid_indices = [i for i in selected_indices if i < 0 or i > max_index]
                if invalid_indices:
                    print(f'ERROR: Invalid indices found: {invalid_indices}. Please only use indices between 0 and {max_index}.')
                    continue

                # Valid indices provided
                edf_paths = [index_dict[i] for i in sorted(list(set(selected_indices)))]
                print(f'Reviewing selected files: {[p.stem for p in edf_paths]}')
                break

            except ValueError:
                print(f'ERROR: Invalid input. Please enter "Q" "O", press Enter, or provide comma-separated numbers (e.g., 1,2,3).')

    # --- END OF INFO BLOCK ---

    # Iterate over edf_paths
    reviewed_files = 0
    for edf in edf_paths:
        selected_components = select_ica(edf, n_components)
        all_ica.update(selected_components)
        reviewed_files += 1
        edf_path_obj = Path(edf)
        print(f'Selection for "{edf_path_obj.stem}" complete. {len(selected_components[edf_path_obj][1])} bad components chosen.')
    
    # Iterate over dictionary
    if reviewed_files != 0:
        for key, value in all_ica.items():
            if value is None:
                continue

            # The dictionary to be saved in the pickle file
            data_to_save = {key.name: value}

            pkl_filename = 'ica_selection.pkl'
            pkl_path = key.parent / pkl_filename
            with open(pkl_path, 'wb') as f:
                pickle.dump(data_to_save, f)
        print(f'ICA components for {reviewed_files} recordings selected and saved.')
    else:
        print('No new selections made, no changes to save.')

    return all_ica