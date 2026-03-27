import numpy as np
import mne
import os
import pickle
from pathlib import Path

from .file_io import extract_edf_from_session as edf_from_dir
from .file_io import extract_pkl_from_directory as pkl_from_dir

# Function to select bad channels manually (based on visual inspection and statistical measures)
def select_bad_channels(file_path, filter_params={'notch': 50, 'lowpass': 40, 'highpass': 0.5}):
    """
    Function to detect and visually inspect potentially bad channels.
    Selected channels are excluded/interpolated in preprocessing.py.

    Parameters:
    file_path (str): Path to the input file (EDF file).
    
    Returns:
    selected_bad_channels (dict): Key: filepath (str); value: indices of bad channels based on user selection.
    """

    # Open file
    file_path = Path(file_path)
    if file_path.suffix == '.edf':
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.notch_filter(50, fir_design='firwin', verbose=False)

    # Get channel names and data
    ch_names = raw.ch_names
    data = raw.get_data()

    # Dictionary to store detection results
    detection_results = {}

    # --- Statistical measures for bad channel detection ---
    # 1. Variance-based detection
    ch_vars = np.var(data, axis=1)
    mean_var = np.mean(ch_vars)
    std_var = np.std(ch_vars)
    high_var_threshold = mean_var + 2 * std_var
    low_var_threshold = mean_var - 2 * std_var

    # Find channels with unusually high or low variance
    high_var_channels = [ch_names[i] for i in range(len(ch_names)) 
                         if ch_vars[i] > high_var_threshold]
    low_var_channels = [ch_names[i] for i in range(len(ch_names)) 
                        if ch_vars[i] < low_var_threshold and ch_vars[i] > 0]
    
    for ch in high_var_channels:
        idx = ch_names.index(ch)
        detection_results[ch] = {'measure': 'high variance', 'value': f'{ch_vars[idx]:.8f}', 
                               'threshold': f'> {high_var_threshold:.8f}'}
    
    for ch in low_var_channels:
        idx = ch_names.index(ch)
        detection_results[ch] = {'measure': 'low variance', 'value': f'{ch_vars[idx]:.8f}', 
                               'threshold': f'< {low_var_threshold:.8f}'}
            
    # 2. Amplitude-based detection: identifies channels with extreme amplitudes
    amplitude_threshold = np.percentile(np.abs(data), 99.9) * 1.5
    for i, ch in enumerate(ch_names):
        max_amp = np.max(np.abs(data[i]))
        if max_amp > amplitude_threshold:
            detection_results[ch] = {'measure': 'high amplitude', 'value': f'{max_amp:.8f}', 
                                   'threshold': f'> {amplitude_threshold:.8f}'}
    
    # Print results
    print(f'\n--- Suggested bad channels for {os.path.basename(file_path)} ---')
    if not detection_results:
        print("No bad channels detected by statistical measures.")
    else:
        for i, (ch, details) in enumerate(detection_results.items()):
            print(f"{i+1}. {ch}: {details['measure']} ({details['value']}, threshold {details['threshold']})")
    print('------------------------------------------------------')

    # --- Visual inspection: time series for all channels ---
    print(f"Displaying raw data for '{os.path.basename(file_path)}'. Close the plot when you're done reviewing.")
    raw.plot(block=True, scalings='auto')

    # After plots are closed, prompt user to enter channels to mark as bad
    print('\nAfter reviewing the data, please enter the names of bad channels to mark.')
    print('Suggested bad channels:')
    for ch in detection_results.keys():
        print(f"- {ch}")
    while True:
        try:
            channels_str = input('\nEnter bad channel indices (only number part) separated by commas (e.g. "23, 1"). If none, press enter: ')
            
            # Parse channels
            if channels_str.strip():
                # Split by comma and strip whitespace
                num_inputs = [ch.strip() for ch in channels_str.split(',') if ch.strip()]
                
                # Validate numeric inputs
                non_numeric = [num for num in num_inputs if not num.isdigit()]
                if non_numeric:
                    print(f"ERROR: These inputs are not numeric: {non_numeric}. Please enter only channel numbers.")
                    continue
                    
                # Convert to integers
                channel_nums = [int(num) for num in num_inputs]
                
                # Convert to proper channel format and remove duplicates
                formatted_channels = [f"EEG {num}" for num in channel_nums]
                channels = list(dict.fromkeys(formatted_channels))
                
                if len(channels) < len(formatted_channels):
                    print(f'Note: Duplicate entries were removed. Using channels: {channels}')
                    
                # Validate channels exist in the data
                invalid_channels = [ch for ch in channels if ch not in ch_names]
                if invalid_channels:
                    print(f'ERROR: These channels are not in the data: {invalid_channels}.')
                    print(f'Valid channel numbers are 1-{len(ch_names)}.')
                    continue
            else:
                channels = []
            
            break
            
        except ValueError:
            print('ERROR: Please enter valid channel names separated by commas. Try again.')
    
    # Create and return dictionary
    result_dict = {file_path: channels}
    return result_dict
    
def directory_bad_channels(directory_path, edf_paths=None):
    """
    Function to iterate over directory and detect bad channels (using select_bad_channels function).

    Directory structure: specified directory > recording directory > EDF file. Each folder within the specified directory is treated as a recording
    directory only if an EDF file is found in it.

    To track selected bad channels, a pickle file is created/updated in the directory of each individual recording.

    Parameters:
    directory_path (str): Path to input directory to iterate over (e.g. recordings of a subject, pooled recordings etc.).
    edf_paths (list): A list of the edf paths in the directory (optional).
    """

    if not directory_path:
        print('ERROR: Directory path is not defined.')
        return
    
    # Collect all EDF paths (if not provided)
    if not edf_paths:
        edf_paths = edf_from_dir(directory_path)
    # Ensure all paths are Path objects
    edf_paths = [Path(p) for p in edf_paths]

    # Dictionary to track selected bad channels; key: edf absolute path
    bad_channels_dict = {}    

    # Collect all pickle paths
    pkl_list = pkl_from_dir(directory_path, 'bad_ch_selection')

    # Iterate over pickle files
    for pkl_path_str in pkl_list:
        try:
            pkl_path = Path(pkl_path_str)
            with open(pkl_path, 'rb') as pkl:
                # Read PKL file to retrieve EDF file name, selected bad channels
                bad_ch_dict = pickle.load(pkl)
                edf_filename = next(iter(bad_ch_dict))
                bad_channel_data = next(iter(bad_ch_dict.values()))
                # Construct absolute path to EDF (PKL path + EDF filename)
                edf_absolute_path = pkl_path.with_name(edf_filename)
                # Record existing selection to dictionary
                bad_channels_dict[edf_absolute_path] = bad_channel_data
        except Exception as e:
            print(f'ERROR: Failed reading pickle file. \n<{e}>. \nFile: "{pkl_path_str}".')
    
    for edf in edf_paths:
        if edf not in bad_channels_dict:
            bad_channels_dict[edf] = None

    # --- INFO BLOCK ---
    print('--- BAD CHANNEL INFO ---')
    print(f'{len(edf_paths)} recordings (EDF files) found.')

    reviewed = {}
    not_reviewed = {}

    # Iterate over dictionary & determine if recording has entry
    for key, value in bad_channels_dict.items():
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
            print(f'[{index}] File: "{key.stem}"; {len(value)} bad channels selected. Channels: {value}.')
            index_dict[index] = key
            index += 1
    print()
    
    print('To quit without making any changes, enter Q.\n')
    if num_reviewed == 0:
        print('No bad channel data found for any recordings. Press Enter to review all.\n')
    elif num_new == 0:
        print('Bad channel data found for all recordings. Enter indices of files to review (e.g. 1,2 or 3) or press Enter to review all.\n')
    else:
        print('To only review new recordings, press Enter.\n'
              'To enter Overwrite-mode (review and overwrite all recordings), enter O.\n'
              'To review only select recordings, enter corresponding indices (e.g. 1,2 or 3).')

    # --- USER INPUT HANDLING (by modifying edf_paths list) ---
    while True:
        user_input = input('Your choice: ').strip()
        max_index = index - 1

        # Quit
        if user_input.lower() == 'q':
            print('Quitting without changes.')
            return

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
                print('Reviewing all files.')
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
        selected_channels = select_bad_channels(edf)
        bad_channels_dict.update(selected_channels)
        reviewed_files += 1
        edf_path_obj = Path(edf)
        print(f'Selection for "{edf_path_obj.stem}" complete. {len(selected_channels[edf_path_obj])} bad channels chosen.')
    
    # Iterate over dictionary
    if reviewed_files != 0:
        for key, value in bad_channels_dict.items():
            if value is None:
                continue

            # The dictionary to be saved in the pickle file
            data_to_save = {key.name: value}

            pkl_filename = 'bad_ch_selection.pkl'
            pkl_path = key.parent / pkl_filename
            with open(pkl_path, 'wb') as f:
                pickle.dump(data_to_save, f)
        print(f'Bad channels for {reviewed_files} recordings selected and saved.')
    else:
        print('No new selections made, no changes to save.')
    
    return bad_channels_dict

if __name__ == '__main__':
    directory_bad_channels(r"C:\Users\fekbe\Documents\TTK\Tunnel\DATA\NEW_protocol\AKI_Tabor\David\07-25\Baseline")