# Extract epochs from EDF
'''
Evoked vs. Epochs object in MNE:

mne.Epochs: retains data for individual trials
mne.Evoked: created by averaging Epochs objects; does not retain individual trials, only the average
'''
import mne
import numpy as np

def edf_to_epochs(edf_file, filter_params={'notch': 50, 'lowpass': 70, 'highpass': 1},
                  channels=['EEG 35', 'EEG 37', 'EEG 39'],
                  triggers={'left': 'DIN21', 'top': 'DIN31', 'right': 'DIN41', 'global': 'DIN51'},
                  time_window=(-0.1, 1), reject_amplitude_uv=100, baseline_win=None):
    '''
    Perform preprocessing steps.
    Extract epochs from EDF file for local and global mismatch triggers.

    Parameters:
    edf_file (str): The path to the EDF file to read.
    filter_params (dict): Dictionary of filter parameters (notch, highpass, lowpass).
    channels (list): The list of channels to include.
    triggers (dict): Dictionary of mismatch triggers to create epochs based on (key: mismatch type; value: trigger value (or values in a list)). 
        Any trigger type can have a list of trigger values assigned to it -> all corresponding events will be epoched together. E.g. 'local': ['DIN21', 'DIN31', 'DIN41'].
    time_window (tuple): The time window to analyze (start, end).
    reject_amplitude_uv (int): Peak-to-peak amplitude cut-off (in µV).
    baseline_win (tuple): Tuple of beginning [0] and end [1] of baseline window in seconds. If set to None, no baselining is applied.

    Returns:
    unaveraged_epochs (dict): dictionary of concatenated (trial-by-trial) mismatch epochs (key: mismatch type; value mne.Epochs object)
    averaged_epochs (dict): dictionary of averaged, concatenated mismatch epochs (key: mismatch type; value: averaged mne.Evoked object)
    '''
    
    # Check baseline parameters availability
    if baseline_win is None:
        baseline = None
    elif baseline_win[0] is None or baseline_win[1] is None:
        print('WARNING: Incomplete baseline window parameters. Baselining skipped.')
        baseline = None
    elif baseline_win[0] > baseline_win[1]:
        print('WARNING: Invalid baseline window: start time > end time. Baselining skipped.')
        baseline = None
    else:
        baseline = baseline_win
    
    # Helper function for epoch creation
    def create_mismatch_epoch(raw, events, event_id, time_window, picks, reject, baseline):
        return mne.Epochs(raw, events, event_id, tmin=time_window[0], tmax=time_window[1], 
                          picks=picks, baseline=baseline, preload=True, 
                          event_repeated='merge', reject=reject, verbose=False)
    
    unaveraged_epochs = {}   # Dictionary: store mismatch epochs in concatenated Epochs objects
    averaged_epochs = {}     # Dictionary: store mismatch epochs in concatenated & averaged Evoked objects

    # Load EDF file
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    except Exception as e:
        print(f'ERROR: Failed to load EDF file {edf_file}: {e}')
        return {}, {}

    # Apply filtering to Raw object
    if filter_params:
        raw.filter(filter_params['highpass'], filter_params['lowpass'], fir_design='firwin', verbose=False)
        raw.notch_filter(filter_params['notch'], fir_design='firwin', verbose=False)

    # Extract events from annotations and pick channels
    events, event_ids = mne.events_from_annotations(raw, verbose=False)
    picks = mne.pick_channels(raw.ch_names, include=channels, verbose=False)

    # Populate lists of epochs
    try:
        # Set rejection threshold (max peak-to-peak amplitude)
        reject = {'eeg': reject_amplitude_uv * 1e-6} if reject_amplitude_uv else None

        # Create epochs based on mismatch triggers
        for mismatch_type, trigger_values in triggers.items():
            # Convert single trigger to list            
            if not isinstance(trigger_values, list):
                trigger_values = [trigger_values]
            
            # List to store epochs for each trigger value
            type_epochs = []

            for trigger_value in trigger_values:
                if trigger_value in event_ids:
                    mismatch_epoch = create_mismatch_epoch(raw, events, event_ids[trigger_value], 
                                                           time_window, picks, reject, baseline)
                    type_epochs.append(mismatch_epoch)
                else:
                    print(f'WARNING: No events found for {trigger_value} ({mismatch_type} mismatch start).')
            
            # Combine epochs from multiple triggers
            if type_epochs:
                if len(type_epochs) == 1:
                    unaveraged_epochs[mismatch_type] = type_epochs[0]
                else:
                    unaveraged_epochs[mismatch_type] = mne.concatenate_epochs(type_epochs, verbose=False)
        
        # Create the averaged epochs from the unaveraged ones -> mne.Evoked object
        for mismatch_type, epochs in unaveraged_epochs.items():
            averaged_epochs[mismatch_type] = epochs.average()

    except Exception as e:
        print('ERROR: Epoch extraction failed for raw object.')
        return {}, {}
    
    # debug: print number of epochs
    print('--- Number of Epochs per trigger type ---')
    for key, epochs in unaveraged_epochs.items():
        print(f'Type: {key} - number of epochs: {len(epochs)}')

    return unaveraged_epochs, averaged_epochs

def raw_to_epochs(raw, channels=None, time_window=None, time_window_with_baseline=None,
                  triggers={'left': 'DIN21', 'top': 'DIN31', 'right': 'DIN41', 'global': 'DIN51'},
                  reject_amplitude_uv=100, baseline_win=None, use_trigger_chan=False):
    '''
    Used in preprocessing.py.
    Extract epochs from mne.io.Raw object for local and global mismatch triggers.

    Parameters:
    raw (mne.io.Raw): Raw object to extract epochs from.
    channels (list): The list of channels to include.
    time_window (tuple): The time window to analyze (start, end).
    time_window_with_baseline (tuple): The time window to analyze (start, end) extended by the baseline window.
    triggers (dict): Dictionary of mismatch triggers to create epochs based on (key: mismatch type; value: trigger value (or values in a list)). 
        Any trigger type can have a list of trigger values assigned to it -> all corresponding events will be epoched together. E.g. 'local': ['DIN21', 'DIN31', 'DIN41'].
    reject_amplitude_uv (int): Peak-to-peak amplitude cut-off (in µV).
    baseline_win (tuple): Tuple of beginning [0] and end [1] of baseline window in seconds. If set to None, no baselining is applied.
    use_trigger_chan (bool): Whether to find triggers in a separate trigger channel ('STI 014') (True) or from annotations (False)

    Returns:
    unaveraged_epochs (dict): dictionary of concatenated (trial-by-trial) mismatch epochs (key: mismatch type; value mne.Epochs object)
    averaged_epochs (dict): dictionary of averaged, concatenated mismatch epochs (key: mismatch type; value: averaged mne.Evoked object)
    '''
    
    if not isinstance(raw, mne.io.BaseRaw):
        print(f'ERROR: Input must be an mne.io.Raw object, but got {type(raw)} instead.')
        return {}, {}
    
    # determine which time window to use for initial epoch creation
    creation_window = time_window_with_baseline if (baseline_win and time_window_with_baseline) else time_window

    if creation_window is None:
        print('ERROR: No time window specified for epoch creation.')
        return {}, {}

    # Check baseline parameters availability
    if baseline_win is None:
        baseline = None
    elif baseline_win[0] is None or baseline_win[1] is None:
        print('WARNING: Incomplete baseline window parameters. Baselining skipped.')
        baseline = None
    elif baseline_win[0] > baseline_win[1]:
        print('WARNING: Invalid baseline window: start time > end time. Baselining skipped.')
        baseline = None
    else:
        baseline = baseline_win
        print(f'Baseline window: {baseline}')
    
    unaveraged_epochs = {}   # Dictionary: store mismatch epochs in concatenated Epochs objects
    averaged_epochs = {}     # Dictionary: store mismatch epochs in concatenated & averaged Evoked objects

    picks = mne.pick_channels(raw.ch_names, include=channels, verbose=False)

    # Get events - either from stim channel or annotations
    if use_trigger_chan:
        # - STIM CHANNEL
        try:
            events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
            unique_codes = np.unique(events[:, 2])
            event_ids = {f'DIN{code}': code for code in unique_codes}
            print(f"Found {len(events)} events from stim channel")
        except Exception as e:
            print(f"ERROR: Failed to extract events from stim channel: {e}")
            return {}, {}
    else:
        # - ANNOTATIONS
        try:
            events, event_ids = mne.events_from_annotations(raw, verbose=False)
        except Exception as e:
            print(f"ERROR: Failed to extract events from annotations: {e}")
            return {}, {}

    # Set rejection threshold
    reject = {'eeg': reject_amplitude_uv * 1e6} if reject_amplitude_uv else None

    # Create epochs based on mismatch triggers
    for mismatch_type, trigger_values in triggers.items():
        # Convert single trigger to list            
        if not isinstance(trigger_values, list):
            trigger_values = [trigger_values]
        
        # List to store epochs for each trigger value
        type_epochs = []

        for trigger_value in trigger_values:
            if trigger_value in event_ids:
                # create epochs with the extended window that includes baseline period
                epochs = mne.Epochs(raw, events, event_ids[trigger_value], 
                                    tmin=creation_window[0], tmax=creation_window[1], 
                                    picks=picks, baseline=baseline, preload=True, 
                                    event_repeated='merge', reject=reject, verbose=False)
                type_epochs.append(epochs)
            else:
                print(f'WARNING: No events found for {trigger_value} ({mismatch_type} mismatch start).')
        
        # Combine epochs from multiple triggers
        if type_epochs:
            if len(type_epochs) == 1:
                unaveraged_epochs[mismatch_type] = type_epochs[0]
            else:
                unaveraged_epochs[mismatch_type] = mne.concatenate_epochs(type_epochs, verbose=False)
        
        # Create the averaged epochs from the unaveraged ones -> mne.Evoked object
        for mismatch_type, epochs in unaveraged_epochs.items():
            averaged_epochs[mismatch_type] = epochs.average()

    # debug: print number of epochs
    print('--- Number of Epochs per trigger type ---')
    for key, epochs in unaveraged_epochs.items():
        print(f'Type: {key} - number of epochs: {len(epochs)}')
    
    return unaveraged_epochs, averaged_epochs

def local_average_epoch(epoch_dict):
    '''
    Average local mismatch epochs across mismatch types.

    Parameters:
    epoch_dict (dict): dictionary of averaged mismatch epochs (key: mismatch type; value: mne.Evoked object)

    Returns:
    local_average (mne.Evoked): the average of local mismatch epochs
    '''

    local_evoked_list = [evoked for key, evoked in epoch_dict.items() if key != 'global']
    local_average_evoked = mne.combine_evoked(local_evoked_list, weights='equal')

    return {'local_average': local_average_evoked}

def average_epoch_dicts(epoch_dicts):
    '''
    Average a list of epoch dictionaries into a single one (using rolling average).

    Parameters:
    epoch_dicts (list): list of dictionaries of mismatch epochs for a given recording (key: mismatch type; value: mne.Evoked object)

    Returns:
    averaged_epochs (dict): dictionary of averaged mismatch epochs (key: mismatch type; value: mne.Evoked object)
    '''

    if not epoch_dicts:
        return {}

    averaged_epochs = {}

    for mismatch_type in epoch_dicts[0].keys():
        evoked_list = [d[mismatch_type] for d in epoch_dicts if mismatch_type in d]
        averaged_epochs[mismatch_type] = mne.combine_evoked(evoked_list, weights='equal')

    return averaged_epochs
