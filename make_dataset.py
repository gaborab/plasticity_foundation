# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------

from pathlib import Path
from typing import Optional

import argparse
import mne
import torch

from core.epoch_io import EpochIO
from libs.unscrambly.preprocessing import DataPreprocessor
from core import settings
from core import utils
from typing import Dict, Any, Tuple


def match_and_order(raw_io: mne.io.Raw):
    # drop channels not in the LUT
    useless_channels = []
    for ch in raw_io.ch_names:
        if ch not in settings.EGI_TO_1020_MAP:
            useless_channels.append(ch)
    raw_io.drop_channels(useless_channels)

    # rename channels
    raw_io.rename_channels(settings.EGI_TO_1020_MAP)
    # reorder channels
    if len(settings.CHORDER_STANDARD) == len(raw_io.ch_names):
        raw_io.reorder_channels(settings.CHORDER_STANDARD)

def rescale_to_uv(epochs: Dict[str, Dict[str, Dict[str, Any]]]):
    for subj, trial_conds in epochs.items(): # STMD1, baseline
        for trial_cond, conds in trial_conds.items(): # baseline, left_1
            for cond, epochs_data in conds.items(): # left_1, data
                print(f"Rescaling {subj} {trial_cond} {cond} to uV, original scale: {epochs_data.get_data().mean()} V")
                epochs_data.apply_function(lambda x: x * 1e6) # rescale to uV
                print(f"Rescaled {subj} {trial_cond} {cond} to uV, new scale: {epochs_data.get_data().mean()} uV")
    return epochs




def save_dataset(epochs: Dict[str, Dict[str, mne.Epochs]], path: Path, io_handler: EpochIO,
                 save_superlet: bool, save_wavelet: bool,
                 scale_to_uv: Optional[bool] = True):
    """
    Dumps all epochs found in epochs, named according to which subject and which condition(side and eccentricity) it belongs to.
    Also saves superlet and wavelet as targets for spectral prediction if specified.
    :param scale_to_uv: if set to True, scales mne content to use uV instead of V (required by LaBraM)
    :param epochs: preprocessed dictionary containing epochs of either baseline or post-patching data separated by subject and condition
    :param path: save folder path
    """
    path.mkdir(parents=True, exist_ok=True)
    for subj in epochs.keys():
        for cond in epochs[subj].keys():
            epoch_list = epochs[subj][cond].get_data()
            try:
                id = 1
                for epoch in epoch_list:
                    epoch_path = path / settings.EPOCHS_DIR_NAME
                    epoch_path.mkdir(parents=True, exist_ok=True)
                    superlet_path = path / settings.SUPERLET_DIR_NAME
                    superlet_path.mkdir(parents=True, exist_ok=True)
                    wavelet_path = path / settings.WAVELET_DIR_NAME
                    wavelet_path.mkdir(parents=True, exist_ok=True)

                    tensor_data = torch.tensor(epoch, dtype=torch.float32)
                    # save tensor to path with filename convention: {subject}_{condition}_epoch{index}.pt



                    if save_superlet:
                        # TODO compute superlet target and save to path with filename convention: {subject}_{condition}_superlet{index}.pt
                        pass
                    if save_wavelet:
                        print(f"Computing wavelet transform for epoch {id} for {subj} {cond}")
                        mag, freqs = utils.calc_wavelet(tensor_data, fs=settings.RSFREQ, **settings.SPECTRAL_DECODER_PARAMS)
                        wavelet_tensor = torch.tensor(mag, dtype=torch.float32)
                        wavelet_tensor = wavelet_tensor.permute(1, 0, 2) # reorder to (channels, freqs, time) for LaBraM compatibility
                        print(f"Original wavelet tensor shape (channels, freqs, time): {wavelet_tensor.shape}")
                        wavelet_tensor = utils.cut_to_timewindow(wavelet_tensor, settings.EXTENDED_TIME_WINDOW, settings.TIME_WINDOW, settings.RSFREQ)
                        print(f"Cut wavelet tensor shape (channels, freqs, time): {wavelet_tensor.shape}")
                        io_handler.save(tensor_data=wavelet_tensor, directory=wavelet_path, subject=subj, condition=cond,
                                        epoch_id=id, saved_type='wavelet')
                        print(f"Saved wavelet target for epoch {id} for {subj} {cond} to {wavelet_path}")

                    tensor_data = utils.cut_to_timewindow(tensor_data, settings.EXTENDED_TIME_WINDOW, settings.TIME_WINDOW, settings.RSFREQ)
                    io_handler.save(tensor_data=tensor_data, directory=epoch_path, subject=subj, condition=cond,
                                    epoch_id=id,
                                    saved_type='epoch')
                    print(f"Saved epoch {id} for {subj} {cond} to {epoch_path}")
                    id += 1

            except Exception as e:
                print(f"ERROR: Failed to dump epoch for {subj} {cond}: {e}")


def main(opts):
    print("Preparing preprocessed dataset...")
    # root to dataset

    io_handler = EpochIO(ext='.pt')

    # initialize unscrambly preprocessor, set required filter frequencies
    preprocessor = DataPreprocessor(
        directory=settings.DATA_ROOT,  # Root containing subject folders
        filter_params={'highpass': settings.L_FREQ, 'lowpass': settings.H_FREQ, 'notch': 50},
        ica_enabled=True,
        ica_n_components=15,
        bad_channels='interpolate',
        time_window=settings.EXTENDED_TIME_WINDOW,
        baseline_win=(-0.2, 0),
        split_by='both',
        re_reference='average',
    )

    # preprocess subjects, order into a dict with hierarchy: subject, condition
    preproc_data = preprocessor.process_subjects(included_conditions=settings.INCLUDED_CONDITIONS,
                                                 excluded_subjects=settings.EXCLUDED_SUBJECTS,
                                                 recordings_per_condition=settings.RECORDINGS_PER_CONDITION)
    # resample
    for subject in preproc_data.keys():
        for condition in preproc_data[subject].keys():
            preproc_data[subject][condition].resample(settings.RSFREQ)
    print(f"Resampled to {settings.RSFREQ} Hz for all data")

    # match channels to labram and reorder (for compatibility with electrode localization)
    for subject in preproc_data.keys():
        for condition in preproc_data[subject].keys():
            try:
                match_and_order(preproc_data[subject][condition])
            except Exception as e:
                print(f"ERROR: Failed to reorder channels for {subject}_{condition}: {e}")

    print("Channel reordering to LaBraM representation complete.")

    # extract epochs (MNE built-in baselining)
    all_epochs = preprocessor.create_subject_epochs(preproc_data, output_format='epochs', manual_baseline=False)
    # rescale to uV
    all_epochs = rescale_to_uv(all_epochs)  # subject - baseline/pp
    print("Created epochs for all subjects.")
    print("Dumping epochs...")
    # separate epochs
    baseline_epochs = {subj: all_epochs[subj]["Baseline"] for subj in all_epochs.keys()}
    patch_epochs = {subj: all_epochs[subj]["Post-patching"] for subj in all_epochs.keys()}

    # dump epochs data for finetuning (pre-patching)

    save_dataset(epochs=baseline_epochs, path=settings.FINETUNE_DIR, io_handler=io_handler, scale_to_uv=True,
                 save_superlet=opts.save_superlet, save_wavelet=opts.save_wavelet)
    # dump epochs data for plasticity analysis (post-patching)
    save_dataset(epochs=patch_epochs, path=settings.PATCH_DIR, io_handler=io_handler, scale_to_uv=True,
                 save_superlet=opts.save_superlet, save_wavelet=opts.save_wavelet)

    print(f"Preprocessed epochs saved to {settings.FINETUNE_EPOCHS_DIR} and {settings.PATCH_EPOCHS_DIR}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocessing pipeline', add_help=False)
    parser.add_argument('--save_superlet', type=bool, default=False, #TODO
                        help='Whether to save superlet targets for spectral prediction')
    parser.add_argument('--save_wavelet', type=bool, default=True,
                        help='Whether to save wavelet targets for spectral prediction')

    args = parser.parse_args()
    main(args)
