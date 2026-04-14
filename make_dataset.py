# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------

from pathlib import Path
from typing import Optional

import mne
import torch

from core.epoch_io import EpochIO
from libs.unscrambly.preprocessing import DataPreprocessor
from core import settings


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


def dump_epoch_data(epochs: dict[str, dict[str, mne.Epochs]], path: Path, io_handler: EpochIO,
                    scale_to_uv: Optional[bool] = True):
    """
    Dumps all epochs found in epochs, named according to which subject and which condition(side and eccentricity) it belongs to.
    :param scale_to_uv: if set to True, scales mne content to use uV instead of V (required by LaBraM)
    :param epochs: preprocessed dictionary containing epochs of either baseline or post-patching data separated by subject and condition
    :param path: save folder path
    """
    for subj in epochs.keys():
        for cond in epochs[subj].keys():
            epoch_list = epochs[subj][cond].get_data()
            if scale_to_uv:
                epoch_list *= 1e6
            try:
                id = 1
                for epoch in epoch_list:
                    tensor_data = torch.tensor(epoch, dtype=torch.float32)
                    # save tensor to path with filename convention: {subject}_{condition}_epoch{index}.pt
                    io_handler.save(epoch_tensor=tensor_data, directory=path, subject=subj, condition=cond, epoch_id=id)
                    id += 1
            except:
                print(f"ERROR: Failed to dump epoch for {subj} {cond}")


if __name__ == "__main__":
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
        time_window= settings.TIME_WINDOW,
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
    print("Created epochs for all subjects.")

    print("Dumping epochs...")
    # separate epochs
    baseline_epochs = {subj: all_epochs[subj]["Baseline"] for subj in all_epochs.keys()}
    patch_epochs = {subj: all_epochs[subj]["Post-patching"] for subj in all_epochs.keys()}

    # dump epochs data (baseline for training, post-patching for plasticity analysis)
    dump_epoch_data(epochs=baseline_epochs, path=settings.FINETUNE_DIR, io_handler=io_handler, scale_to_uv=True)
    dump_epoch_data(epochs=patch_epochs, path=settings.PATCH_DIR, io_handler=io_handler, scale_to_uv=True)

    print(f"Preprocessed epochs saved to {settings.FINETUNE_DIR} and {settings.PATCH_DIR}.")
