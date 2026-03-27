# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import os
import pickle

from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import mne
import torch

from libs.unscrambly.preprocessing import DataPreprocessor


EGI_TO_1020_MAP = {
    "EEG 10": "EEG FP1-REF",
    "EEG 5": "EEG FP2-REF",
    "EEG 12": "EEG F3-REF",
    "EEG 60": "EEG F4-REF",
    "EEG 20": "EEG C3-REF",
    "EEG 50": "EEG C4-REF",
    "EEG 28": "EEG P3-REF",
    "EEG 42": "EEG P4-REF",
    "EEG 35": "EEG O1-REF",
    "EEG 39": "EEG O2-REF",
    "EEG 18": "EEG F7-REF",
    "EEG 58": "EEG F8-REF",
    "EEG 24": "EEG T3-REF",
    "EEG 52": "EEG T4-REF",
    "EEG 30": "EEG T5-REF",
    "EEG 44": "EEG T6-REF",
    "EEG 29": "EEG A1-REF",
    "EEG 47": "EEG A2-REF",
    "EEG 6": "EEG FZ-REF",
    "EEG VREF": "EEG CZ-REF",
    "EEG 34": "EEG PZ-REF",
    "EEG 17": "EEG T1-REF",
    "EEG 1": "EEG T2-REF",
}
CHORDER_STANDARD = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']

L_FREQ = 0.1
H_FREQ = 75.0
RSFREQ = 200
INCLUDED_CONDITIONS = ['baseline', 'post-patching']
EXCLUDED_SUBJECTS = ['STMD35', 'STMD37', 'STMD38', 'STMD39', 'STMD40', 'STMD41', 'STMD42', 'STMD43', 'STMD44', 'STMD45']
RECORDINGS_PER_CONDITION = None

def match_and_order(raw_io: mne.io.Raw):

    # drop channels not in the LUT
    useless_channels = []
    for ch in raw_io.ch_names:
        if ch not in EGI_TO_1020_MAP:
            useless_channels.append(ch)
    raw_io.drop_channels(useless_channels)

    # rename channels
    raw_io.rename_channels(EGI_TO_1020_MAP)
    # reorder channels
    if len(CHORDER_STANDARD) == len(raw_io.ch_names):
        raw_io.reorder_channels(CHORDER_STANDARD)


def dump_epoch_data(epochs: dict[str, dict[str, mne.Epochs]], path: Path, scale_to_uv: Optional[bool] = True):
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
                    save_path = Path(path / f"{subj}_{cond}_epoch{id}.pkl")
                    torch.save(tensor_data, save_path)
                    id += 1
            except:
                print(f"ERROR: Failed to dump epoch for {subj} {cond}")


if __name__ == "__main__":
    print("Preparing preprocessed dataset...")
    # root to dataset
    data_root = r".\data.\STMD_data"

    # initialize unscrambly preprocessor, set required filter frequencies
    preprocessor = DataPreprocessor(
        directory= data_root,  # Root containing subject folders
        filter_params={'highpass': L_FREQ, 'lowpass': H_FREQ, 'notch': 50},
        ica_enabled=True,
        ica_n_components=15,
        bad_channels='interpolate',
        time_window=(-0.1, 0.5),
        baseline_win=(-0.2, 0),
        split_by= 'both',
        re_reference='average',
    )

    # preprocess subjects, order into a dict with hierarchy: subject, condition
    preproc_data = preprocessor.process_subjects(included_conditions= INCLUDED_CONDITIONS, excluded_subjects= EXCLUDED_SUBJECTS, recordings_per_condition= RECORDINGS_PER_CONDITION)
    # resample
    for subject in preproc_data.keys():
        for condition in preproc_data[subject].keys():
            preproc_data[subject][condition].resample(RSFREQ)
    print(f"Resampled to {RSFREQ} Hz for all data")

    # match channels to labram and reorder (for compatibility with electrode localization)
    for subject in preproc_data.keys():
        for condition in preproc_data[subject].keys():
            try:
                match_and_order(preproc_data[subject][condition])
            except Exception as e:
                print(f"ERROR: Failed to reorder channels for {subject}_{condition}: {e}")

    print("Channel reordering to LaBraM representation complete.")

    # extract epochs (MNE built-in baselining)
    all_epochs = preprocessor.create_subject_epochs(preproc_data, output_format='epochs', manual_baseline= False)
    print("Created epochs for all subjects.")

    print("Dumping epochs...")
    # separate epochs
    baseline_epochs = {subj: all_epochs[subj]["Baseline"] for subj in all_epochs.keys()}
    patch_epochs = {subj: all_epochs[subj]["Post-patching"] for subj in all_epochs.keys()}

    # dump epochs data (baseline for training, post-patching for plasticity analysis)
    finetune_dir = Path('.') / "finetune_epochs"
    finetune_dir.mkdir(exist_ok=True)
    patch_dir = Path('.') / "postpatch_epochs"
    patch_dir.mkdir(exist_ok=True)


    dump_epoch_data(baseline_epochs, finetune_dir, scale_to_uv= True)
    dump_epoch_data(patch_epochs, patch_dir, scale_to_uv= True)

    print(f"Preprocessed epochs saved to {finetune_dir} and {patch_dir}.")

    # train_val_dir = os.path.join(root, "Baseline")
    # for edf_file in train_val_dir.rglob("*.edf"):
    #     print(f'processing {edf_file.name}')
    #     eegData, chOrder = preprocessing_edf(edf_file, l_freq, h_freq, rsfreq, channel_rename_lut=EGI_TO_1020_MAP,
    #                                          standard_channels=CHORDER_STANDARD)
    #     # ensure channel number compatibility
    #     chunks = (len(chOrder), rsfreq)
    #     chOrder = [s.upper() for s in chOrder]
    #     eegData = eegData[:, :-10 * rsfreq]
    #
    #
    # np.random.shuffle(train_val_a_sub)
    # train_a_sub, val_a_sub = (
    #     train_val_a_sub[: int(len(train_val_a_sub) * 0.8)],
    #     train_val_a_sub[int(len(train_val_a_sub) * 0.8) :],
    # )
    #
    # # train, val normal subjects
    # train_val_normal = os.path.join(root, "train", "normal", channel_std)
    # train_val_n_sub = list(
    #     set([item.split("_")[0] for item in os.listdir(train_val_normal)])
    # )
    # np.random.shuffle(train_val_n_sub)
    # train_n_sub, val_n_sub = (
    #     train_val_n_sub[: int(len(train_val_n_sub) * 0.8)],
    #     train_val_n_sub[int(len(train_val_n_sub) * 0.8) :],
    # )
    #
    # # test abnormal subjects
    # test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    # test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))
    #
    # # test normal subjects
    # test_normal = os.path.join(root, "eval", "normal", channel_std)
    # test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))
    #
    # # create the train, val, test sample folder
    # if not os.path.exists(os.path.join(root, "processed")):
    #     os.makedirs(os.path.join(root, "processed"))
    #
    # if not os.path.exists(os.path.join(root, "processed", "train")):
    #     os.makedirs(os.path.join(root, "processed", "train"))
    # train_dump_folder = os.path.join(root, "processed", "train")
    #
    # if not os.path.exists(os.path.join(root, "processed", "val")):
    #     os.makedirs(os.path.join(root, "processed", "val"))
    # val_dump_folder = os.path.join(root, "processed", "val")
    #
    # if not os.path.exists(os.path.join(root, "processed", "test")):
    #     os.makedirs(os.path.join(root, "processed", "test"))
    # test_dump_folder = os.path.join(root, "processed", "test")
    #
    # # fetch_folder, sub, dump_folder, labels
    # parameters = []
    # for train_sub in train_a_sub:
    #     parameters.append([train_val_abnormal, train_sub, train_dump_folder, 1])
    # for train_sub in train_n_sub:
    #     parameters.append([train_val_normal, train_sub, train_dump_folder, 0])
    # for val_sub in val_a_sub:
    #     parameters.append([train_val_abnormal, val_sub, val_dump_folder, 1])
    # for val_sub in val_n_sub:
    #     parameters.append([train_val_normal, val_sub, val_dump_folder, 0])
    # for test_sub in test_a_sub:
    #     parameters.append([test_abnormal, test_sub, test_dump_folder, 1])
    # for test_sub in test_n_sub:
    #     parameters.append([test_normal, test_sub, test_dump_folder, 0])
    #
    # # split and dump in parallel
    # with Pool(processes=24) as pool:
    #     # Use the pool.map function to apply the square function to each element in the numbers list
    #     result = pool.map(split_and_dump, parameters)