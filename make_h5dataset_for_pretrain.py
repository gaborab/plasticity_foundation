"""
This script prepares a hdf5 file for pretraining.

Applies the following preprocessing steps to match LaBraM's internal representation:
    - low pass filter: 75 Hz
    - high pass filter: 0.1 Hz
    - resampling to 200 Hz
    - removes channels which can't be mapped to the 10-20 system
    - orders channels

It reads all .edf files in rawDataPath and saves to savePath.
"""

from pathlib import Path
from shock.utils import h5Dataset
from shock.utils import preprocessing_cnt
from shock.utils import  preprocessing_edf

savePath = Path(r'.\pretrain')
rawDataPath = Path(r'.\data\STMD_data\STMD1\Baseline')
group = rawDataPath.rglob('*.edf')

# preprocessing parameters
l_freq = 0.1
h_freq = 75.0
rsfreq = 200

egi_to_1020_map = {
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
}
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

# channel number * rsfreq
chunks = (22, rsfreq)

# ensure specific dataset name
name = "dataset"
sp_name = name
i = 1
print(Path(rf"{savePath}\{sp_name}"))
while Path(rf"{savePath}\{sp_name}.hdf5").exists():
    sp_name = name + str(i)
    i+=1

dataset = h5Dataset(savePath, sp_name)
for edf_file in group:
    print(f'processing {edf_file.name}')
    eegData, chOrder = preprocessing_edf(edf_file, l_freq, h_freq, rsfreq, channel_rename_lut = egi_to_1020_map, standard_channels = chOrder_standard)
    # ensure channel number compatibility
    chunks = (len(chOrder), rsfreq)
    chOrder = [s.upper() for s in chOrder]
    eegData = eegData[:, :-10*rsfreq]
    grp = dataset.addGroup(grpName=edf_file.stem)
    dset = dataset.addDataset(grp, 'eeg', eegData, chunks)

    # dataset attributes
    dataset.addAttributes(dset, 'lFreq', l_freq)
    dataset.addAttributes(dset, 'hFreq', h_freq)
    dataset.addAttributes(dset, 'rsFreq', rsfreq)
    dataset.addAttributes(dset, 'chOrder', chOrder)

dataset.save()
