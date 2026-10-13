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
import pandas as pd
import torch

import numpy as np

from core.io import EpochIO
from libs.unscrambly.preprocessing import DataPreprocessor
from core.io_utils import load_trigger_codes
from core import settings, io, io_utils
from core import utils
from typing import Dict, Any, Tuple, List
import argparse
from pathlib import PureWindowsPath


def load_autoreject(qc_dir: Path):
    ar = pd.read_csv(qc_dir / "autoreject_report.csv")
    ar = ar.rename(columns={"subject_name": "subject", "conditions": "session"})
    ar["dropped_share"] = ar["n_dropped"] / ar["n_epochs"]
    return ar[["subject", "session", "n_epochs", "n_dropped", "dropped_share", "mean_interp_per_kept_epoch"]]


def load_auto_bad_channels(qc_dir: Path):
    bc = pd.read_csv(qc_dir / "auto_bad_chans.csv")
    # data\raw\STMD_data\<subject>\<session>\<recording folder>\<file>.edf
    parts = bc["recording"].map(lambda p: PureWindowsPath(p).parts)
    bc["subject"], bc["session"], bc["rec"] = parts.str[-4], parts.str[-3], parts.str[-2]
    bc["flagged"] = bc["reasons"].fillna("") != ""
    per_rec = bc.groupby(["subject", "session", "rec"])["flagged"].sum().rename("n_flagged").reset_index()
    per_session = per_rec.groupby(["subject", "session"])["n_flagged"].agg(
        auto_max_per_recording="max", auto_mean_per_recording="mean").reset_index()
    return per_rec, per_session


def print_hits(title, df, cols):
    print(f"\n=== {title} ===")
    if df.empty:
        print("none")
        return
    print(df[cols].round(2).to_string(index=False))
    print(f"-> subjects: {', '.join(sorted(df['subject'].unique(), key=lambda s: int(s.removeprefix('STMD'))))}")

def run_qc_check(qc_dir: Path, max_dropped_share, max_channels, show_all):
    ar = load_autoreject(qc_dir)
    per_rec, per_session = load_auto_bad_channels(qc_dir)
    sessions = ar.merge(per_session, on=["subject", "session"], how="outer")
    sessions["combined_mean"] = sessions["auto_mean_per_recording"].fillna(0) + sessions[
        "mean_interp_per_kept_epoch"].fillna(0)

    print(f"{sessions['subject'].nunique()} subjects, {len(sessions)} sessions | thresholds: "
          f"dropped share > {max_dropped_share:.3f}, channels > {max_channels}")

    print_hits(f"Epochs: more than {max_dropped_share:.0%} of a session's epochs dropped",
               sessions[sessions["dropped_share"] > max_dropped_share],
               ["subject", "session", "n_dropped", "n_epochs", "dropped_share"])
    print_hits(f"Autoreject: more than {max_channels} channels interpolated per kept epoch (session average)",
               sessions[sessions["mean_interp_per_kept_epoch"] > max_channels],
               ["subject", "session", "mean_interp_per_kept_epoch"])
    print_hits(f"Automatic bad channel detector: more than {max_channels} channels in a recording",
               per_rec[per_rec["n_flagged"] > max_channels],
               ["subject", "session", "rec", "n_flagged"])
    print_hits(f"Combined: detector channels (session mean) + autoreject average > {max_channels}",
               sessions[sessions["combined_mean"] > max_channels],
               ["subject", "session", "auto_mean_per_recording", "mean_interp_per_kept_epoch", "combined_mean"])

    if show_all:
        print("\n=== All sessions ===")
        print(sessions.sort_values(["subject", "session"]).round(2).to_string(index=False))



def match_and_order(raw_io: mne.io.Raw):
    """
    Keeps only the channels of EGI_TO_1020_MAP, renames them to their 10-20 names and orders them as CHORDER_STANDARD.
    Raises a ValueError if a mapped channel is missing, as the channel order (ch_0 ... ch_22 in all later outputs) would otherwise be wrong.
    """
    missing_channels = [ch for ch in settings.EGI_TO_1020_MAP if ch not in raw_io.ch_names]
    if missing_channels:
        raise ValueError(f"Channels {missing_channels} of EGI_TO_1020_MAP are missing, cannot order the channels as CHORDER_STANDARD.")

    # drop channels not in the LUT
    useless_channels = []
    for ch in raw_io.ch_names:
        if ch not in settings.EGI_TO_1020_MAP:
            useless_channels.append(ch)
    raw_io.drop_channels(useless_channels)

    # rename channels
    raw_io.rename_channels(settings.EGI_TO_1020_MAP)
    # reorder channels
    raw_io.reorder_channels(settings.CHORDER_STANDARD)

def rescale_to_uv(epochs: Dict[str, Dict[str, Dict[str, Any]]]):
    for subj, trial_conds in epochs.items(): # STMD1, baseline
        for trial_cond, conds in trial_conds.items(): # baseline, left_1
            for cond, epochs_data in conds.items(): # left_1, data
                print(f"Rescaling {subj} {trial_cond} {cond} to uV, original scale: {epochs_data.get_data().mean()} V")
                epochs_data.apply_function(lambda x: x * 1e6) # rescale to uV
                print(f"Rescaled {subj} {trial_cond} {cond} to uV, new scale: {epochs_data.get_data().mean()} uV")
    return epochs






def press_reaction_times(raw: mne.io.Raw, onset_codes, button_code: int) -> Dict[Tuple[str, int], float]:
    """
    Returns {(recording, onset sample): reaction time (s)} for every stimulus onset in a preprocessed session raw:
    the time to the first button press before the next stimulus onset of the same recording (no entry for onsets without a press).
    Uses the 'DIN<code>/rec_<stem>' trigger annotations of the raw, the onset sample counts from raw.first_samp like epochs.metadata['onset_s'].
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    desc_of = {code: desc for desc, code in event_id.items()}
    by_recording = {}
    for sample, _, code in events:
        desc = desc_of[code]
        if '/rec_' not in desc:
            continue
        din = int(desc.split('/')[0].removeprefix('DIN'))
        recording = desc.split('/rec_')[1].split('/')[0]
        by_recording.setdefault(recording, []).append((sample - raw.first_samp, din))

    rts = {}
    sfreq = raw.info['sfreq']
    for recording, triggers in by_recording.items():
        triggers.sort()
        onsets = [sample for sample, din in triggers if din in onset_codes]
        presses = np.array([sample for sample, din in triggers if din == button_code])
        for k, onset in enumerate(onsets):
            next_onset = onsets[k + 1] if k + 1 < len(onsets) else np.inf
            in_window = presses[(presses > onset) & (presses < next_onset)]
            if len(in_window):
                rts[(recording, onset)] = (in_window[0] - onset) / sfreq
    return rts


def drop_fast_presses(all_epochs: Dict[str, Dict[str, Dict[str, mne.Epochs]]], preproc_data: Dict[str, Dict[str, mne.io.Raw]],
                      min_reaction_time: float) -> List[Dict[str, Any]]:
    """
    Drops, in place, every epoch whose first button press comes earlier than min_reaction_time after stimulus onset,
    as such presses fall into the post-stimulus wavelet targets. Epochs without a press are kept.
    Returns one report row per subject, session and condition.
    """
    onset_codes, button_code, _ = load_trigger_codes()
    report = []
    for subj, sessions in all_epochs.items():
        for session, conds in sessions.items():
            raw = preproc_data[subj][session]
            rts = press_reaction_times(raw, onset_codes, button_code)
            for cond, epochs in conds.items():
                onset_samples = np.round(epochs.metadata['onset_s'].to_numpy() * raw.info['sfreq']).astype(int)
                fast = [i for i, (recording, onset) in enumerate(zip(epochs.metadata['recording'], onset_samples))
                        if rts.get((recording, onset), np.inf) < min_reaction_time]
                n_epochs = len(epochs)
                epochs.drop(fast, reason='fast button press', verbose=False)
                report.append({'subject': subj, 'session': session, 'condition': cond, 'n_epochs': n_epochs, 'n_dropped': len(fast)})
    return report


def patch_wavelet_target(tensor_data: torch.Tensor) -> torch.Tensor:
    """
    Computes the wavelet target of a full-length trial (channels, time over settings.WHOLE_EPOCH_WINDOW) as (channels, freqs, time over settings.EPOCH_WINDOW).
    The pre- and post-stimulus patches are transformed separately, each from its own data with reflect padding, and only then joined
    at their shared edge, so no wavelet bin contains data of the other patch.
    Returned as float16 to halve the storage, convert it to float32 before taking the log.
    """
    patch_targets = []
    for patch_window in (settings.PRE_WINDOW, settings.POST_WINDOW):
        patch_data = utils.cut_to_timewindow(tensor_data, settings.WHOLE_EPOCH_WINDOW, patch_window, settings.RSFREQ)
        mag, _ = utils.calc_wavelet(patch_data, fs=settings.RSFREQ, freqs=settings.FREQS, pad=settings.PADDING, **settings.WAVELET_PARAMS)
        # reorder to (channels, freqs, time) for LaBraM compatibility
        patch_targets.append(torch.tensor(mag, dtype=torch.float32).permute(1, 0, 2))
    return torch.cat(patch_targets, dim=-1).to(torch.float16)


def save_dataset(epochs: Dict[str, Dict[str, mne.Epochs]], path: Path, io_handler: EpochIO,
                 save_superlet: bool, save_wavelet: bool,
                 scale_to_uv: Optional[bool] = True):
    """
    Dumps all epochs found in epochs, named according to which subject and which condition(side and eccentricity) it belongs to.
    Every trial is saved as one input epoch over settings.EPOCH_WINDOW (pre- and post-stimulus patch together, split at onset during training).
    Also saves the wavelets of both patches as one target for spectral prediction if specified, see patch_wavelet_target.
    :param scale_to_uv: if set to True, scales mne content to use uV instead of V (required by LaBraM)
    :param epochs: preprocessed dictionary containing epochs of either baseline or post-patching data separated by subject and condition
    :param path: save folder path
    """
    egi_to_idx = {egi: settings.CHORDER_STANDARD.index(name) for egi, name in settings.EGI_TO_1020_MAP.items()}
    to_idx = lambda s: ';'.join(str(egi_to_idx[ch]) for ch in s.split(';') if ch in egi_to_idx) if s else ''
    manifest = []

    path.mkdir(parents=True, exist_ok=True)
    for subj in epochs.keys():
        for cond in epochs[subj].keys():
            epoch_list = epochs[subj][cond].get_data()
            meta = epochs[subj][cond].metadata.reset_index(drop=True)
            id = 1
            try:
                for epoch in epoch_list:
                    input_epoch_path = path / settings.INPUT_EPOCHS_DIR_NAME
                    input_epoch_path.mkdir(parents=True, exist_ok=True)

                    wavelet_path = path / settings.WAVELET_DIR_NAME
                    wavelet_path.mkdir(parents=True, exist_ok=True)

                    row = meta.iloc[id - 1]
                    manifest.append({'subject': subj, 'stimulus': cond, 'id': id, 'recording': row.recording,
                                     'onset_s': row.onset_s,
                                     'rec_interpolated': to_idx(row.rec_interpolated),
                                     'ar_interpolated': to_idx(row.ar_interpolated)})

                    tensor_data = torch.tensor(epoch, dtype=torch.float32)
                    # pre- and post-stimulus patch together, the last MNE sample (at EPOCH_WINDOW[1]) is dropped
                    input_data = utils.cut_to_timewindow(tensor_data, settings.WHOLE_EPOCH_WINDOW, settings.EPOCH_WINDOW, settings.RSFREQ)
                    io_handler.save(tensor_data=input_data, directory=input_epoch_path, subject=subj, condition=cond,
                                    epoch_id=id, saved_type='input')

                    if save_wavelet:
                        print(f"Computing wavelet transform for epoch {id} for {subj} {cond}")
                        wavelet_tensor = patch_wavelet_target(tensor_data)
                        print(f"Wavelet target shape (channels, freqs, time): {wavelet_tensor.shape}")

                        io_handler.save(tensor_data=wavelet_tensor, directory=wavelet_path, subject=subj, condition=cond,
                                        epoch_id=id, saved_type='wavelet')
                        print(f"Saved wavelet target for epoch {id} for {subj} {cond} to {wavelet_path}")

                    print(f"Saved epoch {id} for {subj} {cond} to {input_epoch_path}")
                    id += 1

            except Exception as e:
                print(f"ERROR: Failed to dump epoch for {subj} {cond}: {e}")
    io.save_report(manifest, "epochs_manifest.csv", path)


def epoch_file_dirs(path: Path, save_wavelet: bool) -> Dict[str, Tuple[Path, str]]:
    """
    Returns the folders save_dataset writes epoch files to under path as {name: (folder, filename marker)}.
    """
    dirs = {"input epochs": (path / settings.INPUT_EPOCHS_DIR_NAME, "_input")}
    if save_wavelet:
        dirs["wavelet targets"] = (path / settings.WAVELET_DIR_NAME, "_wavelet")
    return dirs


def check_output_dirs_empty(dirs: List[Path]):
    """
    Raises a RuntimeError if any of dirs already contains .pt files.
    Inputs and targets are paired by file name only, so files left from an earlier run would be paired with new epochs or mixed into the dataset.
    """
    non_empty = [str(d) for d in dirs if d.exists() and any(d.glob("*.pt"))]
    if non_empty:
        raise RuntimeError("Output folders already contain epoch files from an earlier run. "
                           f"Empty or remove them before running make_dataset.py: {', '.join(non_empty)}")


def check_saved_consistency(path: Path, save_wavelet: bool) -> bool:
    """
    Checks that the input epochs, the wavelet targets (if saved) and epochs_manifest.csv under path hold exactly the same (subject, condition, id) keys.
    Prints the mismatching keys and returns True if all key sets are equal.
    """
    manifest = pd.read_csv(path / "epochs_manifest.csv")
    key_sets = {"manifest": set(zip(manifest["subject"].astype(str), manifest["stimulus"].astype(str), manifest["id"].astype(int)))}
    for name, (directory, marker) in epoch_file_dirs(path, save_wavelet).items():
        keys = set()
        for f in directory.glob("*.pt"):
            _, idx, subject, cond = io_utils.vars_from_filename(f.stem, marker=marker)
            keys.add((subject, cond, idx))
        key_sets[name] = keys

    reference = key_sets["input epochs"]
    consistent = True
    for name, keys in key_sets.items():
        missing, extra = sorted(reference - keys), sorted(keys - reference)
        if missing or extra:
            consistent = False
            print(f"ERROR: {name} in {path} do not match the input epochs: {len(missing)} missing (first: {missing[:5]}), "
                  f"{len(extra)} extra (first: {extra[:5]})")
    if consistent:
        print(f"Consistency check passed for {path}: {len(reference)} epochs, identical keys in {', '.join(key_sets)}.")
    return consistent


def main(opts):
    print("Preparing preprocessed dataset...")
    # root to dataset

    # the default build goes to settings.PROCESS_DIR with its QC reports in settings.QC_DIR, any other build keeps both in its own folder
    output_dir = Path(opts.output_dir)
    finetune_dir, patch_dir = output_dir / settings.FINETUNE_DIR.name, output_dir / settings.PATCH_DIR.name
    qc_dir = settings.QC_DIR if output_dir.resolve() == settings.PROCESS_DIR.resolve() else output_dir / "qc"
    print(f"Writing the dataset to {output_dir} (QC reports: {qc_dir}), ICA: {'off' if opts.no_ica else 'on'}")

    # stop before the preprocessing if files of an earlier run could be mixed with the new ones
    session_dirs = [finetune_dir, patch_dir]
    check_output_dirs_empty([directory for path in session_dirs
                             for directory, _ in epoch_file_dirs(path, opts.save_wavelet).values()])

    io_handler = EpochIO(ext='.pt')

    # initialize unscrambly preprocessor, set required filter frequencies
    preprocessor = DataPreprocessor(
        directory=settings.DATA_ROOT,  # Root containing subject folders
        filter_params={'highpass': settings.L_FREQ, 'lowpass': settings.H_FREQ, 'notch': 50, 'notch_width': settings.NOTCH_WIDTH},
        ica_n_components=15,
        ica_method=None if opts.no_ica else 'iclabel',
        ica_params=settings.ICA_PARAMS,
        iclabel_thresholds=settings.ICLABEL_THRESHOLDS,
        bad_channels='interpolate',
        time_window=settings.WHOLE_EPOCH_WINDOW,
        baseline_win=settings.BASELINE_WINDOW,
        split_by='both',
        re_reference='average',
        auto_bad_channel_params=settings.AUTO_BAD_CHANNEL_PARAMS,
        qc_dir=qc_dir,
        cache_dir=settings.CACHE_DIR,
        autoreject_mode="local",
        autoreject_params=settings.AUTOREJECT_PARAMS,
        autoreject_window=settings.AUTOREJECT_WINDOW,
        interactive_selection=False
    )

    # preprocess subjects, order into a dict with hierarchy: subject, condition
    preproc_data = preprocessor.process_subjects(included_conditions=settings.INCLUDED_CONDITIONS,
                                                 excluded_subjects=settings.EXCLUDED_SUBJECTS,
                                                 recordings_per_condition=settings.RECORDINGS_PER_CONDITION)
    io.save_report(preprocessor.auto_bad_channel_report, "auto_bad_chans.csv", qc_dir)
    io.save_report(preprocessor.ica_report, "ica_report.csv", qc_dir)

    # resample
    for subject in preproc_data.keys():
        for condition in preproc_data[subject].keys():
            preproc_data[subject][condition].resample(settings.RSFREQ)
    print(f"Resampled to {settings.RSFREQ} Hz for all data")


    # extract epochs (no baseline correction, see settings.BASELINE_WINDOW)
    all_epochs = preprocessor.create_subject_epochs(preproc_data, output_format='epochs', manual_baseline=False)
    io.save_report(preprocessor.autoreject_report, "autoreject_report.csv", qc_dir)

    # drop trials with a button press inside the post-stimulus wavelet targets
    fast_press_report = drop_fast_presses(all_epochs, preproc_data, settings.MIN_REACTION_TIME)
    io.save_report(fast_press_report, "fast_press_report.csv", qc_dir)
    print(f"Dropped {sum(r['n_dropped'] for r in fast_press_report)} epochs with a button press earlier than "
          f"{settings.MIN_REACTION_TIME} s after onset.")

    # match channels to labram and reorder (for compatibility with electrode localization)
    for subj in all_epochs.keys():
        for cond in all_epochs[subj].keys():
            for id, epoch in all_epochs[subj][cond].items():
                try:
                    match_and_order(epoch)
                except Exception as e:
                    # stop instead of saving epochs with a wrong channel order
                    raise RuntimeError(f"Failed to reorder channels for {subj}_{cond}_{id}") from e


    print("Channel reordering to LaBraM representation complete.")

    # rescale to uV
    all_epochs = rescale_to_uv(all_epochs)  # subject - baseline/pp
    print("Created epochs for all subjects.")
    print("Dumping epochs...")
    # separate epochs
    baseline_epochs = {subj: all_epochs[subj]["Baseline"] for subj in all_epochs.keys()}
    patch_epochs = {subj: all_epochs[subj]["Post-patching"] for subj in all_epochs.keys()}

    # dump epochs data for finetuning (pre-patching)

    save_dataset(epochs=baseline_epochs, path=finetune_dir, io_handler=io_handler, scale_to_uv=True,
                 save_superlet=opts.save_superlet, save_wavelet=opts.save_wavelet)
    # dump epochs data for plasticity analysis (post-patching)
    save_dataset(epochs=patch_epochs, path=patch_dir, io_handler=io_handler, scale_to_uv=True,
                 save_superlet=opts.save_superlet, save_wavelet=opts.save_wavelet)

    print(f"Preprocessed epochs saved to {finetune_dir} and {patch_dir}.")

    print("Running quality check for bad channel rejection and autoreject...")
    run_qc_check(qc_dir, settings.QC_MAX_DROPPED_SHARE, settings.QC_MAX_DROPPED_CHANS, False)

    print("Checking that inputs, targets and manifest hold the same epochs...")
    consistent = [check_saved_consistency(path, opts.save_wavelet) for path in session_dirs]
    if not all(consistent):
        raise RuntimeError("Saved epochs are inconsistent, see the errors above. Do not train on this dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Preprocessing pipeline', add_help=False)
    parser.add_argument('--save_superlet', action='store_true', default=False,
                        help='Whether to save superlet targets for spectral prediction')
    parser.add_argument('--save_wavelet', action='store_true', default=True,
                        help='Whether to save wavelet targets for spectral prediction')
    parser.add_argument('--no_ica', action='store_true', default=False,
                        help='Skip ICA (no components removed), e.g. to measure what ICA contributes; everything else is unchanged')
    parser.add_argument('--output_dir', default=str(settings.PROCESS_DIR),
                        help='Folder for the pre_patching/ and post_patching/ dataset folders. Builds outside settings.PROCESS_DIR '
                             'keep their QC reports in <output_dir>/qc. Training and inference read settings.PROCESS_DIR')

    args = parser.parse_args()
    main(args)
