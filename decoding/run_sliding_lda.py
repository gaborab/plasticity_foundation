"""
Time-resolved location decoding (HANDOFF_location_decoding.md, Task A): a sliding shrinkage LDA, one per time point and
band, trained within each session.

Trials, window and labels are those of the LaBraM location decoder: the baseline folds of a LaBraM run (its keys.json),
and the post-patching folds of split_within_subjects with the same seed and k, whose test sets are the ones its inference
saved (postpatch_keys.json, checked). Every trial's CLASS_PATCH_WINDOW (-0.5 to 0.5 s) is band-passed into every
DECODING_BANDS band and decimated to 100 Hz (core.decoding.data), giving trials x channels x bands x times.
Per fold and session one pooled model is trained on the fold's training and validation trials of all subjects (the LDA
needs no model selection), every feature standardized per subject with the subject's training trials, and applied to
the fold's test trials. Every trial is in one test set, so every trial gets one out-of-fold prediction per band and time.

With --searchlight, the same sliding LDA also runs on every channel's cluster alone (the channel and its N_NEIGHBOURS
nearest electrodes, core.utils.build_neighbor_mask) in one band (--searchlight_band, default the broadband signal):
where location is decodable.

Output (outputs/inference_outputs/{name}/{fold}/), in the layout of the class runs of run_inference.py:
    baseline_reconstruction.h5, postpatch_reconstruction.h5
        attrs: sets (band names), times (s), envelope
        {subject}/{condition}/{id}: target (class id), output (n_bands, n_times, n_classes) LDA decision values
        (their softmax is the class probabilities), attrs: interpolated
    baseline_searchlight.h5, postpatch_searchlight.h5 (with --searchlight): the same, sets = the channels at the centre
        of the clusters, output (n_clusters, n_times, n_classes), attrs also band and n_neighbors
Run from the project root: python -m decoding.run_sliding_lda --keys_from class_nomask_10_22
"""
import argparse
import json
import time

import h5py
import numpy as np

from core import io, settings
from core import utils as core_utils
from core.datasets import KFoldSplitter
from core.decoding.data import band_pass_trials, decoding_times, stack_class_samples
from core.decoding.models import decode_bands, searchlight

CH_NAMES = [name.replace("EEG ", "").replace("-REF", "") for name in settings.CHORDER_STANDARD]


def band_data(datasets, envelope):
    """Band-passed inputs (n_trials, n_chans, n_bands, n_times), class ids and metadata of the trials of datasets."""
    X, y, metas = stack_class_samples(datasets)
    return band_pass_trials(X, settings.RSFREQ, settings.DECODING_BANDS, settings.DECODING_DECIM, envelope), y, metas


def save_outputs(path, outputs, labels, metas, times, set_names, **attrs):
    """Writes the decision values of every test trial, see the module docstring. attrs: further file attributes."""
    with h5py.File(path, "w") as h5file:
        h5file.attrs["sets"] = list(set_names)
        h5file.attrs["times"] = times
        for name, value in attrs.items():
            h5file.attrs[name] = value
        for output, label, meta in zip(outputs, labels, metas):
            subject, condition, idx = meta["subject"], meta["condition"], meta["id"]
            io.save_inference_trial_hdf(subject, condition, idx, h5file, int(label), output.astype(np.float32), window_metrics={}, patch=None)
            h5file[f"{subject}/{condition}/{idx}"].attrs["interpolated"] = np.flatnonzero(~meta["allowed"].numpy())


def main(opts):
    keys_dir = settings.FINETUNE_OUTPUT_DIR / opts.keys_from
    k_folds = json.load(open(keys_dir / "metadata.json", encoding="utf-8"))["hyperparameters"]["k_folds"]
    splitter = KFoldSplitter(input_data_dir=settings.FINETUNE_INPUT_EPOCHS_DIR, target_data_dir=None, target_mode="classify",
                             seed=settings.SEED, k_folds=k_folds, no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
    baseline_folds = splitter.split_from_keys(io.load_training_split_keys(keys_dir, k_folds))
    postpatch_splitter = KFoldSplitter(input_data_dir=settings.PATCH_INPUT_EPOCHS_DIR, target_data_dir=None, target_mode="classify",
                                       seed=settings.SEED, k_folds=k_folds, no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
    postpatch_folds = postpatch_splitter.split_within_subjects()

    # the post-patching test sets have to be those of the LaBraM inference
    saved_test_keys = io.load_training_split_keys(settings.INFERENCE_DIR / opts.keys_from, k_folds, filename="postpatch_keys.json")
    for k, (fold, saved) in enumerate(zip(postpatch_folds, saved_test_keys), start=1):
        if set(map(tuple, fold.keys["test"])) != set(saved["test"]):
            raise ValueError(f"Post-patching fold {k} differs from the test set of the {opts.keys_from} inference")

    interpolations = {"baseline": io.load_interpolated_channels(settings.FINETUNE_DIR / "epochs_manifest.csv"),
                      "postpatch": io.load_interpolated_channels(settings.PATCH_DIR / "epochs_manifest.csv")}
    class_window = {"epoch_window": settings.EPOCH_WINDOW, "patch_window": settings.CLASS_PATCH_WINDOW, "fs": settings.RSFREQ,
                    "condition_classes": settings.CONDITION_CLASS_IDS}
    output_root = settings.INFERENCE_DIR / opts.name
    clusters = core_utils.build_neighbor_mask(n_neighbours=opts.n_neighbors, n_chans=settings.N_CHANNELS) if opts.searchlight else None
    searchlight_band = list(settings.DECODING_BANDS)[opts.searchlight_band]
    print(f"Sliding LDA on {len(settings.DECODING_BANDS)} bands ({'envelopes' if opts.envelope else 'band-passed signals'}), "
          f"{k_folds} folds, output {output_root}")

    for k in range(k_folds):
        fold_dir = output_root / str(k + 1)
        fold_dir.mkdir(parents=True, exist_ok=True)
        for session, fold in (("baseline", baseline_folds[k]), ("postpatch", postpatch_folds[k])):
            start = time.time()
            train, val, test = fold.initialize_class_datasets(**class_window, interpolations=interpolations[session])
            X_train, y_train, metas_train = band_data([train, val], opts.envelope)
            X_test, y_test, metas_test = band_data([test], opts.envelope)
            outputs = decode_bands(X_train, y_train, [m["subject"] for m in metas_train], X_test,
                                   [m["subject"] for m in metas_test], n_jobs=opts.n_jobs)
            times = decoding_times(settings.CLASS_PATCH_WINDOW, settings.RSFREQ, settings.DECODING_DECIM, X_test.shape[-1])
            save_outputs(fold_dir / f"{session}_reconstruction.h5", outputs, y_test, metas_test, times,
                         settings.DECODING_BANDS, envelope=opts.envelope)
            if opts.searchlight:
                b = opts.searchlight_band
                cluster_outputs = searchlight(X_train[:, :, b], y_train, [m["subject"] for m in metas_train], X_test[:, :, b],
                                              [m["subject"] for m in metas_test], clusters, n_jobs=opts.n_jobs)
                save_outputs(fold_dir / f"{session}_searchlight.h5", cluster_outputs, y_test, metas_test, times, CH_NAMES,
                             envelope=opts.envelope, band=searchlight_band, n_neighbors=opts.n_neighbors)
            print(f"Fold {k + 1} {session}: {len(y_train)} training, {len(y_test)} test trials, {time.time() - start:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Time-resolved location decoding with a sliding LDA per band")
    parser.add_argument("--keys_from", required=True, help="LaBraM class run whose folds (keys.json, postpatch_keys.json) are used")
    parser.add_argument("--name", default="sliding_lda", help="Output folder in settings.INFERENCE_DIR")
    parser.add_argument("--envelope", action="store_true", help="Decode the Hilbert amplitude of every band instead of the band-passed signal")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs over the time points")
    parser.add_argument("--searchlight", action="store_true", help="Also decode from every channel's cluster alone")
    parser.add_argument("--searchlight_band", type=int, default=0, help="Index into DECODING_BANDS of the searchlight band (0 = broadband)")
    parser.add_argument("--n_neighbors", type=int, default=settings.N_NEIGHBOURS, help="Nearest electrodes in every searchlight cluster")
    main(parser.parse_args())
