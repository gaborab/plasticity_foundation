"""
Adds the signed error to the stored metrics of a model's inference outputs (run_inference.py), in place, for outputs
written before utils.get_inference_metrics computed it: per channel signed_error = recon_mean - signal_mean, the mean
output minus the mean target over the window's bins (dB for wavelet targets, > 0: the predicted level is too high), and
per band signed_error_bandwise; the trial-level value is the mean over the channels. Unlike level_error (its absolute
value) and the MAE, it is linear in the target, so a level change the model does not follow shows up in it without the
target's spread mixing in.

Every group with stored metrics ({window}_metrics_json and {window}_channel_metrics_json attributes, both patches of every
trial) in every {session}_reconstruction.h5 of every fold is updated; groups that already have it are skipped, as HDF5
doesn't reuse the space of rewritten attributes (the first run grows a file by ~6 %, h5repack gives the space back). The
files must not be open elsewhere (e.g. loaded in a notebook kernel) while it runs.

Run from the repository root, e.g.
    python add_signed_error.py --model_name masking_10_19
"""
import argparse
import json
import warnings
from pathlib import Path

import h5py
import numpy as np

from core import settings

CHANNEL_SUFFIX = "_channel_metrics_json"


def add_channel_signed_errors(channel_metrics: dict) -> dict:
    """
    Adds signed_error (and signed_error_bandwise, if the band means are stored) to every channel's metrics in
    channel_metrics ({channel: metrics dict}) and returns the trial-level values, the means over the channels.
    """
    for metrics in channel_metrics.values():
        metrics["signed_error"] = metrics["recon_mean"] - metrics["signal_mean"]
        if metrics.get("signal_mean_bandwise"):
            metrics["signed_error_bandwise"] = {band: metrics["recon_mean_bandwise"][band] - value
                                                for band, value in metrics["signal_mean_bandwise"].items()}
    # interpolated channels have NaN means, all-NaN bands give NaN without RuntimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        trial = {"signed_error": float(np.nanmean([m["signed_error"] for m in channel_metrics.values()]))}
        bandwise = [m["signed_error_bandwise"] for m in channel_metrics.values() if "signed_error_bandwise" in m]
        if bandwise:
            trial["signed_error_bandwise"] = {band: float(np.nanmean([errors[band] for errors in bandwise])) for band in bandwise[0]}
    return trial


def add_signed_errors_file(h5_path: Path) -> int:
    """Adds the signed error to every group with stored metrics of h5_path, returns the number of updated (group, window) pairs."""
    groups = []
    with h5py.File(h5_path, "r+") as h5file:
        h5file.visititems(lambda _, obj: groups.append(obj) if isinstance(obj, h5py.Group)
                          and any(key.endswith(CHANNEL_SUFFIX) for key in obj.attrs) else None)
        n_updated = 0
        for group in groups:
            for window in [key[:-len(CHANNEL_SUFFIX)] for key in group.attrs if key.endswith(CHANNEL_SUFFIX)]:
                channel_metrics = json.loads(group.attrs[f"{window}{CHANNEL_SUFFIX}"])
                if not all("signal_mean" in m and "recon_mean" in m for m in channel_metrics.values()):
                    continue  # e.g. the channel metrics of a class run, without levels
                if all("signed_error" in m for m in channel_metrics.values()):
                    continue  # already added: rewritten attributes take new space in the file, the old space is not reused
                metrics = json.loads(group.attrs[f"{window}_metrics_json"])
                metrics.update(add_channel_signed_errors(channel_metrics))
                group.attrs[f"{window}{CHANNEL_SUFFIX}"] = json.dumps(channel_metrics)
                group.attrs[f"{window}_metrics_json"] = json.dumps(metrics)
                n_updated += 1
    return n_updated


def add_signed_errors(run_dir: Path) -> None:
    """Adds the signed error to every *_reconstruction.h5 of every fold folder of run_dir (an inference output folder)."""
    files = sorted(path for fold in run_dir.iterdir() if fold.is_dir() and fold.name.isdigit()
                   for path in fold.glob("*_reconstruction.h5"))
    if not files:
        raise FileNotFoundError(f"No *_reconstruction.h5 files in the fold folders of {run_dir}")
    for path in files:
        print(f"{path.relative_to(run_dir)}: signed error added to {add_signed_errors_file(path)} (group, window) pairs (others skipped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add the signed error to the stored metrics of a model's inference outputs")
    parser.add_argument("--model_name", type=str, required=True, help="inference output folder under settings.INFERENCE_DIR")
    add_signed_errors(settings.INFERENCE_DIR / parser.parse_args().model_name)
