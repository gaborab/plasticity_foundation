"""
Planted-change checks of the sliding LDA (HANDOFF_location_decoding.md, Task C), on baseline data only.

C1/C2: every subject's baseline trials are split into two pseudo-sessions, a random half of every condition each
(split_halves, seeded with --split_seed). Each half is split into K_FOLDS within-subject folds and decoded like a real
session by decoding.run_sliding_lda.run_lda, the first half saved as "baseline" (pseudo pre), the second as "postpatch"
(pseudo post), so the Task A / B cells of notebooks/LDA.ipynb analyse them unchanged with sliding_name = the output
folder. Without a planted change ("null") no Task B contrast may come out significant (the handoff repeats this with 3 splits).
C3/C4: the same, with a change planted into the pseudo post trials of one condition at a time (--conditions): a delay
of every --delays size (core.decoding.data.plant_delay, no searchlight), and an attenuation of the posterior evoked
response by every --attenuations size (plant_attenuation; with --class_specific only its location-specific part,
plant_class_attenuation). The folds of the planted pseudo post equal the null ones.

Output: outputs/inference_outputs/{name}_{plant}_split{split_seed}/{fold}/, as decoding/run_sliding_lda.py, with plant
"null", e.g. "delay_right_2_20ms", "attenuation_right_2_0.25" or "classattenuation_right_2_1".
Run from the project root: python -m decoding.run_lda_injection --split_seed 0
(the further null splits: python -m decoding.run_lda_injection --split_seed 1 --null_only)
"""
import argparse

from core import io, settings
from core.datasets import KFoldSplitter
from core.decoding import data
from core.decoding.data import Epochs, split_halves
from decoding.run_sliding_lda import CH_NAMES, run_lda

K_FOLDS = 3  # within each pseudo-session, which holds half of the trials


def main(opts):
    epochs = io.EpochIO(ext='.pt').load_all_epochs(settings.FINETUNE_INPUT_EPOCHS_DIR, marker="_input")
    pseudo_pre, pseudo_post = split_halves(epochs, opts.split_seed)

    def split(session_epochs):
        splitter = KFoldSplitter(input_data=session_epochs, target_data_dir=None, target_mode="classify", seed=settings.SEED,
                                 k_folds=K_FOLDS, no_press_threshold=settings.NO_PRESS_FROM_SUBJECT)
        return splitter.split_within_subjects()

    folds = {"baseline": split(pseudo_pre), "postpatch": split(pseudo_post)}

    # both pseudo-sessions are baseline trials: their interpolated channels are in the baseline manifest
    baseline_interpolations = io.load_interpolated_channels(settings.FINETUNE_DIR / "epochs_manifest.csv")
    interpolations = {"baseline": baseline_interpolations, "postpatch": baseline_interpolations}
    searchlight_bands = None
    if opts.searchlight:
        searchlight_bands = opts.searchlight_bands if opts.searchlight_bands is not None else list(range(len(settings.DECODING_BANDS)))

    def decode(planted_post: Epochs, plant: str, bands):
        """Decodes pseudo pre and the planted pseudo post into {name}_{plant}_split{seed}, on the null run's folds."""
        post_folds = split(planted_post)
        if [fold.keys for fold in post_folds] != [fold.keys for fold in folds["postpatch"]]:
            raise ValueError(f"The folds of the planted pseudo post ({plant}) differ from the null ones")
        run_lda({"baseline": folds["baseline"], "postpatch": post_folds}, interpolations,
                settings.INFERENCE_DIR / f"{opts.name}_{plant}_split{opts.split_seed}", n_jobs=opts.n_jobs,
                searchlight_bands=bands, n_neighbors=opts.n_neighbors, per_subject=opts.per_subject)

    decode(pseudo_post, "null", searchlight_bands)
    if opts.null_only:
        return
    for condition in opts.conditions:
        for delay in opts.delays:
            delayed = data.plant_delay(pseudo_post, [condition], delay=delay, fs=settings.RSFREQ)
            decode(delayed, f"delay_{condition}_{round(delay * 1000)}ms", None)  # the searchlight isn't needed for a delay
        posterior = [name in settings.POSTERIOR_CHANNELS for name in CH_NAMES]
        for attenuation in opts.attenuations:
            plant_args = {"channel_mask": posterior, "epoch_window": settings.EPOCH_WINDOW, "fs": settings.RSFREQ,
                          "window": settings.PLANT_ATTENUATION_WINDOW}
            if opts.class_specific:
                attenuated = data.plant_class_attenuation(pseudo_post, [condition], attenuation=attenuation, **plant_args,
                                                          location_conditions=list(settings.LOCATION_CLASSES))
                decode(attenuated, f"classattenuation_{condition}_{attenuation:g}", searchlight_bands)
            else:
                attenuated = data.plant_attenuation(pseudo_post, [condition], attenuation=attenuation, **plant_args)
                decode(attenuated, f"attenuation_{condition}_{attenuation:g}", searchlight_bands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Planted-change checks of the sliding LDA on baseline pseudo-sessions")
    parser.add_argument("--name", default="lda_injection", help="Output folder prefix in settings.INFERENCE_DIR, _{plant}_split{seed} appended")
    parser.add_argument("--split_seed", type=int, default=settings.SEED, help="Seed of the split into pseudo-sessions")
    parser.add_argument("--null_only", action="store_true",
                        help="Run the null only (C2), e.g. for the further split seeds, without the planted runs")
    parser.add_argument("--conditions", nargs="*", default=["right_2", "left_2"],
                        help="Conditions planted one at a time (handoff C3: R_mid, then L_mid)")
    parser.add_argument("--delays", type=float, nargs="*", default=[0.01, 0.02, 0.04], help="Planted delays in s")
    parser.add_argument("--attenuations", type=float, nargs="*", default=[0.1, 0.25, 0.5],
                        help="Planted attenuations, fractions of the class-average evoked response subtracted")
    parser.add_argument("--class_specific", action="store_true",
                        help="Attenuate only the location-specific part of the response (plant_class_attenuation: the class "
                             "mean minus the mean over the location classes), saved as classattenuation_{condition}_{size}")
    parser.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs over the time points")
    parser.add_argument("--searchlight", action="store_true", help="Also run the searchlight")
    parser.add_argument("--searchlight_bands", type=int, nargs="+", default=None,
                        help="Indices into DECODING_BANDS of the searchlight bands (0 = broadband), default all")
    parser.add_argument("--n_neighbors", type=int, default=settings.N_NEIGHBOURS, help="Nearest electrodes in every searchlight cluster")
    parser.add_argument("--per_subject", action="store_true",
                        help="Separate models per subject instead of one pooled model (use another --name to keep the pooled runs)")
    main(parser.parse_args())
