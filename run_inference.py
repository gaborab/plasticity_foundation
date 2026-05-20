import argparse

import numpy as np
import torch
from torch import utils as torch_utils

from core.architectures import EEGAutoencoder, EEGSpectralDecoder
from core.datasets import KFoldSplitter, FinetuneDataset
from core.io import EpochIO
from core import settings
from libs.labram import utils
from einops import rearrange
from core import utils as core_utils
import h5py

def run_reconstruction(model, dataset, ch_names, device, output_path, freqs=None, freq_bands=None):
    sampler = torch_utils.data.SequentialSampler(dataset)
    loader = torch_utils.data.DataLoader(dataset, sampler=sampler, batch_size=256)

    with h5py.File(output_path / "reconstruction_results.h5", "w") as h5file:
        with torch.no_grad():
            input_chans = utils.get_input_chans(ch_names)
            for i, batch in enumerate(loader):
                EEG  = batch[0]
                target = batch[1]
                metadata = batch[2] if len(batch) > 2 else None
                EEG = EEG.float().to(device, non_blocking=True) / 100
                EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
                target = target.to(device, dtype=torch.float32, non_blocking=True).flatten(1)

                with torch.cuda.amp.autocast():
                    output = model(EEG, input_chans=input_chans)

                output_np = output.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                # calculate MAE, bandwise MAE, bandwise R2 and R2 metrics for each trial and save to disk
                for j in range(output_np.shape[0]):
                    trial_output = output_np[j]
                    trial_target = target_np[j]
                    trial_metadata = metadata[j]
                    subject, condition, idx = trial_metadata["subject"], trial_metadata["condition"], trial_metadata["id"]

                    trial_path = f"{subject}/{condition}/{idx}"
                    trial_group = h5file.create_group(trial_path)
                    trial_group.create_dataset("target", data = trial_target, compression="gzip")
                    trial_group.create_dataset("output", data = trial_output, compression="gzip")

                    ch_prefix = "ch_"
                    metrics = core_utils.get_inference_metrics(trial_output, trial_target, freqs=freqs, freq_bands=freq_bands, ch_prefix=ch_prefix)

                    metrics_group = trial_group.create_group("metrics")
                    for key, value in metrics.items(): # ch2 -> {r2, mae, r2_bandwise, mae_bandwise}
                        if not ch_prefix in key:
                            metrics_group.create_dataset(key, data = value, compression="gzip")
                        else:
                            # for channel-wise metrics, save each channel as a separate dataset
                            for ch_metric, ch_value in value.items():
                                channel_group = metrics_group.create_group(key)
                                channel_group.create_dataset(ch_metric, data = ch_value, compression="gzip")



                    print(f"Saved reconstruction and metrics for trial {trial_path} to disk.")
                    print("Structure of saved trial group:")
                    trial_group.visititems(lambda name, obj: print(name))  # print the structure of the saved group for verification
                    # save metrics to disk

                    # save reconstructed output and target to disk for later analysis


def get_args():
    parser = argparse.ArgumentParser(description='Run inference with a trained model')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model architecture to use for inference')
    parser.add_argument('--target_mode', type=str, default='wavelet', help='Target mode for the model (e.g., wavelet, superlet)')
    parser.add_argument('--postpatch', action='store_true', help='Whether to run inference on postpatch data')
    return parser.parse_args()

def main(opts):
    io_handler = EpochIO(ext='.pt')
    # loads data in format {subject: {condition: [(id1, epoch_tensor1), (id2, epoch_tensor2), ...]}}
    finetune_all_data = io_handler.load_all_epochs(settings.FINETUNE_EPOCHS_DIR)
    postpatch_all_data = io_handler.load_all_epochs(settings.PATCH_EPOCHS_DIR)

    splitter = KFoldSplitter(data_dir=settings.FINETUNE_EPOCHS_DIR, seed=settings.SEED, all_data=finetune_all_data, k_folds=6)
    folds = splitter.split_by_subject()

    ch_names = settings.CHORDER_STANDARD
    # cut off "-REF" and "EEG " from channel names for compatibility with get_input_chans function
    ch_names = [name.split(' ')[-1].split('-')[0] for name in ch_names]

    freq_window = settings.SPECTRAL_DECODER_PARAMS["freq_window"]
    n_freqs = settings.SPECTRAL_DECODER_PARAMS["n_freqs"]
    freqs = np.geomspace(freq_window[0], freq_window[1], n_freqs) if opts.target_mode in ["wavelet", "superlet"] else None
    bands = settings.SPECTRAL_DECODER_PARAMS["bands"] if opts.target_mode in ["wavelet", "superlet"] else None

    for i, fold in enumerate(folds):
        print(f"Running inference for fold {i + 1}/{len(folds)}")
        testing_ids = list(fold.testing_epochs.keys())
        print("Testing subjects for this fold:", testing_ids)
        _, _, testing_dataset = fold.initialize_datasets(target_mode=opts.target_mode, wavelet_dir=settings.FINETUNE_WAVELET_DIR, log=True, normalize=True)

        if opts.target_mode == "autoencoder":
            model_params = settings.AUTOENCODER_PARAMS
            model = EEGAutoencoder(**model_params)
        elif opts.target_mode in ["wavelet", "superlet"]:
            model_params = settings.SPECTRAL_DECODER_PARAMS
            model = EEGSpectralDecoder(**model_params)
        else:
            raise ValueError(f"Invalid target mode: {opts.target_mode}")

            # loading checkpoint into model
        checkpoint_path = settings.FINETUNE_OUTPUT_DIR / opts.model_name / str(i + 1) / "checkpoint-best_val.pth"
        print("using checkpoint:", checkpoint_path)
        if not checkpoint_path.exists():
            continue
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        # set model to eval mode and move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        output_path = settings.INFERENCE_DIR / opts.model_name / str(i + 1)
        output_path.mkdir(parents=True, exist_ok=True)
        run_reconstruction(model, testing_dataset, ch_names, device, output_path, freqs, bands)

        if opts.postpatch:
            postpatch_fold_items = {subject: postpatch_all_data[subject] for subject in testing_ids}
            postpatch_dataset = FinetuneDataset(postpatch_fold_items, target_mode=opts.target_mode, wavelet_dir=settings.PATCH_WAVELET_DIR, window_size=settings.WINDOW_SIZE, log=True, normalize=True)






if __name__ == '__main__':
    opts = get_args()
    main(opts)