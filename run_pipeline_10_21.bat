# 1. environment (Python >= 3.10, PyTorch 2.0.1 + CUDA 11.8 wheels, resolved by uv)
uv sync

# 2. preprocess raw EEG -> cleaned epoch tensors + wavelet targets + QC reports
uv run python make_dataset.py

# 3. fine-tune one model per fold (6 folds by default)
uv run python run_general_finetune.py --model_name masking_10_22 --model_desc "masking model using 45 cut, hiding interpolations as inputs and downsampled" --masked

# 4. reconstruct held-out baseline + post-patching trials, compute metrics
uv run python run_inference.py --model_name masking_10_22 --postpatch