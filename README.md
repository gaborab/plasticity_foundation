# EEG Foundational Model Fine-Tuning & Reconstruction Pipeline

> **Project Status: Active Development (In Progress)**  
> The core pipeline is functional, but there are still major architectural improvements and code cleanups ongoing as the project advances.

---

## Overview

This repository houses an end-to-end PyTorch pipeline designed to adapt foundational EEG deep learning models (such as LaBraM) to specialized datasets. By engineering an autoencoding head for signal reconstruction, the pipeline enables quantitative evaluation and visual characterization of induced changes in neural activity.

The project is structured as a modular Python package (utilizing a vendored package layout with `__init__.py`) and handles the entire lifecycle: data preprocessing, multi-subject cross-validation, fine-tuning, and interactive performance visualization.

---

## Pipeline Architecture & Implementation Steps

### 1. Data Preparation (`make_dataset.py`)
Extracts, preprocesses, and standardizes raw EEG data into structured tensor formats optimized for model ingestion.

* **Preprocessing Routine (via `unscrambly`):**
  * **Bandpass Filter:** 0.1 – 75 Hz
  * **Notch Filter:** 50 Hz
  * **Resampling:** 200 Hz
  * **Unit Standardization:** $\mu V$ (microvolts)
  * **Spatial Correctness:** Strict channel matching and ordering to match the foundational model's training expectations.
* **Workflow:**
  1. `process_subjects()`: Generates a dictionary separating data by subject.
  2. `create_subject_epochs()`: Organizes data hierarchically (`subject` $\rightarrow$ `condition` $\rightarrow$ `stimulus_type`). Extracts individual epochs for both baseline and post-patching trials (as well as trial-averaged data).
  3. **Tensor Dumping:** Exports individual epochs to the process directory using the naming convention:  
     `{subject}_{stimulus_type}_epoch{number}.pt`

---

### 2. Fine-Tuning (`run_general_finetune.py`)
Performs supervised training of the transformer layers and the custom autoencoding head on preprocessed EEG epochs.

* **Data Splitting & Cross-Validation:**
  * **`KFoldSplitter`:** Automates data loading and sets up $k$-fold cross-validation ($k=6$ by default). Splits data into Train (66%), Validation (16%), and Test (16%).
  * **Subject-Mix Spliting Strategy:** Ensures training, validation, and test sets contain mixed subjects and conditions to prevent the model from overfitting to individual subject signatures.
  * **`FinetuneLoader`:** PyTorch Dataset wrapper handling sample loading for the training, validation, and test splits.
* **Model Definition (`EEGAutoencoder`):**
  * Extends `nn.Module`.
  * Loads the foundational LaBraM backbone.
  * Attaches a custom autoencoding head comprising a 2-layer MLP optimized with Mean Squared Error (MSE) loss.
* **Training Mechanics:**
  * Trains a separate model for each cross-validation fold using a modified fine-tuning script.
  * Utilizes **Cosine Annealing** for learning rate and weight decay scheduling to stabilize deeper transformer layers.
  * Logs training, validation (`evaluate()`), and testing metrics per epoch, saving model checkpoints and metadata to the output directory.

---

### 3. Visualization & Analysis (`training_results_viewer.ipynb`)
An interactive Jupyter notebook environment used to track model performance, inspect reconstructions, and run statistical comparisons.

* **Inputs:** Model name, checkpoint epoch selection, and preprocessed data paths.
* **Workflow:**
  * `KFoldSplitter` reloads preprocessed files to reconstruct test/train splits.
  * Test set samples are passed through loaded model checkpoints.
  * Interactive UI sliders allow side-by-side visual comparison of **original vs. reconstructed EEG signals**.
  * Results can be exported directly as image assets to the plot directory.

---

## Project Structure

```text
├── make_dataset.py           # Preprocessing and tensor export script
├── run_general_finetune.py   # Multi-subject fine-tuning and training loop
├── training_results_viewer.ipynb # Interactive evaluation and visualization notebook
└── __init__.py               # Package initialization
