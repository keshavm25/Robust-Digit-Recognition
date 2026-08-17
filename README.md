# Robust Digit Recognition (Audio Classification)

A PyTorch pipeline that classifies spoken digits (0–9) from short audio recordings. Raw waveforms are converted into log-mel spectrograms and fed into a custom convolutional neural network (`RobustAudioCNN`). The project was built for the **EE708 Digit Recognition** Kaggle competition and reaches a **best validation Macro F1 score of 0.9806**.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline in Detail](#pipeline-in-detail)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [File Structure](#file-structure)
- [Possible Improvements](#possible-improvements)
- [License](#license)

---

## Overview

The task is a **10-class audio classification problem**: given a short `.wav` recording of a spoken digit, predict which digit (0–9) was spoken. Rather than working directly on raw waveforms, the pipeline transforms each clip into a **log-mel spectrogram** — a 2D time-frequency representation — and treats the problem as an image classification task using a CNN. This is a standard and effective approach for small-vocabulary speech classification because it lets convolutional filters learn local time-frequency patterns (like formants and pitch contours) that correspond to different spoken digits.

The notebook is self-contained and covers the full ML lifecycle: data loading, preprocessing, augmentation, model definition, training with validation tracking, checkpointing the best model, and generating a submission file for the competition's test set.

---

## Dataset

The data is expected in the Kaggle competition format:

- `train.csv` — contains an `id` column (audio filename without extension) and a `label` column (digit 0–9).
- `train_audio/train_audio/` — folder of training `.wav` files.
- `test_audio/test_audio/` — folder of test `.wav` files (unlabeled; used only for generating predictions).

The training set is split into an **80% training / 20% validation** split using `sklearn.train_test_split`, with **stratification on the label column** so that the class balance of digits is preserved in both splits. A fixed `random_state=42` is used for reproducibility.

At inference time, the test set is built directly by scanning the test audio directory for `.wav` files rather than relying on a separate test CSV, so the pipeline is robust even if a test label file isn't provided.

---

## Pipeline in Detail

### 1. Audio Loading & Preprocessing (`AdvancedDigitDataset.__getitem__`)
Each audio file is loaded with `torchaudio.load`, which returns a waveform tensor and its native sample rate. Several normalization steps are then applied so that every sample fed to the model has an identical shape and format:

- **Resampling** — if the file's sample rate differs from the target `SAMPLE_RATE` (16,000 Hz), it is resampled using `torchaudio.transforms.Resample`. This matters because mixed sample rates would otherwise produce spectrograms with inconsistent time/frequency resolution.
- **Mono conversion** — if the waveform has more than one channel (stereo), it is averaged across channels to produce a single mono channel, since digit content doesn't depend on stereo imaging.
- **Fixed-length padding/truncation** — waveforms are truncated to `MAX_LEN` (16,000 samples = 1 second at 16kHz) if too long, or zero-padded if too short. This guarantees a constant input size to the network, which is required since the model doesn't use recurrent layers or variable-length pooling.

### 2. Feature Extraction
The fixed-length waveform is converted into a **Mel-spectrogram** using `torchaudio.transforms.MelSpectrogram` with `N_MELS = 64` mel filterbanks, then converted from amplitude to a decibel (log) scale with `AmplitudeToDB`. Log-mel spectrograms compress the dynamic range of audio energy and mimic human auditory perception, which tends to produce more discriminative and learnable features than raw amplitude spectrograms.

### 3. Data Augmentation (SpecAugment)
During training only (`mode="train"`), two augmentation transforms are applied directly to the spectrogram:

- **Frequency Masking** (`freq_mask_param=15`) — randomly zeroes out a contiguous band of frequency channels.
- **Time Masking** (`time_mask_param=35`) — randomly zeroes out a contiguous span of time steps.

This is the **SpecAugment** technique, commonly used in speech recognition to reduce overfitting to specific speaker characteristics, background noise, or precise timing, by forcing the network to rely on broader, more robust patterns rather than memorizing exact spectrogram regions. Augmentation is deliberately **not applied** to validation or test data, since evaluation should reflect the true, unmodified signal.

### 4. Dataset Class Design
The `AdvancedDigitDataset` class handles three modes:
- `"train"` — returns `(spectrogram, label)` with augmentation applied.
- `"valid"` — returns `(spectrogram, label)` without augmentation.
- `"test"` — returns `(spectrogram, audio_id)` since test labels are unknown; the ID is carried through so predictions can be matched back to filenames for submission.

### 5. Model Forward Pass
Spectrograms are batched via `DataLoader` and passed through `RobustAudioCNN` (architecture detailed below) to produce class logits over the 10 digit classes.

### 6. Training Loop
Standard supervised training loop: for each epoch, the model is run in `train()` mode over all training batches with backpropagation and Adam optimization, then switched to `eval()` mode to compute validation loss and **Macro F1 score** (via `sklearn.metrics.f1_score`) on the held-out validation set. Macro F1 is used instead of accuracy because it weights all classes equally, which is a more reliable metric if class distribution is even slightly imbalanced.

### 7. Checkpointing
After each epoch's validation pass, if the Macro F1 score is the best seen so far, the model's weights are saved to disk (`best_model.pth`). This ensures the final model used for inference is the best-performing checkpoint across all epochs — not necessarily the one from the final epoch, which can overfit or fluctuate (as seen at epoch 15, where F1 actually *drops* relative to epoch 11).

### 8. Inference & Submission
After training completes, the **best saved checkpoint** (not the final epoch's live weights) is reloaded, the model is set to `eval()` mode, and predictions are generated for every file in the test audio directory. Results are written to `submission.csv` with two columns: `id` and `label`, matching the Kaggle submission format.

---

## Model Architecture

`RobustAudioCNN` is a compact 4-block convolutional network designed for single-channel (mono) spectrogram input:

```
Input: (batch, 1, 64, T)   # 1 channel, 64 mel bands, T time steps

Block 1: Conv2D(1  → 32,  k=3, s=1, p=1) → BatchNorm2d → ReLU → MaxPool2d(2,2)
Block 2: Conv2D(32 → 64,  k=3, s=1, p=1) → BatchNorm2d → ReLU → MaxPool2d(2,2)
Block 3: Conv2D(64 → 128, k=3, s=1, p=1) → BatchNorm2d → ReLU → MaxPool2d(2,2)
Block 4: Conv2D(128→ 256, k=3, s=1, p=1) → BatchNorm2d → ReLU
         AdaptiveAvgPool2d((1, 1))        # collapses spatial dims to 1x1 regardless of input size

Classifier:
  Flatten
  Dropout(0.5)
  Linear(256 → 128) → ReLU
  Dropout(0.3)
  Linear(128 → 10)   # 10 digit classes, raw logits (no softmax — handled by CrossEntropyLoss)
```

**Design notes:**
- Each convolutional block doubles the channel depth (32 → 64 → 128 → 256) while halving spatial resolution via max pooling, a standard CNN pattern that trades spatial detail for richer feature representation.
- **Batch Normalization** after every convolution stabilizes and accelerates training by normalizing activations, and also acts as a mild regularizer.
- **AdaptiveAvgPool2d((1,1))** removes any dependency on a fixed spatial input size — meaning the network would still work even if spectrogram dimensions changed slightly — and reduces the feature map to a single 256-dimensional vector per sample.
- **Two dropout layers (0.5 and 0.3)** in the classifier head are the main regularization mechanism against overfitting, especially important since spoken-digit datasets can have limited speaker diversity.

**Loss & Optimizer:**
- `nn.CrossEntropyLoss()` — standard multi-class classification loss (combines log-softmax + NLL loss internally).
- `torch.optim.Adam(lr=0.001)` — adaptive learning rate optimizer, a strong default choice for CNNs without extensive tuning.

---

## Training Details

- **Device:** Automatically selects CUDA GPU if available, else falls back to CPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).
- **Epochs:** 15
- **Batch size:** 32
- **Validation cadence:** Every epoch, immediately after the training pass.
- **Metric tracked for model selection:** Validation Macro F1 (not loss, and not accuracy).
- **Checkpoint path:** `/kaggle/working/best_model.pth`

Observed training behavior: loss decreases steadily and Macro F1 climbs from **0.60** at epoch 1 to a peak of **0.9806** at epoch 11. After epoch 11, F1 fluctuates and slightly degrades by epoch 15 (0.9658), which is why checkpointing on best validation F1 — rather than simply using the final epoch's weights — is important here.

---

## Results

| Metric | Value |
|---|---|
| **Best Validation Macro F1** | **0.9806** (epoch 11) |
| Final Epoch Val Macro F1 | 0.9658 (epoch 15) |
| Total Epochs | 15 |
| Test Set Size | 16,200 files |

**Full epoch-by-epoch training log:**

| Epoch | Train Loss | Val Loss | Val Macro F1 | Notes |
|---|---|---|---|---|
| 1  | 1.8058 | 1.1022 | 0.6026 | new best |
| 2  | 1.2537 | 0.6314 | 0.7613 | new best |
| 3  | 0.9536 | 0.3517 | 0.8786 | new best |
| 4  | 0.7522 | 0.2071 | 0.9326 | new best |
| 5  | 0.6175 | 0.2299 | 0.9181 | — |
| 6  | 0.5397 | 0.1251 | 0.9630 | new best |
| 7  | 0.4762 | 0.1684 | 0.9414 | — |
| 8  | 0.4381 | 0.1001 | 0.9661 | new best |
| 9  | 0.4121 | 0.1143 | 0.9572 | — |
| 10 | 0.3726 | 0.0931 | 0.9675 | new best |
| 11 | 0.3541 | 0.0609 | **0.9806** | **best overall** |
| 12 | 0.3283 | 0.0650 | 0.9778 | — |
| 13 | 0.3183 | 0.0644 | 0.9771 | — |
| 14 | 0.3104 | 0.0645 | 0.9772 | — |
| 15 | 0.3038 | 0.0976 | 0.9658 | — |

---

## Requirements

```
torch
torchaudio
pandas
numpy
scikit-learn
```

Install with:
```bash
pip install torch torchaudio pandas numpy scikit-learn
```

A CUDA-capable GPU is strongly recommended — training on this dataset took roughly **100 minutes on an NVIDIA Tesla T4** (per the notebook's execution timestamps). CPU training will be substantially slower.

---

## Usage

1. **Set up the data directory structure.** Update `BASE_PATH` in the notebook if your data isn't at `/kaggle/input/competitions/digitrecognition-ee708`. Expected layout:
   ```
   BASE_PATH/
   ├── train.csv
   ├── train_audio/train_audio/*.wav
   └── test_audio/test_audio/*.wav
   ```

2. **Run the notebook top to bottom.** It will, in order:
   - Load `train.csv` and split into stratified 80/20 train/validation sets.
   - Build `DataLoader`s with SpecAugment applied to the training split only.
   - Train `RobustAudioCNN` for 15 epochs, printing loss and Macro F1 per epoch.
   - Save the best-performing checkpoint to `best_model.pth` whenever validation F1 improves.
   - Reload the best checkpoint, run inference over every `.wav` file in the test directory, and write `submission.csv`.

3. **Submit `submission.csv`** to the competition — it will contain two columns, `id` and `label`, one row per test audio file.

---

## Hyperparameters

| Name | Value | Purpose |
|---|---|---|
| `SAMPLE_RATE` | 16000 Hz | Target audio sample rate after resampling |
| `MAX_LEN` | 16000 samples | Fixed waveform length (1 second) |
| `N_MELS` | 64 | Number of mel filterbank bands |
| `BATCH_SIZE` | 32 | Samples per training batch |
| `EPOCHS` | 15 | Number of full passes over the training data |
| `LR` | 0.001 | Adam optimizer learning rate |
| `freq_mask_param` | 15 | Max width of SpecAugment frequency mask |
| `time_mask_param` | 35 | Max width of SpecAugment time mask |

---

## File Structure

```
├── digit-recognition-2.ipynb   # Full training + inference pipeline (this notebook)
├── best_model.pth              # Saved best model weights (generated during training)
└── submission.csv              # Test set predictions in Kaggle submission format (generated)
```

---

## Possible Improvements

- **Cross-validation** instead of a single train/val split, for a more robust estimate of generalization.
- **Learning rate scheduling** (e.g., `ReduceLROnPlateau` or cosine annealing) to squeeze out further gains after the F1 plateau around epoch 11.
- **Early stopping** based on validation F1 to avoid unnecessary training once performance plateaus/degrades.
- **Additional augmentations** such as time-shifting, pitch-shifting, or additive background noise, to further improve robustness beyond SpecAugment alone.
- **Variable-length audio handling** — currently all clips are forced to exactly 1 second; a more flexible approach (e.g., dynamic padding per batch) could reduce information loss for longer utterances.
- **Model ensembling** — averaging predictions across multiple checkpoints or architectures for a small accuracy boost.

---

## License

Specify a license of your choice (e.g., MIT) if you plan to make this repository public.
