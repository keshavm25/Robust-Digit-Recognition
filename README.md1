# Spoken Digit Recognition (Audio Classification)

A PyTorch pipeline that classifies spoken digits (0–9) from raw audio clips using Mel-spectrogram features and a convolutional neural network. Built for the **EE708 Digit Recognition** Kaggle competition.

## Overview

The model converts raw `.wav` audio into log-mel spectrograms and trains a CNN to predict the spoken digit. It reaches a **best validation Macro F1 of 0.9806**.

## Pipeline

1. **Audio loading & preprocessing** — Load `.wav` files with `torchaudio`, resample to 16 kHz, downmix to mono, and pad/truncate to a fixed length of 16000 samples (1 second).
2. **Feature extraction** — Convert waveforms to 64-band Mel-spectrograms, then to log scale (dB).
3. **Data augmentation** — Apply SpecAugment (frequency masking + time masking) on training data only, to improve robustness to speaker/recording variation.
4. **Model** — A 4-block CNN (`RobustAudioCNN`) with Conv2D + BatchNorm + ReLU + MaxPool layers, followed by adaptive average pooling and a fully connected classifier head with dropout.
5. **Training** — Trained for 15 epochs with Adam optimizer and cross-entropy loss, tracking validation Macro F1 and checkpointing the best model.
6. **Inference & submission** — Loads the best checkpoint, runs inference on the test set, and writes predictions to `submission.csv`.

## Model Architecture

\```
Input (1 x 64 x T mel-spectrogram)
 → Conv2D(32) → BatchNorm → ReLU → MaxPool
 → Conv2D(64) → BatchNorm → ReLU → MaxPool
 → Conv2D(128) → BatchNorm → ReLU → MaxPool
 → Conv2D(256) → BatchNorm → ReLU
 → AdaptiveAvgPool2d(1x1)
 → Flatten → Dropout(0.5) → Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→10)
\```

## Results

| Metric | Value |
|---|---|
| Best Validation Macro F1 | **0.9806** |
| Epochs | 15 |
| Batch size | 32 |
| Learning rate | 0.001 |

Training progress (final epochs):

| Epoch | Train Loss | Val Loss | Val Macro F1 |
|---|---|---|---|
| 11 | 0.3541 | 0.0609 | 0.9806 (best) |
| 12 | 0.3283 | 0.0650 | 0.9778 |
| 13 | 0.3183 | 0.0644 | 0.9771 |
| 14 | 0.3104 | 0.0645 | 0.9772 |
| 15 | 0.3038 | 0.0976 | 0.9658 |

## Requirements

\```
torch
torchaudio
pandas
numpy
scikit-learn
\```

## Usage

1. Place the competition data under `BASE_PATH` (expects `train.csv`, `train_audio/train_audio/`, `test_audio/test_audio/`).
2. Run the notebook top to bottom. It will:
   - Split `train.csv` into 80/20 train/validation (stratified by label).
   - Train the CNN and save the best checkpoint to `best_model.pth`.
   - Run inference on the test audio and write `submission.csv` with columns `id, label`.

## Hyperparameters

| Name | Value |
|---|---|
| Sample rate | 16000 Hz |
| Max audio length | 16000 samples |
| Mel bands | 64 |
| Batch size | 32 |
| Epochs | 15 |
| Learning rate | 0.001 |

## File Structure

\```
├── digit-recognition-2.ipynb   # Full training + inference pipeline
├── best_model.pth              # Saved best model weights (generated)
└── submission.csv              # Test predictions (generated)
\```

## License

Specify a license of your choice (e.g., MIT) if you plan to make this repository public.
