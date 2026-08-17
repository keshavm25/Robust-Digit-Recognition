# 🎙️ Robust Spoken Digit Recognition

### EE708 Term Project | Deep Learning | Audio Classification

A deep-learning pipeline for **spoken digit recognition (0–9)** from `.wav` audio recordings using **PyTorch, TorchAudio, Mel Spectrograms, SpecAugment, and a custom CNN architecture**.

---

## 📌 Project Overview

This project implements an end-to-end audio classification system for recognizing **spoken digits (0–9)** from audio recordings. The model uses standardized audio preprocessing, **Mel Spectrogram** feature extraction, **SpecAugment**, and a custom **Convolutional Neural Network (CNN)** for classification.

The primary evaluation metric is **Macro F1 Score**.

---

## 🎯 Objective

Build a robust deep-learning model capable of classifying **spoken digits (0–9)** from `.wav` recordings while maximizing the **Macro F1 score**.

---

## 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **TorchAudio**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **CUDA / GPU**
- **Kaggle**

---

## 🔄 End-to-End Pipeline

```text
Raw .WAV Audio
      ↓
Audio Loading
      ↓
Resampling → 16 kHz
      ↓
Stereo → Mono
      ↓
Pad / Truncate → 1 Second
      ↓
Mel Spectrogram
      ↓
Amplitude-to-dB Conversion
      ↓
SpecAugment
      ↓
Custom CNN
      ↓
10-Class Digit Prediction
      ↓
Macro F1 Evaluation
      ↓
Best Model Checkpoint
      ↓
Test Inference
      ↓
submission.csv


Mel Spectrogram
       │
       ├── Frequency Masking
       │
       └── Time Masking
              ↓
      Augmented Spectrogram

Input Mel Spectrogram
        ↓
Conv2D (1 → 32)
BatchNorm + ReLU
MaxPooling
        ↓
Conv2D (32 → 64)
BatchNorm + ReLU
MaxPooling
        ↓
Conv2D (64 → 128)
BatchNorm + ReLU
MaxPooling
        ↓
Conv2D (128 → 256)
BatchNorm + ReLU
        ↓
Adaptive Average Pooling
        ↓
Flatten
        ↓
Dropout (0.5)
        ↓
Linear (256 → 128)
ReLU
        ↓
Dropout (0.3)
        ↓
Linear (128 → 10)
        ↓
Digit Prediction



Epoch 01 → Macro F1: 0.6026
Epoch 02 → Macro F1: 0.7613
Epoch 03 → Macro F1: 0.8786
Epoch 04 → Macro F1: 0.9326
Epoch 05 → Macro F1: 0.9181
Epoch 06 → Macro F1: 0.9630
Epoch 07 → Macro F1: 0.9414
Epoch 08 → Macro F1: 0.9661
Epoch 09 → Macro F1: 0.9572
Epoch 10 → Macro F1: 0.9675
Epoch 11 → Macro F1: 0.9806  ← Best
Epoch 12 → Macro F1: 0.9778
Epoch 13 → Macro F1: 0.9771
Epoch 14 → Macro F1: 0.9772
Epoch 15 → Macro F1: 0.9658




digit-recognition/
│
├── digit-recognition-2.ipynb
├── README.md
├── submission.csv
│
└── results/
    └── best_model.pth


git clone https://github.com/<your-username>/digit-recognition.git
cd digit-recognition

pip install torch torchaudio pandas numpy scikit-learn

jupyter notebook digit-recognition-2.ipynb


BASE_PATH = "/kaggle/input/competitions/digitrecognition-ee708"



| Component              | Configuration                |
| ---------------------- | ---------------------------- |
| Input                  | `.wav` audio                 |
| Sampling Rate          | **16 kHz**                   |
| Input Duration         | **1 second**                 |
| Feature Representation | **Mel Spectrogram**          |
| Mel Bins               | **64**                       |
| Augmentation           | **Frequency + Time Masking** |
| Backbone               | **Custom CNN**               |
| Normalization          | **Batch Normalization**      |
| Regularization         | **Dropout + SpecAugment**    |
| Optimizer              | **Adam**                     |
| Learning Rate          | **0.001**                    |
| Loss                   | **Cross Entropy**            |
| Classes                | **10**                       |
| Evaluation             | **Macro F1**                 |
| Best Validation F1     | **0.9806**                   |



best_model.pth

submission.csv

id,label

