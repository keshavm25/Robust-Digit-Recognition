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
