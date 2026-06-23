# EEG-Based Brain Computer Interface Classification

## Overview

This project implements an end-to-end Brain-Computer Interface (BCI) pipeline for EEG signal classification. The workflow combines signal preprocessing, artifact removal, spatial feature extraction, and machine learning to classify neural activity from multi-channel EEG recordings.

The objective is to predict feedback events from EEG signals while mitigating noise and physiological artifacts.

---

## Pipeline

### 1. Data Loading

Training and testing EEG recordings are loaded from CSV files using chunked processing to efficiently handle large datasets.

### 2. Signal Preprocessing

#### Notch Filtering

* Removes power-line interference at 50 Hz.

#### Band-Pass Filtering

* Butterworth filter
* Frequency range: 1–40 Hz

These steps suppress noise while retaining relevant EEG frequency components.

### 3. Artifact Removal

Eye-movement artifacts are removed using EOG regression.

For each EEG channel:

* Linear regression is trained using EOG measurements.
* Predicted ocular artifacts are subtracted from EEG signals.
* Cleaned signals are retained for downstream analysis.

### 4. Dataset Balancing

Class imbalance is addressed through random resampling to ensure equal representation of feedback-event classes.

### 5. Feature Extraction

#### Common Spatial Patterns (CSP)

CSP is applied to cleaned EEG channels to learn spatial filters that maximize variance differences between classes.

Configuration:

* Channels used: First 8 EEG channels
* Components: 4 CSP features

The resulting features provide compact and discriminative representations of neural activity.

### 6. Classification

A Support Vector Machine (SVM) classifier is trained using:

* Linear kernel
* Balanced class weights
* Probability estimation enabled

The model is evaluated using a train-validation split and classification metrics.

### 7. Inference Pipeline

For unseen test recordings:

1. Apply preprocessing
2. Remove EOG artifacts
3. Extract CSP features
4. Generate prediction probabilities
5. Create submission file

Output format:

```text
IdFeedBack,Prediction
FB00001,0.81
FB00002,0.24
...
```

## Tech Stack

* Python
* Pandas
* NumPy
* SciPy
* MNE
* Scikit-Learn
* Joblib

## Key Techniques

* EEG Signal Processing
* Brain-Computer Interfaces
* Common Spatial Patterns (CSP)
* Artifact Removal
* Support Vector Machines
* Signal Filtering
* Feature Engineering

## Future Improvements

* Riemannian geometry-based classifiers
* Filter Bank CSP (FBCSP)
* Deep-learning-based EEG decoding
* Hyperparameter optimization
* Subject-independent evaluation
