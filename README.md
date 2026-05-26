# ECG Anomaly Detection with CNN Autoencoders

A deep learning project for detecting anomalies in ECG (electrocardiogram) signals using convolutional autoencoder architectures, developed as part of **AAI-628 — Applied Artificial Intelligence** at **Stevens Institute of Technology**.

**Author:** Mauricio Castro · Fall 2025

---

## 🫀 Project Objective

Build an ECG anomaly detection model that is clinically useful by prioritizing **high recall (sensitivity)** to minimize false negatives (missed anomalies). The approach uses reconstruction-based anomaly detection: CNN autoencoders are trained to reconstruct normal ECG beats, and anomalies are identified by higher reconstruction error.

---

## 📁 Repository Structure

```
ECG-anomaly-detection/
├── Class_Project_Final.py                              # Theory notes & method selection
├── ProjectAAI628_model1-checkpoint.ipynb               # Model 1 - Baseline CNN AE
├── ProjectAAI628_model2-checkpoint.ipynb               # Model 2 - + BatchNorm
├── ProjectAAI628_model_3-checkpoint.ipynb              # Model 3 - + Dropout
├── ProjectAAI628_model_4-checkpoint.ipynb              # Model 4 - LeakyReLU + Nadam
├── ProjectAAI628_model_5-checkpoint.ipynb              # Model 5 - + EarlyStopping
├── ProjectAAI628_win_model_6.ipynb                     # ✅ Model 6 - Best model
├── Final_Report_ECG_Anomaly_dectection_6_models_.pdf   # Final report
├── ecg.csv                                             # Dataset
└── README.md
```

---

## 📊 Dataset

- **Name:** ECG5000
- **Source:** [ECG Dataset (TensorFlow)](http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv)
- **Type:** Time-series · Collective anomalies
- **Format:** Heartbeat segments of 140 timesteps, labeled normal / anomaly
- **Split:** Train/test via `train_test_split` with fixed seed (42) for reproducibility

---

## 📚 Background & Method Selection

### Anomaly Type
This project targets **collective anomalies** in time-series ECG data — where sequences of data points together form an anomalous pattern that wouldn't be flagged individually.

### Deep Learning Methods Considered

| Method | Description | Best For |
|--------|-------------|----------|
| **Autoencoders** | Train on normal data; anomalies have high reconstruction error | ECG signals, time-series |
| **LSTM Networks** | Capture temporal dependencies in time-series anomalies | Sequential data |
| **VAEs** | Probabilistic reconstruction; anomalies deviate from latent distribution | Complex distributions |
| **GANs** | Generate normal samples; anomalies are poorly reconstructed | Image/audio data |

### Hybrid & Advanced Approaches
- **Semi-supervised learning:** Train mostly on normal data with a few labeled anomalies
- **Ensemble methods:** Combine multiple detectors for robustness
- **Attention mechanisms:** Focus on critical segments in sequential data (e.g., ECG)

> Best for: Real-world deployments where anomalies are rare but critical.

### Why Autoencoders?
For ECG anomaly detection, **autoencoders** are the most effective approach since they:
- Capture collective anomalies in time-series signals
- Work well in **unsupervised settings** where anomalies are rare or unlabeled
- Identify anomalies through reconstruction error without requiring labeled anomaly examples

---

## ⚙️ Anomaly Detection Threshold

All models use a consistent threshold derived from normal training data:

```
Threshold = μ_train + σ_train
```

Where μ and σ are the mean and standard deviation of reconstruction errors on normal training samples. This ensures fair and comparable evaluation across all experiments.

---

## 🧠 Model Architecture

All models are **CNN Autoencoders** (`cnn_anomaly_detector`) that learn to reconstruct normal ECG signals. Anomaly detection is based on reconstruction error exceeding the learned threshold.

**Stack:** TensorFlow / Keras · NumPy · Pandas · Scikit-learn · Matplotlib · Seaborn

**Hardware:** AMD Ryzen 7 7800X3D · 64 GB RAM · CPU-only · Windows 11

---

## 🔬 Experiment Results

| Model | Main Strategy Change | Recall | FN | Accuracy | Clinical Outcome |
|-------|----------------------|--------|-----|----------|------------------|
| 1 | Baseline CNN AE | 0.8822 | 49 | 0.901 | Baseline |
| 2 | + BatchNorm | 0.9712 | 12 | 0.952 | ✅ Major improvement |
| 3 | + Dropout | 0.9639 | 15 | 0.951 | Slight sensitivity drop |
| 4 | LeakyReLU + Nadam | 0.2909 | 295 | 0.665 | ❌ Failed |
| 5 | + EarlyStopping | 0.5529 | 114 | 0.774 | ❌ Sensitivity collapsed |
| **6** | **Strided CNN + LayerNorm + PReLU + EarlyStopping** | **0.9832** | **7** | **0.945** | ✅ **Best** |

### False Negatives by Model

![False Negatives vs Model Index](false_negatives_chart.png)

Model 4 (LeakyReLU + Nadam) caused a dramatic spike to 295 FNs — a complete sensitivity collapse. Model 6 recovered and achieved the fewest false negatives (7), making it the most clinically viable configuration.

---

## 📋 Experiment Log

### Model 1 — Baseline CNN Autoencoder
- Conv1D + MaxPooling (encoder) / UpSampling (decoder), sigmoid output
- Optimizer: Adam · Loss: MAE · Epochs: 20 · Batch: 512
- **Recall: 0.8822 | FN: 49 | Accuracy: 0.901**
- 📝 Baseline performs reasonably but 49 missed anomalies is too high for clinical use

### Model 2 — + Batch Normalization
- Added BatchNorm after each Conv1D layer in encoder and decoder
- Optimizer: Adam · Loss: MAE · Epochs: 50 · Batch: 128
- **Recall: 0.9712 | FN: 12 | Accuracy: 0.952**
- 📝 Largest single improvement — FN dropped from 49 → 12

### Model 3 — + Dropout Regularization
- Added Dropout (encoder: 0.2; decoder: 0.5 → 0.1) on top of BatchNorm
- Optimizer: Adam · Loss: MAE · Epochs: 50 · Batch: 128
- **Recall: 0.9639 | FN: 15 | Accuracy: 0.951**
- 📝 Slight sensitivity drop — Dropout needs careful tuning for recall-sensitive tasks

### Model 4 — LeakyReLU + Nadam Optimizer
- Replaced activations with LeakyReLU, switched optimizer to Nadam
- **Recall: 0.2909 | FN: 295 | Accuracy: 0.665**
- 📝 ❌ Complete failure — optimizer/activation combination destroyed anomaly separability

### Model 5 — + Early Stopping
- Added EarlyStopping on top of Model 4 configuration
- **Recall: 0.5529 | FN: 114 | Accuracy: 0.774**
- 📝 ❌ Partial recovery but sensitivity still collapsed

### Model 6 — Strided CNN + LayerNorm + PReLU + EarlyStopping ✅ Best
- Replaced pooling with strided convolutions, LayerNorm, PReLU activations, EarlyStopping
- **Recall: 0.9832 | FN: 7 | Accuracy: 0.945**
- 📝 ✅ Best clinical outcome — temporal structure preserved, sensitivity maximized

---

## 🏆 Final Selected Model — Model 6

**Why Model 6?**
- Highest recall: **98.3%** — only 7 missed anomalies
- Strided convolutions preserve temporal structure better than pooling
- LayerNorm + PReLU provide stable, well-calibrated activations
- EarlyStopping prevents recall drift during training

---

## 💡 Key Takeaways

- **Recall is the key metric** for clinical anomaly detection — accuracy alone is insufficient
- **Batch Normalization** provided the largest early improvement in sensitivity
- **Dropout** improves robustness but must be tuned carefully to avoid hurting recall
- **Optimizer and activation choices** can catastrophically alter anomaly separability (Model 4)
- **Strided convolutions** outperform pooling-based designs for temporal ECG signals
- **Unsupervised autoencoders** are ideal when anomalies are rare or unlabeled

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mcastroavi/ECG-anomaly-detection.git
   cd ECG-anomaly-detection
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
   ```

3. Open the final model notebook:
   ```bash
   jupyter notebook ProjectAAI628_win_model_6.ipynb
   ```

---

## 📌 Course

**AAI-628 — Data Acquisition, Modeling and Analysis: Deep Learning **
 
M.S. Applied Artificial Intelligence 
Stevens Institute of Technology · Fall 2025
