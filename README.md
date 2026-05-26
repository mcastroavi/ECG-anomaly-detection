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
├── ProjectAAI628_model1-checkpoint.ipynb    # Baseline CNN AE
├── ProjectAAI628_model2-checkpoint.ipynb    # + BatchNorm
├── ProjectAAI628_model_3-checkpoint.ipynb   # + Dropout
├── ProjectAAI628_model_4-checkpoint.ipynb   # LeakyReLU + Nadam
├── ProjectAAI628_model_5-checkpoint.ipynb   # + EarlyStopping
├── ProjectAAI628_test_model_6.ipynb         # Test notebook - Model 6
├── Code_ProjectAAI628_model_6.ipynb         # ✅ Final best model
├── false_negatives_chart.png                # FN comparison chart
└── README.md
```

---

## 📊 Dataset

- **Source:** [ECG Dataset (TensorFlow)](http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv)
- **Format:** Heartbeat segments of 140 timesteps, labeled normal / anomaly
- **Split:** Train/test split via `train_test_split` with fixed seed (42) for reproducibility

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

The chart shows how false negatives fluctuated across experiments. Model 4 (LeakyReLU + Nadam) caused a dramatic spike to 295 FNs — a complete sensitivity collapse. Model 6 recovered and achieved the fewest false negatives (7), making it the most clinically viable configuration.

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
- 📝 ❌ Partial recovery but sensitivity still collapsed — EarlyStopping alone can't fix a bad optimizer choice

### Model 6 — Strided CNN + LayerNorm + PReLU + EarlyStopping ✅ Best
- Replaced pooling with strided convolutions, LayerNorm, PReLU activations, EarlyStopping
- **Recall: 0.9832 | FN: 7 | Accuracy: 0.945**
- 📝 ✅ Best clinical outcome — architectural choices that preserve temporal structure outperform pooling-based designs

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
   jupyter notebook Code_ProjectAAI628_model_6.ipynb
   ```

---

## 📌 Course

**AAI-628 — Applied Artificial Intelligence**  
Stevens Institute of Technology · Fall 2025
