---
# Credit Risk Modeling with PyTorch

End-to-end credit default prediction pipeline using the UCI Credit Card dataset, comparing a classical baseline with a neural network model and evaluating performance using industry-relevant metrics.

This project focuses on **practical ML workflow**, not leaderboard tricks.
---

## Problem Statement

Predict whether a credit card client will default on payment next month based on demographic, financial, and payment history features.

This is a standard **binary classification** problem with:

- real-world class imbalance
- noisy tabular features
- business-relevant evaluation metrics

---

## Dataset

**UCI – Default of Credit Card Clients**

- 30,000 records
- 24 input features
- Binary target: `default payment next month`
- Mixed feature types (demographic, behavioral, financial)

The dataset is automatically downloaded (or reused if already present) and parsed from the original Excel format.

---

## Project Structure

```
credit-risk-pytorch/
│
├─ src/
│  ├─ train.py          # Training & evaluation pipeline
│  ├─ dataset.py        # Dataset loading & preprocessing
│  ├─ download_data.py  # Robust dataset download & caching
│  └─ model.py          # PyTorch MLP model
│
├─ data/                # Dataset storage (auto-managed)
├─ runs/                # Saved artifacts & metrics
├─ requirements.txt
└─ README.md
```

---

## Modeling Approach

### Baseline

- Logistic Regression
- Serves as a strong, interpretable reference model

### Neural Model

- Feedforward MLP (PyTorch)
- Trained on standardized tabular features
- Optimized using binary cross-entropy loss

The goal is not complexity, but **measurable improvement over a reasonable baseline**.

---

## Evaluation Metrics

Models are evaluated on a held-out test set using:

- **ROC-AUC** – ranking quality
- **PR-AUC** – performance under class imbalance
- **KS statistic** – separation power (common in credit risk)
- **Brier score** – probability calibration

---

## Results (Test Set)

| Model               | ROC-AUC   | PR-AUC    | KS        | Brier     |
| ------------------- | --------- | --------- | --------- | --------- |
| Logistic Regression | ~0.71     | ~0.49     | ~0.36     | ~0.21     |
| PyTorch MLP         | **~0.77** | **~0.54** | **~0.41** | **~0.19** |

The neural model consistently outperforms the baseline across all metrics.

---

## How to Run

### 1. Setup environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.train --download
```

- Downloads dataset if not present
- Trains baseline + neural model
- Saves metrics and artifacts under `runs/`

Subsequent runs reuse the cached dataset automatically.

---

## Outputs

Each run creates a timestamped folder under `runs/` containing:

- evaluation metrics
- model artifacts
- plots (ROC, PR, calibration, etc.)

---

## Why This Project

This project demonstrates:

- end-to-end ML pipeline ownership
- baseline-first modeling discipline
- use of business-relevant evaluation metrics
- reproducible training and evaluation
- clean separation of data, model, and training logic

It is designed to reflect **real applied data science work**, not toy experiments.

---
