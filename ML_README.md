# Mental Health Treatment Predictor — ML Component

## Overview
This project trains a machine learning pipeline to predict whether a person
is likely to seek mental health treatment, based on workplace survey data.
It includes two algorithms implemented from scratch alongside sklearn baselines:

- **Logistic Regression** — implemented from scratch using gradient descent
- **Manual Bagging** — ensemble of decision trees implemented from scratch
- **Decision Tree, Random Forest, XGBoost** — sklearn baselines for comparison

---

## Project Structure
```text
src/core/
├── train.py                          # Main training pipeline (run this first)
├── model.py                          # Loads trained model and exposes predict()
├── test_model.py                     # Quick test script for a sample prediction
├── cross_validation.py               # Manual k-fold cross-validation from scratch
├── tuning.py                         # Randomized hyperparameter search
├── plots.py                          # Matplotlib evaluation plots
└── models/
    ├── linear_model/
    │   └── logistic_regression.py    # Logistic Regression from scratch
    └── ensemble/
        └── bagging.py                # Manual Bagging from scratch
```
---

## Setup

Install dependencies:
```bash
pip install -r requirements_ml.txt
```

---

## How to Run

### Step 1 — Train the models
```bash
python src/core/train.py
```

This will:
- Load and clean the survey dataset
- Train all 5 models with cross-validation and hyperparameter tuning
- Print AUC results and pre/post tuning comparison
- Save evaluation plots to `outputs/`
- Save the best model to `models/model.pkl`

### Step 2 — Test a prediction
```bash
python src/core/test_model.py
```

This loads the saved model and runs a sample prediction, printing:
- Predicted label and confidence
- Model name and AUC
- Top globally influential features
- SHAP explanation showing which features drove this specific prediction

---

## Results

| Model | CV AUC | Test AUC |
|---|---|---|
| Logistic Regression (scratch) | 0.7606 | 0.7555 |
| Decision Tree | 0.8804 | 0.8871 |
| Random Forest | 0.8875 | 0.9112 |
| XGBoost | 0.8841 | 0.9098 |
| Manual Bagging (scratch) | 0.8713 | 0.9118 |

Best model: **Manual Bagging** (Test AUC 0.9118)

---

## Notes
- Training data must be present at `training_data/survey.csv`
- All plots are saved to `outputs/` which is excluded from version control
- `RANDOM_SEED = 42` is used throughout for full reproducibility