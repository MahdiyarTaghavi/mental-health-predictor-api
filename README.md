# Mental Health in Tech — Predictor API

A machine learning API that predicts whether a tech worker is likely to seek mental health treatment, based on workplace and personal factors. It also includes a computer vision endpoint for facial emotion detection as a complementary signal, exploring a multimodal approach to mental health assessment.

Built with **Python**, **scikit-learn**, **XGBoost**, **FastAPI**, and **DeepFace**.

---

## What It Does

This project combines two AI approaches to mental health assessment:

**1. Survey-based Prediction**
Takes workplace and personal factors as input (company size, mental health
benefits, family history, etc.) and predicts whether a tech worker is likely
to seek mental health treatment. Returns a prediction, confidence score, and
the most influential features that drove the result.

Five models are trained and compared:
- Logistic Regression — implemented from scratch using gradient descent
- Decision Tree: intentionally untuned to demonstrate overfitting
- Random Forest: bagging ensemble, sklearn baseline
- XGBoost: boosting ensemble, sklearn baseline
- Manual Bagging: bagging ensemble implemented from scratch

The best performing model is automatically selected and saved.
Training includes manual k-fold cross-validation and randomized
hyperparameter tuning, both implemented from scratch.

**2. Facial Emotion Detection**
Accepts a facial image and detects the dominant emotional state using a
two-stage computer vision pipeline: face detection followed by emotion
classification. Maps the detected emotion to a mental health relevance note.

---

## Architecture
```text
Survey Input
    ↓
Data Cleaning
    ↓
Feature Encoding
    ↓
Model Training (5 models)
    ↓
Cross-Validation
    ↓
Hyperparameter Tuning
    ↓
Best Model
    ↓
Prediction + Confidence + SHAP Explanation


Image Input
    ↓
Face Detection
    ↓
Emotion Classification
    ↓
Dominant Emotion + Mental Health Note
```
---

## Explainability — SHAP

The `/predict` endpoint uses **SHAP (SHapley Additive exPlanations)** to
explain every individual prediction. Unlike global feature importance —
which shows what the model relies on across all predictions on average —
SHAP shows exactly how each input feature pushed the prediction up or down
for a specific person.

- **Baseline** — the average prediction across the training dataset (starting point)
- **Positive contribution** — pushed the prediction toward "likely to seek treatment"
- **Negative contribution** — pushed the prediction against it

---

## Computer Vision Approach

The emotion detection endpoint uses **DeepFace**, a purpose-built facial analysis library used in CV and biometrics research. Unlike generic image classifiers, DeepFace runs a dedicated two-stage CV pipeline internally:

1. **Face detection** — locates and extracts the face region from the image
2. **Emotion classification** — classifies the detected face into one of 7 emotional states: happy, sad, angry, fear, disgust, surprise, neutral

This multimodal design means the API can assess mental health indicators from both structured survey data and unstructured visual input — two fundamentally different signal types.

---
## Tests

The project includes a test suite covering both the API endpoints and the core prediction logic.

### Running the tests

```bash
pytest tests/ -v
```

### What is tested

**Endpoint tests** (`tests/test_endpoints.py`) — integration tests that send real HTTP requests to the API and verify response status codes and response structure:

**Core tests** (`tests/test_core.py`) — unit tests that call the prediction function directly, bypassing the API layer:

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/MahdiyarTaghavi/mental-health-predictor-api.git
cd mental-health-predictor-api
```

### 2. Download the dataset
The dataset is not included due to Kaggle's licensing terms.

1. Go to 👉 https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey
2. Click **Download** (free Kaggle account required)
3. Create a `training_data/` folder in the project root
4. Extract the zip and place `survey.csv` inside it

```
mental-health-predictor-api/
└── training_data/
    └── survey.csv
```

### 3. Build the Docker image
```bash
docker build -t mental-health-predictor .
```

### 4. Run with Docker
```bash
docker compose -f docker-compose.yml up
```

This will:
1. Train the model automatically and save it to `models/`
2. Start the API server

Server runs at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|-----------------|----------------------------------------------|
| GET | `/model-info` | Model name, AUC score, top 5 features |
| POST | `/predict` | Survey-based mental health prediction |
| POST | `/analyze-emotion` | Facial emotion detection from image upload |

Full request/response schemas are available in the interactive Swagger docs at `/docs`.

---

## Tech Stack

| Layer | Tools |
|------------|-------------------------------------------------------|
| ML | scikit-learn, XGBoost, joblib |
| CV | DeepFace, OpenCV, Pillow |
| API | FastAPI, Pydantic, uvicorn |
| Data | pandas, numpy |

---

## Notes
- `models/` and `data/survey.csv` are excluded from version control
- Re-run `core/train.py` any time to regenerate the model from scratch
- DeepFace downloads its emotion model on first request and caches it locally
- The `/analyze-emotion` endpoint accepts single-face images only (JPEG, PNG, WebP, max 5MB)