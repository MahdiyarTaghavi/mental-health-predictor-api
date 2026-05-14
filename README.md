# Mental Health in Tech — Predictor API

A machine learning API that predicts whether a tech worker is likely to seek mental health treatment, based on workplace and personal factors. It also includes a computer vision endpoint for facial emotion detection as a complementary signal, exploring a multimodal approach to mental health assessment.

Built with **Python**, **scikit-learn**, **XGBoost**, **FastAPI**, and **DeepFace**.

---

## What It Does

This project combines two AI approaches to mental health assessment:

**1. Survey-based Prediction**
Takes workplace and personal factors as input (company size, mental health benefits, family history, etc.) and predicts whether a tech worker is likely to seek mental health treatment. Returns a prediction, confidence score, and the most influential features that drove the result.

**2. Facial Emotion Detection**
Accepts a facial image and detects the dominant emotional state using a two-stage computer vision pipeline: face detection followed by emotion classification. Maps the detected emotion to a mental health relevance note.

---

## Architecture
Survey Input → ML Model (XGBoost/Random Forest) → Prediction + Confidence + Feature Importance
Image Input  → Face Detection → Emotion Classification → Dominant Emotion + Mental Health Note
---

## Explainability — SHAP

The `/predict` endpoint uses **SHAP (SHapley Additive exPlanations)** to explain every individual prediction. Unlike global feature importance — which shows what the model relies on across all predictions on average — SHAP shows exactly how each input feature pushed the prediction up or down for a specific person.

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

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/MahdiyarTaghavi/mental-health-predictor-api.git
cd mental-health-predictor-api
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
The dataset is not included due to Kaggle's licensing terms.

1. Go to 👉 https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey
2. Click **Download** (free Kaggle account required)
3. Extract the zip and place `survey.csv` inside the `data/` folder

### 4. Train the model
```bash
python core/train.py
```
Compares three models (Logistic Regression, Random Forest, XGBoost), picks the best one by AUC score, and saves it to `models/model.pkl`.

### 5. Run the server
```bash
uvicorn src.main:app --reload --reload-dir src
```
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