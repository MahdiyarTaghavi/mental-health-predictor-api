import os
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = Path(os.path.join(BASE_DIR, "models", "model.pkl"))

# Load once at import time — FastAPI will reuse this across requests
_bundle = joblib.load(_MODEL_PATH)
model = _bundle["model"]
encoders = _bundle["encoders"]
feature_names = _bundle["feature_names"]
feature_importance = _bundle["feature_importance"]
model_name = _bundle["model_name"]
auc = _bundle["auc"]


def predict(input_data: dict) -> dict:
    """
    Takes a dict of raw user inputs, encodes them the same way training did,
    runs the model, and returns prediction + confidence + top contributing features.
    """
    row = {}

    for feat in feature_names:
        value = input_data.get(feat)

        if feat in encoders:
            le = encoders[feat]
            row[feat] = le.transform([str(value)])[0]
        else:
            # Numeric feature (e.g. Age)
            row[feat] = float(value) if value is not None else 0.0

    X = pd.DataFrame([row])[feature_names]

    prediction = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0][prediction])

    # Return top 5 features that most influenced this prediction
    top_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "prediction": prediction,                          # 0 = unlikely, 1 = likely to seek treatment
        "label": "Likely to seek treatment" if prediction == 1 else "Unlikely to seek treatment",
        "confidence": round(confidence * 100, 1),          # e.g. 78.3
        "top_influential_features": [f for f, _ in top_features],
        "model_used": model_name,
        "model_auc": round(auc, 4),
    }
