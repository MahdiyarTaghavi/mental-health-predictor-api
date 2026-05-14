import os

import shap
import joblib
import pandas as pd
import numpy as np
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

# TreeExplainer works with XGBoost and Random Forest
# For Logistic Regression it falls back to LinearExplainer
if hasattr(model, "feature_importances_"):
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.LinearExplainer(model, masker=shap.maskers.Independent(data=None))


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

    # SHAP — per-prediction explanation
    shap_values = explainer.shap_values(X)
    values = shap_values[0, :, 1]

    shap_contributions = {
        feat: round(float(val), 4)
        for feat, val in zip(feature_names, values)
    }

    shap_sorted = sorted(
        shap_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    global_top = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # SHAP baseline — average model output across training data
    expected = explainer.expected_value
    baseline = float(expected[1] if isinstance(expected, (list, np.ndarray)) else expected)

    return {
        "prediction": prediction,
        "label": "Likely to seek treatment" if prediction == 1 else "Unlikely to seek treatment",
        "confidence": round(confidence * 100, 1),
        "model_used": model_name,
        "model_auc": round(auc, 4),
        "top_global_influential_features": [f for f, _ in global_top],
        "shap_explanation": {
            "baseline": round(baseline, 4),
            "top_contributors": [
                {"feature": f, "contribution": v}
                for f, v in shap_sorted[:5]
            ]
        }
    }