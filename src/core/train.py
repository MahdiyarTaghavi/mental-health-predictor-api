import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Phase 1. Load the data
df = pd.read_csv(os.path.join(BASE_DIR, "data", "survey.csv"))
print(f"Dataset shape: {df.shape}")

# Phase 2. Clean the data

# These columns are either irrelevant or mostly empty — no point keeping them
df.drop(columns=["Timestamp", "comments", "state", "Country"], inplace=True)

# This is what we're trying to predict: did the person seek treatment?
# Convert 'Yes'/'No' to 1/0 so the model can work with it
df["treatment"] = (df["treatment"] == "Yes").astype(int)

# The gender field was free text — people wrote everything from 'male' to 'Male' to 'm'
# Group them into three clean buckets
def normalize_gender(g):
    g = str(g).strip().lower()
    if g in ["male", "m", "man", "cis male", "male (cis)", "male."]:
        return "Male"
    elif g in ["female", "f", "woman", "cis female", "femail", "female (cis)"]:
        return "Female"
    else:
        return "Other"

df["Gender"] = df["Gender"].apply(normalize_gender)

# Some respondents entered unrealistic ages like 5 or 999 — filter those out
df = df[(df["Age"] >= 18) & (df["Age"] <= 75)]

# Phase 3. Feature Engineering

# All remaining categorical columns
cat_cols = df.select_dtypes(include="str").columns.tolist()

# Models only understand numbers, not text
# LabelEncoder converts each category to an integer — simple and works well with tree models
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le  # save so the API can encode user input identically

# Phase 4. Split
X = df.drop(columns=["treatment"])
y = df["treatment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Phase 5. Train & Compare Three Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        # use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {"model": model, "auc": auc}
    print(f"\n{'─'*40}")
    print(f"  {name}  |  AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

# Phase 6. Pick the Best Model
best_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_name]["model"]
print(f"\n✓ Best model: {best_name}  (AUC {results[best_name]['auc']:.4f})")

# Phase 7. Feature Importance (for the /predict API response)
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    # Logistic Regression: use absolute coefficient as proxy
    importances = np.abs(best_model.coef_[0])

feature_importance = dict(zip(X.columns.tolist(), importances.tolist()))
top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 features:")
for feat, score in top_features:
    print(f"  {feat}: {score:.4f}")

# Phase 8. Save Everything the API Needs
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "model_name": best_name,
        "encoders": encoders,
        "feature_names": X.columns.tolist(),
        "feature_importance": feature_importance,
        "auc": results[best_name]["auc"],
    },
    os.path.join(BASE_DIR, "models", "model.pkl"),
)

print("\n✓ Saved to models/model.pkl — ready for the API.")
