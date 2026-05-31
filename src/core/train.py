import sys
from pathlib import Path
from typing import Literal, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from models.linear_model.logistic_regression import LogisticRegression
from models.ensemble.bagging import ManualBaggingClassifier
from cross_validation import cross_validate
from tunings.tuning import tune_all
from plots import plot_all

BASE_DIR = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42


def _clean_dataset() -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
        Load, clean, encode, and split the survey dataset into features and target.

        Drops irrelevant columns, normalizes free-text gender entries into three
        buckets, filters unrealistic ages, and label-encodes all categorical
        columns. Encoders are saved so the API can encode user input identically
        at inference time.

        Returns
        -------
        X : pd.DataFrame
            Encoded feature matrix of shape (n_samples, n_features).
        y : pd.Series
            Binary target labels of shape (n_samples,) where 1 indicates
            the respondent sought mental health treatment.
        encoders : dict
            Dictionary mapping categorical column names to their fitted
            LabelEncoder instances.
    """
    # Phase 1. Load the data
    df = pd.read_csv(BASE_DIR / "training_data" / "survey.csv")
    print(f"Dataset shape: {df.shape}")

    # Phase 2. Clean the data

    # These columns are either irrelevant or mostly empty — no point keeping them
    df.drop(columns=["Timestamp", "comments", "state", "Country"], inplace=True)

    # This is what we're trying to predict: did the person seek treatment?
    # Convert 'Yes'/'No' to 1/0 so the model can work with it
    df["treatment"] = (df["treatment"] == "Yes").astype(int)

    df["Gender"] = df["Gender"].apply(_normalize_gender)

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

    return X, y, encoders

def _normalize_gender(g: str) -> Literal["Male", "Female", "Other"]:
    """
    The gender field is free text: people wrote everything from 'male' to 'Male' to 'm'
    Group them into three clean buckets

    Parameters
    ----------
    g: str
        The gender field value
    Returns
    -------
    str
        Normalized gender string
    """
    g = str(g).strip().lower()
    if g in ["male", "m", "man", "cis male", "male (cis)", "male."]:
        return "Male"
    elif g in ["female", "f", "woman", "cis female", "femail", "female (cis)"]:
        return "Female"
    else:
        return "Other"

def _train_all(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Train and evaluate all models on the given data.

    Parameters
    ----------
    models : dict
        Dictionary of model name to model instance.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.

    Returns
    -------
    dict
        Results keyed by model name, each containing:
        - model : fitted model instance
        - train_auc : float, AUC on training data
        - test_auc : float, AUC on test data
        - cv_mean : float, mean CV AUC across folds
        - cv_std : float, standard deviation of CV AUC across folds
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results = {}
    for name, model in models.items():

        # Logistic Regression needs scaled data: tree models are scale-invariant
        if isinstance(model, LogisticRegression):
            X_tr = X_train_scaled
            X_te = X_test_scaled
            X_cv = X_train_scaled  # use scaled version for cross-validation too
        else:
            X_tr = X_train
            X_te = X_test
            X_cv = X_train

        # Cross-Validation
        cv_scores = cross_validate(model, X_cv, y_train, k=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Final fit on full training set
        # Final fit on the FULL training set — cross_validate leaves the model
        # trained on only k-1 folds. We refit here to use all available data
        # before evaluating on the held-out test set.
        model.fit(X_tr, y_train)

        # Training AUC: shows overfitting when much higher than test AUC
        # Decision Tree intentionally left untuned to expose this gap
        y_train_prob = model.predict_proba(X_tr)[:, 1]
        train_auc = roc_auc_score(y_train, y_train_prob)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model": model,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

        print(f"\n{'─' * 40}")
        print(f"  {name}")
        print(f"  Train AUC : {train_auc:.4f}  (gap proves overfitting if >> CV AUC)")
        print(f"  CV AUC    : {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"  Test AUC  : {test_auc:.4f}")
        print(classification_report(y_test, y_pred))

    return results

def _pick_the_best_model(models: dict) -> Tuple[str, dict]:
    best_name = max(models, key=lambda k: models[k]["test_auc"])
    best_model = models[best_name]["model"]
    return best_name, best_model

if __name__ == "__main__":
    X, y, encoders = _clean_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Phase 5. Train & Compare Three Models
    models = {
        "Logistic Regression": LogisticRegression(learning_rate=0.01, epochs=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),     # Intentionally untuned — default max_depth=None causes overfitting
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        ),
        "Manual Bagging": ManualBaggingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }

    results = _train_all(models, X_train, X_test, y_train, y_test)

    # Phase 6. Pick the best baseline model
    best_name, best_model = _pick_the_best_model(results)
    print(f"\n✓ Best baseline model: {best_name}  (AUC {results[best_name]['test_auc']:.4f})")

    # Phase 7. Plots (Baseline)
    plot_all(models, results, X_train, X_test, y_test, best_name)

    # Phase 8. Hyperparameter Tuning
    baseline_cv = {name: results[name]["cv_mean"] for name in results}
    models = tune_all(X_train, y_train, baseline_cv)

    # Phase 9. Re-evaluate Tuned Models
    print("\n" + "═" * 50)
    print("  Phase 9 — Tuned Models Evaluation")
    print("═" * 50)

    results = _train_all(models, X_train, X_test, y_train, y_test)
    best_name, best_model = _pick_the_best_model(results)
    print(f"\n✓ Best tuned model: {best_name}  (AUC {results[best_name]['test_auc']:.4f})")
    plot_all(models, results, X_train, X_test, y_test, best_name)

    # Phase 7. Feature Importance (for the /predict API response)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        # Logistic Regression: use absolute coefficients as proxy
        importances = np.abs(best_model.coef_[0])
    elif hasattr(best_model, "trees"):
        # Manual Bagging: average feature importances across all trees
        importances = np.mean(
            [tree.feature_importances_ for tree in best_model.trees], axis=0
        )
    else:
        importances = np.zeros(len(X.columns))

    feature_importance = dict(zip(X.columns.tolist(), importances.tolist()))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 features:")
    for feat, score in top_features:
        print(f"  {feat}: {score:.4f}")

    # Phase 8. Save Everything the API Needs
    (BASE_DIR / "models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "model_name": best_name,
            "encoders": encoders,
            "feature_names": X.columns.tolist(),
            "feature_importance": feature_importance,
            "auc": results[best_name]["test_auc"],
        },
        BASE_DIR / "models" / "model.pkl"
    )

    print("\n✓ Saved to models/model.pkl — ready for the API.")
