import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from core.cross_validation import cross_validate
from core.models.linear_model.logistic_regression import LogisticRegression
from core.models.ensemble.bagging import ManualBaggingClassifier

RANDOM_SEED = 42


def _sample_params(param_grid: dict, rng: np.random.RandomState) -> dict:
    """
    Randomly sample one combination from a parameter grid.

    Parameters
    ----------
    param_grid : dict
        Dictionary of parameter names to lists of candidate values.
    rng : np.random.RandomState
        Random state for reproducibility.

    Returns
    -------
    dict
        One sampled parameter combination.
    """
    return {k: rng.choice(v) for k, v in param_grid.items()}


def _randomized_search(model_class, param_grid: dict, X: np.ndarray,
                       y: np.ndarray, n_iter: int = 20, k: int = 5) -> tuple:
    """
    Randomized hyperparameter search using manual cross-validation.

    Parameters
    ----------
    model_class : class
        The model class to instantiate.
    param_grid : dict
        Dictionary of parameter names to lists of candidate values.
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels.
    n_iter : int
        Number of random parameter combinations to try.
    k : int
        Number of CV folds.

    Returns
    -------
    tuple
        Best parameters dict and best mean CV AUC.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    best_params = None
    best_auc = -np.inf

    for i in range(n_iter):
        params = _sample_params(param_grid, rng)
        model = model_class(random_state=RANDOM_SEED, **params)
        scores = cross_validate(model, X, y, k=k)
        mean_auc = scores.mean()
        print(f"  [{i + 1:02d}/{n_iter}] {params} → CV AUC: {mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    return best_params, best_auc


def _tune_logistic_regression(X: np.ndarray, y: np.ndarray,
                               n_iter: int = 20, k: int = 5) -> tuple:
    """
    Manual randomized search for LogisticRegression scratch.
    Scales data internally since it has no internal scaling.

    Parameters
    ----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels.
    n_iter : int
        Number of random parameter combinations to try.
    k : int
        Number of CV folds.

    Returns
    -------
    tuple
        Best parameters dict and best mean CV AUC.
    """
    param_grid = {
        "learning_rate": [0.001, 0.005, 0.01, 0.05],
        "epochs": [1000, 2000, 3000, 5000],
    }
    rng = np.random.RandomState(RANDOM_SEED)
    best_params = None
    best_auc = -np.inf

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for i in range(n_iter):
        params = _sample_params(param_grid, rng)
        model = LogisticRegression(**params)
        scores = cross_validate(model, X_scaled, y, k=k)
        mean_auc = scores.mean()
        print(f"  [{i + 1:02d}/{n_iter}] {params} → CV AUC: {mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    return best_params, best_auc


def tune_all(X_train, y_train, baseline_cv: dict) -> dict:
    """
    Tune all models and return rebuilt models dict with best parameters.
    Prints pre vs post tuning comparison table.

    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features.
    y_train : np.ndarray or pd.Series
        Training labels.
    baseline_cv : dict
        CV AUC results before tuning, keyed by model name.

    Returns
    -------
    dict
        Rebuilt models dict with tuned parameters.
    """
    if hasattr(X_train, 'values'):
        X_array = X_train.values
    else:
        X_array = X_train

    if hasattr(y_train, 'values'):
        y_array = y_train.values
    else:
        y_array = y_train

    print("\n" + "═" * 50)
    print("  Phase 6 — Hyperparameter Tuning")
    print("═" * 50)

    # ── Decision Tree ────────────────────────────────
    print("\nTuning Decision Tree...")
    dt_best_params, dt_best_auc = _randomized_search(
        DecisionTreeClassifier,
        {
            "max_depth":        [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf":  [1, 2, 4, 8],
        },
        X_array, y_array
    )
    print(f"  ✓ Best: {dt_best_params}  CV AUC: {dt_best_auc:.4f}")

    # ── Random Forest ────────────────────────────────
    print("\nTuning Random Forest...")
    rf_best_params, rf_best_auc = _randomized_search(
        RandomForestClassifier,
        {
            "n_estimators":     [100, 200, 300, 500],
            "max_depth":        [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        X_array, y_array
    )
    print(f"  ✓ Best: {rf_best_params}  CV AUC: {rf_best_auc:.4f}")

    # ── XGBoost ──────────────────────────────────────
    print("\nTuning XGBoost...")
    xgb_best_params, xgb_best_auc = _randomized_search(
        XGBClassifier,
        {
            "n_estimators":  [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth":     [3, 5, 7, 10],
        },
        X_array, y_array
    )
    print(f"  ✓ Best: {xgb_best_params}  CV AUC: {xgb_best_auc:.4f}")

    # ── Logistic Regression ──────────────────────────
    print("\nTuning Logistic Regression...")
    lr_best_params, lr_best_auc = _tune_logistic_regression(X_array, y_array, n_iter=30)
    print(f"  ✓ Best: {lr_best_params}  CV AUC: {lr_best_auc:.4f}")

    # ── Print pre vs post comparison ─────────────────
    after = {
        "Logistic Regression": lr_best_auc,
        "Decision Tree":       dt_best_auc,
        "Random Forest":       rf_best_auc,
        "XGBoost":             xgb_best_auc,
        "Manual Bagging":      baseline_cv.get("Manual Bagging", 0),
    }

    print("\n" + "═" * 60)
    print(f"  {'Model':<25} {'Before':>10} {'After':>10} {'Δ':>8}")
    print("═" * 60)
    for name in after:
        before = baseline_cv.get(name, 0)
        delta = after[name] - before
        print(f"  {name:<25} {before:>10.4f} {after[name]:>10.4f} {delta:>+8.4f}")
    print("═" * 60)

    # ── Rebuild models with best params ──────────────
    tuned_models = {
        "Logistic Regression": LogisticRegression(**lr_best_params),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_SEED, **dt_best_params),
        "Random Forest":       RandomForestClassifier(random_state=RANDOM_SEED, **rf_best_params),
        "XGBoost":             XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", **xgb_best_params),
        "Manual Bagging":      ManualBaggingClassifier(n_estimators=100, random_state=RANDOM_SEED),
    }

    return tuned_models
