import numpy as np
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42


def cross_validate(model, X, y, k: int = 5) -> np.ndarray:
    """
    Manually perform k-fold cross-validation.

    Parameters
    ----------
    model : object
        A model instance with fit and predict_proba methods.
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : pd.Series or np.ndarray
        Binary target labels of shape (n_samples,).
    k : int
        Number of folds. Default is 5.

    Returns
    -------
    np.ndarray
        AUC score per fold of shape (k,).
    """
    np.random.seed(RANDOM_SEED)

    n_samples = len(y)
    indices = np.random.permutation(n_samples)

    fold_size = n_samples // k
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(k)]

    auc_scores = []

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        # Use iloc for pandas DataFrames, index directly for numpy arrays
        if hasattr(X, "iloc"):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
        else:
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]

        if hasattr(y, "iloc"):
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
        else:
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_prob = model.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_prob)
        auc_scores.append(auc)

    return np.array(auc_scores)