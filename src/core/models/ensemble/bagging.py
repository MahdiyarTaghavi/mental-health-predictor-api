import numpy as np
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


class ManualBaggingClassifier:
    """
    Bagging ensemble implemented from scratch using DecisionTreeClassifier as base learner.

    Parameters
    ----------
    n_estimators : int
        Number of trees to train.
    max_samples : float
        Fraction of training samples to draw for each bootstrap sample.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100, max_samples: float = 1.0, random_state: int = RANDOM_SEED):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees = []

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """
        Draw a random bootstrap sample with replacement.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).
        y : np.ndarray
            Target labels of shape (n_samples,).

        Returns
        -------
        tuple of np.ndarray
            Bootstrapped X and y.
        """
        n_samples = int(len(y) * self.max_samples)
        indices = np.random.choice(len(y), size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ManualBaggingClassifier":
        """
        Train n_estimators decision trees on bootstrap samples.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).
        y : np.ndarray
            Target labels of shape (n_samples,).

        Returns
        -------
        self
        """
        # Convert to numpy — bootstrap sampling uses integer indexing
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        np.random.seed(self.random_state)
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(random_state=self.random_state)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Average probability predictions across all trees.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Averaged probabilities of shape (n_samples, 2).
        """
        if hasattr(X, 'values'):
            X = X.values
        all_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        return all_probs.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels via majority vote.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Binary predictions of shape (n_samples,).
        """
        if hasattr(X, 'values'):
            X = X.values
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        # majority vote: round the mean of 0s and 1s
        return (all_preds.mean(axis=0) >= 0.5).astype(int)