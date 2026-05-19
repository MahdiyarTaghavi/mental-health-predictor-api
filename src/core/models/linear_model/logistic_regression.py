import numpy as np


class LogisticRegression:
    """
    Logistic Regression implemented from scratch using gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates.
    epochs : int
        Number of full passes over the training data.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        Parameters
        ----------
        z : np.ndarray
            Linear combination of inputs and weights.

        Returns
        -------
        np.ndarray
            Values squashed to (0, 1).
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features).
        y : np.ndarray
            Binary target labels of shape (n_samples,).

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            # Forward pass
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            # Binary cross-entropy loss
            loss = -np.mean(
                y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9)
            )
            self.loss_history.append(loss)

            # Gradients
            error = y_hat - y
            dw = (X.T @ error) / n_samples
            db = np.mean(error)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, 2) with probabilities for each class.
        """
        prob_positive = self._sigmoid(X @ self.weights + self.bias)
        return np.column_stack([1 - prob_positive, prob_positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels by thresholding at 0.5.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Binary predictions of shape (n_samples,).
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)