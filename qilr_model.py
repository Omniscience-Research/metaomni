import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class QuantumInspiredLogisticRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, n_qubits=5, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.n_qubits = n_qubits
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _quantum_inspired_feature_map(self, X):
        """
        Quantum-inspired feature map using sine and cosine functions.
        """
        mapped_features = []
        for i in range(self.n_qubits):
            angle = np.pi * X / (2 ** (i + 1))
            mapped_features.append(np.sin(angle))
            mapped_features.append(np.cos(angle))
        return np.column_stack(mapped_features)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, theta, X, y):
        z = np.dot(X, theta)
        h = self._sigmoid(z)
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def _gradient(self, theta, X, y):
        z = np.dot(X, theta)
        h = self._sigmoid(z)
        return np.dot(X.T, (h - y)) / len(y)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("QuantumInspiredLogisticRegressor only supports binary classification.")

        self.X_ = X
        self.y_ = y

        n_features = X.shape[1]
        X_quantum = self._quantum_inspired_feature_map(X)
        n_quantum_features = X_quantum.shape[1]

        initial_theta = np.zeros(n_quantum_features)

        res = minimize(
            fun=self._loss,
            x0=initial_theta,
            args=(X_quantum, y),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )

        self.coef_ = res.x
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_quantum = self._quantum_inspired_feature_map(X)
        z = np.dot(X_quantum, self.coef_)
        proba = self._sigmoid(z)
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)