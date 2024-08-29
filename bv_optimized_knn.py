import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class BVOptimizedKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k_range=range(1, 31), n_iterations=10, test_size=0.3, random_state=None):
        self.k_range = k_range
        self.n_iterations = n_iterations
        self.test_size = test_size
        self.random_state = random_state
        self.best_k = None
        self.best_model = None

    def fit(self, X, y):
        best_score = -np.inf
        best_k = None
        best_model = None

        for k in self.k_range:
            bias_scores = []
            variance_scores = []

            for _ in range(self.n_iterations):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state
                )

                model = KNeighborsClassifier(n_neighbors=k)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)

                bias_scores.append(1 - accuracy)
                variance_scores.append(1 - np.mean(y_pred == model.predict(X_val)))

            bias = np.mean(bias_scores)
            variance = np.mean(variance_scores)
            total_error = bias + variance

            if -total_error > best_score:
                best_score = -total_error
                best_k = k
                best_model = KNeighborsClassifier(n_neighbors=k)
                best_model.fit(X, y)

        self.best_k = best_k
        self.best_model = best_model
        return self

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' before making predictions.")
        return self.best_model.predict(X)

    def get_params(self, deep=True):
        return {
            "k_range": self.k_range,
            "n_iterations": self.n_iterations,
            "test_size": self.test_size,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self