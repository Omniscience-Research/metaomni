import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class FractalPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

    def _fractal_activation(self, x):
        return 1 / (1 + np.exp(-x))

    def _fractal_derivative(self, x):
        return x * (1 - x)

    def _initialize_weights(self, n_features):
        np.random.seed(self.random_state)
        self.weights_ = [np.random.randn(n_features + 1) for _ in range(self.max_depth)]

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        if len(self.classes_) != 2:
            raise ValueError("FractalPerceptron only supports binary classification.")

        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        X_bias = np.column_stack([np.ones(n_samples), X])

        for _ in range(self.n_iterations):
            # Forward pass
            layer_outputs = [X_bias]
            for depth in range(self.max_depth):
                z = np.dot(layer_outputs[-1], self.weights_[depth])
                a = self._fractal_activation(z)
                layer_outputs.append(a)

            # Backward pass
            error = y.reshape(-1, 1) - layer_outputs[-1]
            deltas = [error * self._fractal_derivative(layer_outputs[-1])]

            for depth in range(self.max_depth - 1, 0, -1):
                delta = np.dot(deltas[0], self.weights_[depth].T) * self._fractal_derivative(layer_outputs[depth])
                deltas.insert(0, delta)

            # Update weights
            for depth in range(self.max_depth):
                self.weights_[depth] += self.learning_rate * np.dot(layer_outputs[depth].T, deltas[depth])

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        X_bias = np.column_stack([np.ones(n_samples), X])

        # Forward pass
        layer_output = X_bias
        for depth in range(self.max_depth):
            z = np.dot(layer_output, self.weights_[depth])
            layer_output = self._fractal_activation(z)

        # Convert probabilities to class labels
        y_pred = (layer_output >= 0.5).astype(int)
        return self.classes_[y_pred.ravel()]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        X_bias = np.column_stack([np.ones(n_samples), X])

        # Forward pass
        layer_output = X_bias
        for depth in range(self.max_depth):
            z = np.dot(layer_output, self.weights_[depth])
            layer_output = self._fractal_activation(z)

        # Return probabilities for both classes
        return np.column_stack([1 - layer_output, layer_output])