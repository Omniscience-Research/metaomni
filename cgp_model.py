import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler

class CompressionGuidedPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iterations=1000, compression_rate=0.5, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.compression_rate = compression_rate
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        if len(self.classes_) != 2:
            raise ValueError("CompressionGuidedPerceptron only supports binary classification.")
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize weights and bias
        rng = np.random.RandomState(self.random_state)
        self.weights_ = rng.randn(self.n_features_in_)
        self.bias_ = rng.randn(1)
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Perform compression-guided training
        for _ in range(self.n_iterations):
            # Compute predictions
            y_pred = self._predict_raw(X_scaled)
            
            # Compute error
            error = y - y_pred
            
            # Update weights and bias
            self.weights_ += self.learning_rate * np.dot(X_scaled.T, error)
            self.bias_ += self.learning_rate * np.sum(error)
            
            # Apply compression
            self._compress_weights()
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Standardize features
        X_scaled = self.scaler_.transform(X)
        
        # Compute raw predictions
        y_pred_raw = self._predict_raw(X_scaled)
        
        # Convert to class labels
        return np.where(y_pred_raw >= 0, self.classes_[1], self.classes_[0])

    def _predict_raw(self, X):
        return np.dot(X, self.weights_) + self.bias_

    def _compress_weights(self):
        # Sort weights by absolute value
        sorted_indices = np.argsort(np.abs(self.weights_))
        
        # Determine number of weights to keep
        n_keep = int(self.n_features_in_ * self.compression_rate)
        
        # Zero out the smallest weights
        self.weights_[sorted_indices[:self.n_features_in_ - n_keep]] = 0