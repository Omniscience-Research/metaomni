import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class AdaptiveAbstractionPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iterations=1000, learning_rate=0.01, abstraction_threshold=0.1):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.abstraction_threshold = abstraction_threshold

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize weights and bias
        self.weights_ = np.zeros(self.n_features_in_)
        self.bias_ = 0
        
        # Initialize abstraction levels
        self.abstraction_levels_ = np.ones(self.n_features_in_)
        
        # Perform training
        for _ in range(self.n_iterations):
            for xi, yi in zip(X, y):
                # Make prediction
                y_pred = self._predict_instance(xi)
                
                # Update weights and bias
                error = yi - y_pred
                self.weights_ += self.learning_rate * error * xi * self.abstraction_levels_
                self.bias_ += self.learning_rate * error
                
                # Update abstraction levels
                self.abstraction_levels_ += self.learning_rate * error * np.abs(xi * self.weights_)
                self.abstraction_levels_ = np.clip(self.abstraction_levels_, self.abstraction_threshold, 1)
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions
        return np.array([self._predict_instance(xi) for xi in X])

    def _predict_instance(self, x):
        activation = np.dot(x * self.abstraction_levels_, self.weights_) + self.bias_
        return 1 if activation >= 0 else 0