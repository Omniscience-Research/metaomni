import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DimAwarePerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iterations=1000, dim_threshold=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.dim_threshold = dim_threshold

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

        # Compute feature importance (using variance as a simple metric)
        feature_importance = np.var(X, axis=0)
        
        # Identify important dimensions
        self.important_dims_ = feature_importance > self.dim_threshold * np.max(feature_importance)

        # Train the perceptron
        for _ in range(self.n_iterations):
            for xi, yi in zip(X, y):
                # Focus on important dimensions
                xi_important = xi[self.important_dims_]
                weights_important = self.weights_[self.important_dims_]

                # Compute prediction
                y_pred = np.dot(xi_important, weights_important) + self.bias_

                # Update weights and bias
                if yi * y_pred <= 0:
                    self.weights_[self.important_dims_] += self.learning_rate * yi * xi_important
                    self.bias_ += self.learning_rate * yi

        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Check if X has the correct number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")

        # Make predictions
        X_important = X[:, self.important_dims_]
        weights_important = self.weights_[self.important_dims_]
        y_pred = np.dot(X_important, weights_important) + self.bias_
        
        return np.where(y_pred >= 0, self.classes_[1], self.classes_[0])

    def decision_function(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Check if X has the correct number of features
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fitting.")

        # Compute decision function
        X_important = X[:, self.important_dims_]
        weights_important = self.weights_[self.important_dims_]
        return np.dot(X_important, weights_important) + self.bias_