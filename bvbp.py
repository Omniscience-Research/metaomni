import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class BiasVarianceBalancingPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=0.01, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize weights and bias
        rng = np.random.RandomState(self.random_state)
        self.weights_ = rng.randn(self.n_features_in_)
        self.bias_ = rng.randn()
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X, self.weights_) + self.bias_
            predictions = self._sigmoid(linear_output)
            
            # Compute gradients
            d_weights = (1 / len(y)) * np.dot(X.T, (predictions - y)) + self.lambda_reg * self.weights_
            d_bias = (1 / len(y)) * np.sum(predictions - y)
            
            # Update weights and bias
            self.weights_ -= self.learning_rate * d_weights
            self.bias_ -= self.learning_rate * d_bias
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions
        linear_output = np.dot(X, self.weights_) + self.bias_
        probabilities = self._sigmoid(linear_output)
        return (probabilities >= 0.5).astype(int)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))