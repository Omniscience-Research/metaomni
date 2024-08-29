import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class RandomnessAdaptivePerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        if len(self.classes_) != 2:
            raise ValueError("RandomnessAdaptivePerceptron only supports binary classification.")
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize weights and bias
        self.rng_ = np.random.RandomState(self.random_state)
        self.w_ = self.rng_.randn(self.n_features_in_)
        self.b_ = self.rng_.randn()
        
        # Initialize randomness factor
        self.randomness_ = 1.0
        
        # Training loop
        for _ in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                # Make prediction
                y_pred = self._predict_instance(xi)
                
                # Update weights if prediction is incorrect
                if y_pred != yi:
                    update = self.learning_rate * (yi - y_pred)
                    self.w_ += update * xi * self.randomness_
                    self.b_ += update * self.randomness_
                    errors += 1
            
            # Adjust randomness factor based on error rate
            error_rate = errors / len(y)
            self.randomness_ = max(0.1, min(1.0, 1.0 - error_rate))
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions
        return np.array([self._predict_instance(xi) for xi in X])

    def _predict_instance(self, x):
        activation = np.dot(x, self.w_) + self.b_
        return 1 if activation >= 0 else 0