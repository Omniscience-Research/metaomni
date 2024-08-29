import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder

class CompressRegNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, lambda_reg=0.1):
        self.alpha = alpha  # Laplace smoothing parameter
        self.lambda_reg = lambda_reg  # Compression regularization parameter

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Initialize parameters
        self.class_priors_ = np.zeros(self.n_classes_)
        self.feature_log_prob_ = np.zeros((self.n_classes_, self.n_features_))

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Calculate class priors and feature probabilities
        for c in range(self.n_classes_):
            X_c = X[y_encoded == c]
            self.class_priors_[c] = len(X_c) / len(X)
            
            # Calculate feature probabilities with Laplace smoothing
            feature_counts = X_c.sum(axis=0) + self.alpha
            total_counts = feature_counts.sum()
            
            # Apply compression-based regularization
            feature_probs = feature_counts / total_counts
            regularized_probs = self._compress_regularize(feature_probs)
            
            self.feature_log_prob_[c] = np.log(regularized_probs)

        return self

    def _compress_regularize(self, probs):
        # Implement compression-based regularization
        H = entropy(probs)  # Calculate entropy
        regularized_probs = probs + self.lambda_reg * (1/len(probs) - probs) * H
        return regularized_probs / regularized_probs.sum()  # Normalize

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Calculate log-likelihood for each class
        jll = (self.feature_log_prob_[:, np.newaxis, :] * X).sum(axis=2)
        jll += np.log(self.class_priors_)[:, np.newaxis]

        # Return predictions
        return self.classes_[np.argmax(jll, axis=0)]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Calculate log-likelihood for each class
        jll = (self.feature_log_prob_[:, np.newaxis, :] * X).sum(axis=2)
        jll += np.log(self.class_priors_)[:, np.newaxis]

        # Compute probabilities
        log_prob_x = logsumexp(jll, axis=0)
        return np.exp(jll - log_prob_x)

def logsumexp(arr, axis=0):
    """Compute the log of the sum of exponentials of input elements."""
    arr_max = np.max(arr, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(arr - arr_max), axis=axis, keepdims=True)) + arr_max