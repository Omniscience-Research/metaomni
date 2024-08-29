import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB

class RandomnessAdaptiveNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, var_smoothing=1e-9):
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_ = X.shape[1]
        
        # Initialize the Gaussian Naive Bayes classifier
        self.gnb_ = GaussianNB(var_smoothing=self.var_smoothing)
        self.gnb_.fit(X, y)
        
        # Calculate feature importance based on mutual information
        self.feature_importance_ = self._calculate_feature_importance(X, y)
        
        # Calculate randomness scores for each feature
        self.randomness_scores_ = self._calculate_randomness_scores(X)
        
        # Calculate adaptive weights
        self.adaptive_weights_ = self._calculate_adaptive_weights()
        
        # Store X and y for prediction
        self.X_ = X
        self.y_ = y
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions using the Gaussian Naive Bayes classifier
        predictions = self.gnb_.predict(X)
        
        # Apply randomness adaptation
        for i in range(X.shape[0]):
            sample = X[i]
            randomness_adjusted_proba = self._randomness_adjusted_probability(sample)
            predictions[i] = self.classes_[np.argmax(randomness_adjusted_proba)]
        
        return predictions

    def _calculate_feature_importance(self, X, y):
        feature_importance = np.zeros(self.n_features_)
        for i in range(self.n_features_):
            feature_importance[i] = self._mutual_information(X[:, i], y)
        return feature_importance / np.sum(feature_importance)

    def _mutual_information(self, x, y):
        # Calculate mutual information between feature x and target y
        c_xy = np.histogram2d(x, y, bins=20)[0]
        mi = entropy(np.sum(c_xy, axis=0)) + entropy(np.sum(c_xy, axis=1)) - entropy(c_xy.flatten())
        return mi

    def _calculate_randomness_scores(self, X):
        randomness_scores = np.zeros(self.n_features_)
        for i in range(self.n_features_):
            randomness_scores[i] = entropy(X[:, i])
        return randomness_scores / np.sum(randomness_scores)

    def _calculate_adaptive_weights(self):
        return (1 - self.randomness_scores_) * self.feature_importance_

    def _randomness_adjusted_probability(self, sample):
        class_probs = self.gnb_.predict_proba([sample])[0]
        adjusted_probs = np.zeros_like(class_probs)
        
        for i, c in enumerate(self.classes_):
            mask = (self.y_ == c)
            class_samples = self.X_[mask]
            
            distances = np.sum(((class_samples - sample) * self.adaptive_weights_) ** 2, axis=1)
            similarity = np.exp(-distances / self.alpha)
            
            adjusted_probs[i] = class_probs[i] * np.mean(similarity)
        
        return adjusted_probs / np.sum(adjusted_probs)