import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.naive_bayes import GaussianNB, MultinomialNB

class AbstractionSwitchingNB(BaseEstimator, ClassifierMixin):
    def __init__(self, abstraction_threshold=0.5):
        self.abstraction_threshold = abstraction_threshold
        self.gaussian_nb = GaussianNB()
        self.multinomial_nb = MultinomialNB()
        self.is_fitted_ = False

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Determine which features to use for each classifier
        self.feature_variance = np.var(X, axis=0)
        self.gaussian_features = self.feature_variance >= self.abstraction_threshold
        self.multinomial_features = ~self.gaussian_features
        
        # Fit Gaussian NB on high-variance features
        if np.any(self.gaussian_features):
            self.gaussian_nb.fit(X[:, self.gaussian_features], y)
        
        # Fit Multinomial NB on low-variance features
        if np.any(self.multinomial_features):
            self.multinomial_nb.fit(X[:, self.multinomial_features], y)
        
        # Return the classifier
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted_'])

        # Input validation
        X = check_array(X)

        # Make predictions using both classifiers
        gaussian_pred = np.zeros((X.shape[0], len(self.classes_)))
        multinomial_pred = np.zeros((X.shape[0], len(self.classes_)))
        
        if np.any(self.gaussian_features):
            gaussian_pred = self.gaussian_nb.predict_proba(X[:, self.gaussian_features])
        
        if np.any(self.multinomial_features):
            multinomial_pred = self.multinomial_nb.predict_proba(X[:, self.multinomial_features])
        
        # Combine predictions
        combined_pred = gaussian_pred + multinomial_pred
        
        # Return class labels
        return self.classes_[np.argmax(combined_pred, axis=1)]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['is_fitted_'])

        # Input validation
        X = check_array(X)

        # Make predictions using both classifiers
        gaussian_pred = np.zeros((X.shape[0], len(self.classes_)))
        multinomial_pred = np.zeros((X.shape[0], len(self.classes_)))
        
        if np.any(self.gaussian_features):
            gaussian_pred = self.gaussian_nb.predict_proba(X[:, self.gaussian_features])
        
        if np.any(self.multinomial_features):
            multinomial_pred = self.multinomial_nb.predict_proba(X[:, self.multinomial_features])
        
        # Combine predictions
        combined_pred = gaussian_pred + multinomial_pred
        
        # Normalize probabilities
        return combined_pred / np.sum(combined_pred, axis=1, keepdims=True)