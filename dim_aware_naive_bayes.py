import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DimAwareNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_ = X.shape[1]
        
        # Initialize parameters
        self.class_priors_ = {}
        self.class_means_ = {}
        self.class_covariances_ = {}
        
        for c in self.classes_:
            X_c = X[y == c]
            
            # Compute class priors
            self.class_priors_[c] = (len(X_c) + self.alpha) / (len(X) + len(self.classes_) * self.alpha)
            
            # Compute class means
            self.class_means_[c] = np.mean(X_c, axis=0)
            
            # Compute class covariances
            cov = np.cov(X_c, rowvar=False)
            
            # Handle singularity by adding a small value to the diagonal
            cov += np.eye(cov.shape[0]) * 1e-6
            
            self.class_covariances_[c] = cov
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Check if the input has the correct number of features
        if X.shape[1] != self.n_features_:
            raise ValueError("The number of features in predict is different from the number of features in fit.")

        # Compute probabilities for each class
        probas = np.array([self._class_log_probability(X, c) for c in self.classes_]).T
        
        # Return the class with highest probability
        return self.classes_[np.argmax(probas, axis=1)]

    def _class_log_probability(self, X, c):
        log_prior = np.log(self.class_priors_[c])
        log_likelihood = multivariate_normal.logpdf(X, mean=self.class_means_[c], cov=self.class_covariances_[c])
        return log_prior + log_likelihood