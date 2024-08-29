import numpy as np
from scipy.special import i0
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DirectionalNaiveBayes(BaseEstimator, ClassifierMixin):
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
        self.class_prior_ = np.zeros(len(self.classes_))
        self.mu_ = np.zeros((len(self.classes_), self.n_features_))
        self.kappa_ = np.zeros((len(self.classes_), self.n_features_))
        
        # Fit the model for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_prior_[idx] = len(X_c) / len(X)
            
            for j in range(self.n_features_):
                angles = X_c[:, j]
                R = np.mean(np.exp(1j * angles))
                R_bar = np.abs(R)
                self.mu_[idx, j] = np.angle(R)
                self.kappa_[idx, j] = self._estimate_kappa(R_bar)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute log-likelihood for each class
        log_likelihood = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            class_ll = np.log(self.class_prior_[idx])
            for j in range(self.n_features_):
                class_ll += self._von_mises_log_pdf(X[:, j], self.mu_[idx, j], self.kappa_[idx, j])
            log_likelihood[:, idx] = class_ll

        # Return the class with highest log-likelihood
        return self.classes_[np.argmax(log_likelihood, axis=1)]

    def _von_mises_log_pdf(self, x, mu, kappa):
        return kappa * np.cos(x - mu) - np.log(2 * np.pi * i0(kappa))

    def _estimate_kappa(self, R_bar):
        if R_bar < 0.53:
            kappa = 2 * R_bar + R_bar**3 + 5 * R_bar**5 / 6
        elif R_bar < 0.85:
            kappa = -0.4 + 1.39 * R_bar + 0.43 / (1 - R_bar)
        else:
            kappa = 1 / (R_bar**3 - 4 * R_bar**2 + 3 * R_bar)
        return kappa