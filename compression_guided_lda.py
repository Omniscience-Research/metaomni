import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

class CompressionGuidedLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, alpha=1.0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Initialize parameters
        self.class_priors_ = np.zeros(self.n_classes_)
        self.means_ = np.zeros((self.n_classes_, self.n_features_))
        self.covariances_ = np.zeros((self.n_classes_, self.n_features_, self.n_features_))

        for k in range(self.n_classes_):
            X_k = X[y_encoded == k]
            self.class_priors_[k] = len(X_k) / len(X)
            self.means_[k] = np.mean(X_k, axis=0)
            self.covariances_[k] = np.cov(X_k.T) + np.eye(self.n_features_) * self.alpha

        # Compression-guided optimization
        for _ in range(self.max_iter):
            old_means = self.means_.copy()
            old_covariances = self.covariances_.copy()

            # E-step: Compute responsibilities
            responsibilities = self._compute_responsibilities(X)

            # M-step: Update parameters
            for k in range(self.n_classes_):
                resp_k = responsibilities[:, k]
                self.means_[k] = np.sum(resp_k[:, np.newaxis] * X, axis=0) / np.sum(resp_k)
                diff = X - self.means_[k]
                self.covariances_[k] = np.dot(resp_k * diff.T, diff) / np.sum(resp_k)
                self.covariances_[k] += np.eye(self.n_features_) * self.alpha

            # Check for convergence
            if np.allclose(old_means, self.means_, atol=self.tol) and \
               np.allclose(old_covariances, self.covariances_, atol=self.tol):
                break

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        log_probs = self._compute_log_probs(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def _compute_responsibilities(self, X):
        log_probs = self._compute_log_probs(X)
        log_resp = log_probs - logsumexp(log_probs, axis=1)[:, np.newaxis]
        return np.exp(log_resp)

    def _compute_log_probs(self, X):
        log_probs = np.zeros((X.shape[0], self.n_classes_))
        for k in range(self.n_classes_):
            log_probs[:, k] = np.log(self.class_priors_[k]) + \
                multivariate_normal.logpdf(X, mean=self.means_[k], cov=self.covariances_[k])
        return log_probs

    def score(self, X, y):
        return np.mean(self.predict(X) == y)