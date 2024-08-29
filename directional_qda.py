import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DirectionalQuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, n_directions=2, reg_param=1e-4):
        self.n_directions = n_directions
        self.reg_param = reg_param

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Calculate class priors
        self.priors_ = np.bincount(y) / len(y)

        # Calculate class means
        self.means_ = np.array([np.mean(X[y == c], axis=0) for c in self.classes_])

        # Calculate directional covariance matrices for each class
        self.covs_ = []
        for c in self.classes_:
            X_c = X[y == c]
            centered_X = X_c - self.means_[c]
            
            # Compute the covariance matrix
            cov = np.cov(centered_X.T)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = eigh(cov)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Select top n_directions
            top_eigenvalues = eigenvalues[:self.n_directions]
            top_eigenvectors = eigenvectors[:, :self.n_directions]
            
            # Construct directional covariance matrix
            directional_cov = np.dot(top_eigenvectors * top_eigenvalues, top_eigenvectors.T)
            
            # Add regularization
            directional_cov += np.eye(self.n_features_) * self.reg_param
            
            self.covs_.append(directional_cov)

        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute log-likelihood for each class
        log_likelihoods = []
        for i, c in enumerate(self.classes_):
            diff = X - self.means_[i]
            log_det = np.log(np.linalg.det(self.covs_[i]))
            inv_cov = np.linalg.inv(self.covs_[i])
            mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            log_likelihood = -0.5 * (log_det + mahalanobis) + np.log(self.priors_[i])
            log_likelihoods.append(log_likelihood)

        # Return the class with highest log-likelihood
        return self.classes_[np.argmax(log_likelihoods, axis=0)]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute log-likelihood for each class
        log_likelihoods = []
        for i, c in enumerate(self.classes_):
            diff = X - self.means_[i]
            log_det = np.log(np.linalg.det(self.covs_[i]))
            inv_cov = np.linalg.inv(self.covs_[i])
            mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            log_likelihood = -0.5 * (log_det + mahalanobis) + np.log(self.priors_[i])
            log_likelihoods.append(log_likelihood)

        # Convert log-likelihoods to probabilities
        log_likelihoods = np.array(log_likelihoods).T
        log_prob_sum = logsumexp(log_likelihoods, axis=1)
        probabilities = np.exp(log_likelihoods - log_prob_sum[:, np.newaxis])

        return probabilities

def logsumexp(a, axis=None):
    a_max = np.amax(a, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max