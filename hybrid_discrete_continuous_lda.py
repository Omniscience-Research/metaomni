import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder

class HybridDiscreteContinuousLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, discrete_features=None, continuous_features=None):
        self.discrete_features = discrete_features
        self.continuous_features = continuous_features

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Encode labels
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        
        # Determine discrete and continuous features if not specified
        if self.discrete_features is None and self.continuous_features is None:
            self.discrete_features_ = np.where(np.all(X.astype(int) == X, axis=0))[0]
            self.continuous_features_ = np.setdiff1d(np.arange(X.shape[1]), self.discrete_features_)
        else:
            self.discrete_features_ = self.discrete_features
            self.continuous_features_ = self.continuous_features
        
        # Separate discrete and continuous features
        X_discrete = X[:, self.discrete_features_]
        X_continuous = X[:, self.continuous_features_]
        
        # Compute class priors
        self.class_priors_ = np.bincount(y_encoded) / len(y)
        
        # Compute means and covariances for continuous features
        self.means_ = np.array([np.mean(X_continuous[y_encoded == c], axis=0) for c in range(self.n_classes_)])
        self.covariances_ = np.array([np.cov(X_continuous[y_encoded == c].T) for c in range(self.n_classes_)])
        
        # Compute probabilities for discrete features
        self.discrete_probs_ = []
        for feature in range(X_discrete.shape[1]):
            feature_probs = []
            for c in range(self.n_classes_):
                class_data = X_discrete[y_encoded == c, feature]
                unique, counts = np.unique(class_data, return_counts=True)
                probs = dict(zip(unique, counts / len(class_data)))
                feature_probs.append(probs)
            self.discrete_probs_.append(feature_probs)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Separate discrete and continuous features
        X_discrete = X[:, self.discrete_features_]
        X_continuous = X[:, self.continuous_features_]
        
        # Compute log-likelihood for each class
        log_likelihood = np.zeros((X.shape[0], self.n_classes_))
        
        for c in range(self.n_classes_):
            # Continuous part
            diff = X_continuous - self.means_[c]
            log_likelihood[:, c] = -0.5 * np.sum(np.dot(diff, np.linalg.inv(self.covariances_[c])) * diff, axis=1)
            log_likelihood[:, c] -= 0.5 * np.log(np.linalg.det(self.covariances_[c]))
            
            # Discrete part
            for feature in range(X_discrete.shape[1]):
                probs = self.discrete_probs_[feature][c]
                feature_values = X_discrete[:, feature]
                log_likelihood[:, c] += np.log([probs.get(val, 1e-10) for val in feature_values])
            
            # Add log prior
            log_likelihood[:, c] += np.log(self.class_priors_[c])
        
        # Return the class with highest log-likelihood
        return self.le_.inverse_transform(np.argmax(log_likelihood, axis=1))

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Compute log-likelihood (same as in predict method)
        log_likelihood = self._compute_log_likelihood(X)
        
        # Compute probabilities
        proba = np.exp(log_likelihood - np.max(log_likelihood, axis=1)[:, np.newaxis])
        proba /= np.sum(proba, axis=1)[:, np.newaxis]
        
        return proba

    def _compute_log_likelihood(self, X):
        X_discrete = X[:, self.discrete_features_]
        X_continuous = X[:, self.continuous_features_]
        
        log_likelihood = np.zeros((X.shape[0], self.n_classes_))
        
        for c in range(self.n_classes_):
            # Continuous part
            diff = X_continuous - self.means_[c]
            log_likelihood[:, c] = -0.5 * np.sum(np.dot(diff, np.linalg.inv(self.covariances_[c])) * diff, axis=1)
            log_likelihood[:, c] -= 0.5 * np.log(np.linalg.det(self.covariances_[c]))
            
            # Discrete part
            for feature in range(X_discrete.shape[1]):
                probs = self.discrete_probs_[feature][c]
                feature_values = X_discrete[:, feature]
                log_likelihood[:, c] += np.log([probs.get(val, 1e-10) for val in feature_values])
            
            # Add log prior
            log_likelihood[:, c] += np.log(self.class_priors_[c])
        
        return log_likelihood