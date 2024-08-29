import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder

class EntropyRegularizedQDA(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Use LabelEncoder to convert class labels to integers
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Compute class priors
        self.priors_ = np.bincount(y_encoded) / len(y)
        
        # Compute mean and covariance for each class
        self.means_ = []
        self.covariances_ = []
        
        for i in range(self.n_classes_):
            X_class = X[y_encoded == i]
            self.means_.append(np.mean(X_class, axis=0))
            self.covariances_.append(np.cov(X_class, rowvar=False))
        
        # Compute entropy regularization term
        self.entropy_reg_ = self._compute_entropy_regularization(X, y_encoded)
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute log-likelihood for each class
        log_likelihoods = []
        for i in range(self.n_classes_):
            mvn = multivariate_normal(mean=self.means_[i], cov=self.covariances_[i])
            log_likelihood = mvn.logpdf(X) + np.log(self.priors_[i])
            log_likelihoods.append(log_likelihood)

        # Add entropy regularization term
        log_likelihoods = np.array(log_likelihoods).T - self.alpha * self.entropy_reg_

        # Return the class with highest log-likelihood
        return self.classes_[np.argmax(log_likelihoods, axis=1)]

    def _compute_entropy_regularization(self, X, y):
        entropy_reg = np.zeros(self.n_features_)
        
        for j in range(self.n_features_):
            feature_values = X[:, j]
            
            for i in range(self.n_classes_):
                class_values = feature_values[y == i]
                
                # Compute histogram
                hist, _ = np.histogram(class_values, bins='auto', density=True)
                
                # Compute entropy
                entropy = -np.sum(hist * np.log(hist + 1e-10))
                
                entropy_reg[j] += entropy
        
        return entropy_reg / self.n_classes_