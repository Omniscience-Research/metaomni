import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy

class CompressionGuidedQDA(BaseEstimator, ClassifierMixin):
    def __init__(self, compression_ratio=0.8, n_components=None):
        self.compression_ratio = compression_ratio
        self.n_components = n_components

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Perform PCA for initial dimensionality reduction
        if self.n_components is None:
            self.n_components = int(self.n_features_ * self.compression_ratio)
        
        self.pca_ = PCA(n_components=self.n_components)
        X_compressed = self.pca_.fit_transform(X_scaled)

        # Compute class-wise statistics
        self.priors_ = []
        self.means_ = []
        self.covariances_ = []

        for idx, c in enumerate(self.classes_):
            X_c = X_compressed[y == c]
            self.priors_.append(X_c.shape[0] / X.shape[0])
            self.means_.append(np.mean(X_c, axis=0))
            self.covariances_.append(np.cov(X_c, rowvar=False))

        # Compute information gain for each feature
        self.feature_importance_ = self._compute_feature_importance(X_compressed, y)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # Standardize and compress input data
        X_scaled = self.scaler_.transform(X)
        X_compressed = self.pca_.transform(X_scaled)

        # Compute log-likelihood for each class
        log_likelihood = []
        for idx, c in enumerate(self.classes_):
            diff = X_compressed - self.means_[idx]
            log_like = (
                -0.5 * np.sum(np.dot(diff, linalg.inv(self.covariances_[idx])) * diff, axis=1)
                - 0.5 * np.log(linalg.det(self.covariances_[idx]))
                + np.log(self.priors_[idx])
            )
            log_likelihood.append(log_like)

        log_likelihood = np.array(log_likelihood).T
        return self.classes_[np.argmax(log_likelihood, axis=1)]

    def _compute_feature_importance(self, X, y):
        feature_importance = np.zeros(X.shape[1])
        
        for feature in range(X.shape[1]):
            feature_entropy = entropy(X[:, feature])
            conditional_entropy = 0
            
            for c in self.classes_:
                X_c = X[y == c, feature]
                conditional_entropy += np.sum(entropy(X_c)) * (len(X_c) / len(X))
            
            information_gain = feature_entropy - conditional_entropy
            feature_importance[feature] = information_gain

        return feature_importance / np.sum(feature_importance)