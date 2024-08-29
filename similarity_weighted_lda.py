import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler

class SimilarityWeightedLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, similarity_metric='euclidean', alpha=1.0):
        self.similarity_metric = similarity_metric
        self.alpha = alpha

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Compute class means and covariance matrices
        self.class_means_ = []
        self.class_covs_ = []
        
        for c in self.classes_:
            X_c = X_scaled[y == c]
            self.class_means_.append(np.mean(X_c, axis=0))
            self.class_covs_.append(np.cov(X_c, rowvar=False))
        
        # Compute pooled covariance matrix
        self.pooled_cov_ = np.sum(self.class_covs_, axis=0) / len(self.classes_)
        
        # Compute inverse of pooled covariance matrix
        self.pooled_cov_inv_ = np.linalg.inv(self.pooled_cov_)
        
        # Store training data for similarity computation
        self.X_train_ = X_scaled
        self.y_train_ = y
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Standardize features
        X_scaled = self.scaler_.transform(X)
        
        # Compute similarities
        similarities = 1 / (1 + cdist(X_scaled, self.X_train_, metric=self.similarity_metric))
        
        # Compute weighted means and covariances
        weighted_means = []
        weighted_covs = []
        
        for c in self.classes_:
            weights = similarities[:, self.y_train_ == c]
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            
            X_c = self.X_train_[self.y_train_ == c]
            weighted_mean = np.sum(weights[:, :, np.newaxis] * X_c, axis=1)
            weighted_means.append(weighted_mean)
            
            centered_X_c = X_c - weighted_mean[:, np.newaxis, :]
            weighted_cov = np.sum(weights[:, :, np.newaxis, np.newaxis] * 
                                  np.einsum('ijk,ijl->ijkl', centered_X_c, centered_X_c), axis=1)
            weighted_covs.append(weighted_cov)
        
        # Compute discriminant scores
        scores = []
        
        for i in range(len(self.classes_)):
            diff = X_scaled - weighted_means[i]
            mahalanobis = np.sum(diff @ self.pooled_cov_inv_ * diff, axis=1)
            log_det = np.log(np.linalg.det(weighted_covs[i] + self.alpha * np.eye(self.n_features_in_)))
            score = -0.5 * (mahalanobis + log_det)
            scores.append(score)
        
        # Return class with highest score
        return self.classes_[np.argmax(scores, axis=0)]