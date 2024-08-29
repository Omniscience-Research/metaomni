import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class SimilarityWeightedQDA(BaseEstimator, ClassifierMixin):
    def __init__(self, similarity_metric='euclidean', similarity_sigma=1.0):
        self.similarity_metric = similarity_metric
        self.similarity_sigma = similarity_sigma

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Store the training data
        self.X_ = X
        self.y_ = y

        # Compute class-wise means and covariances
        self.means_ = []
        self.covariances_ = []
        self.priors_ = []

        for c in self.classes_:
            X_c = X[y == c]
            self.means_.append(np.mean(X_c, axis=0))
            self.covariances_.append(np.cov(X_c, rowvar=False))
            self.priors_.append(len(X_c) / len(X))

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute similarities between test points and training points
        similarities = np.exp(-cdist(X, self.X_, metric=self.similarity_metric) / 
                              (2 * self.similarity_sigma**2))

        # Normalize similarities
        similarities /= np.sum(similarities, axis=1, keepdims=True)

        # Compute weighted log-likelihood for each class
        log_likelihoods = []
        for c in range(self.n_classes_):
            class_mask = (self.y_ == self.classes_[c])
            class_similarities = similarities[:, class_mask]
            
            diff = X[:, np.newaxis, :] - self.means_[c]
            maha_dist = np.einsum('ijk,kl,ijl->ij', diff, np.linalg.inv(self.covariances_[c]), diff)
            
            weighted_log_likelihood = (
                -0.5 * np.sum(class_similarities * maha_dist, axis=1) -
                0.5 * np.log(np.linalg.det(self.covariances_[c])) +
                np.log(self.priors_[c])
            )
            log_likelihoods.append(weighted_log_likelihood)

        # Return the class with highest log-likelihood
        return self.classes_[np.argmax(log_likelihoods, axis=0)]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute similarities between test points and training points
        similarities = np.exp(-cdist(X, self.X_, metric=self.similarity_metric) / 
                              (2 * self.similarity_sigma**2))

        # Normalize similarities
        similarities /= np.sum(similarities, axis=1, keepdims=True)

        # Compute weighted log-likelihood for each class
        log_likelihoods = []
        for c in range(self.n_classes_):
            class_mask = (self.y_ == self.classes_[c])
            class_similarities = similarities[:, class_mask]
            
            diff = X[:, np.newaxis, :] - self.means_[c]
            maha_dist = np.einsum('ijk,kl,ijl->ij', diff, np.linalg.inv(self.covariances_[c]), diff)
            
            weighted_log_likelihood = (
                -0.5 * np.sum(class_similarities * maha_dist, axis=1) -
                0.5 * np.log(np.linalg.det(self.covariances_[c])) +
                np.log(self.priors_[c])
            )
            log_likelihoods.append(weighted_log_likelihood)

        # Convert log-likelihoods to probabilities
        log_likelihoods = np.array(log_likelihoods).T
        likelihoods = np.exp(log_likelihoods - np.max(log_likelihoods, axis=1, keepdims=True))
        probabilities = likelihoods / np.sum(likelihoods, axis=1, keepdims=True)

        return probabilities