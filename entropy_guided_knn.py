import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class EntropyGuidedKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, entropy_weight=0.5):
        self.n_neighbors = n_neighbors
        self.entropy_weight = entropy_weight

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the training data and labels
        self.X_ = X
        self.y_ = y
        
        # Calculate entropy for each feature
        self.feature_entropy_ = np.apply_along_axis(self._calculate_entropy, 0, X)
        
        # Calculate the entropy-weighted distances
        self.weighted_X_ = X * (1 - self.entropy_weight * self.feature_entropy_)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Calculate entropy-weighted distances
        weighted_X = X * (1 - self.entropy_weight * self.feature_entropy_)
        
        # Compute distances between test samples and training samples
        distances = euclidean_distances(weighted_X, self.weighted_X_)
        
        # Find k nearest neighbors
        nearest_neighbor_indices = distances.argsort()[:, :self.n_neighbors]
        
        # Get the labels of the k nearest neighbors
        nearest_neighbor_labels = self.y_[nearest_neighbor_indices]
        
        # Predict the class by majority vote
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=1, 
            arr=nearest_neighbor_labels
        )
        
        return self.classes_[predictions]

    def _calculate_entropy(self, x):
        # Normalize the data
        x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
        
        # Calculate histogram
        hist, _ = np.histogram(x_normalized, bins=20, density=True)
        
        # Calculate entropy
        return entropy(hist)