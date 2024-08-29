import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances

class DirectionalKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the training data
        self.X_ = X
        self.y_ = y
        
        # Calculate the mean of the training data
        self.mean_ = np.mean(X, axis=0)
        
        # Calculate the directions from mean to each training point
        self.directions_ = X - self.mean_
        
        # Normalize the directions
        self.directions_ /= np.linalg.norm(self.directions_, axis=1)[:, np.newaxis]
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Calculate directions from mean to test points
        test_directions = X - self.mean_
        test_directions /= np.linalg.norm(test_directions, axis=1)[:, np.newaxis]

        # Calculate cosine similarities
        similarities = np.dot(test_directions, self.directions_.T)

        # Get indices of k nearest neighbors
        neigh_ind = similarities.argsort()[:, -self.n_neighbors:]

        # Get labels of these neighbors
        neigh_labels = self.y_[neigh_ind]

        if self.weights == 'uniform':
            # Majority voting
            return np.array([np.argmax(np.bincount(x)) for x in neigh_labels])
        elif self.weights == 'distance':
            # Distance-weighted voting
            weights = similarities[np.arange(len(X))[:, np.newaxis], neigh_ind]
            weights = np.clip(weights, 0, None)  # Clip negative values to 0
            sum_weights = np.sum(weights, axis=1)
            sum_weights[sum_weights == 0] = 1  # Avoid division by zero
            weights /= sum_weights[:, np.newaxis]
            
            weighted_votes = np.zeros((X.shape[0], len(self.classes_)))
            for i, class_label in enumerate(self.classes_):
                mask = neigh_labels == class_label
                weighted_votes[:, i] = np.sum(weights * mask, axis=1)
            
            return self.classes_[np.argmax(weighted_votes, axis=1)]
        else:
            raise ValueError("Unsupported weight option. Use 'uniform' or 'distance'.")