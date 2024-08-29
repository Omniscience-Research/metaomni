import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances

class FuzzyKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, m=2):
        self.n_neighbors = n_neighbors
        self.m = m  # Fuzzy strength parameter

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the training data
        self.X_ = X
        self.y_ = y
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Calculate distances between X and the training data
        distances = euclidean_distances(X, self.X_)
        
        # Sort the distances and get indices of the k-nearest neighbors
        nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        # Initialize the output array
        y_pred = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, neighbors in enumerate(nearest_neighbor_indices):
            for neighbor in neighbors:
                # Calculate the fuzzy membership
                distance = distances[i, neighbor]
                if distance == 0:
                    y_pred[i, :] = 0
                    y_pred[i, self.y_[neighbor]] = 1
                    break
                membership = 1 / (distance ** (2 / (self.m - 1)))
                
                # Update the fuzzy votes
                y_pred[i, self.y_[neighbor]] += membership
        
        # Normalize the fuzzy votes
        y_pred /= np.sum(y_pred, axis=1, keepdims=True)
        
        # Return the class with the highest fuzzy vote
        return self.classes_[np.argmax(y_pred, axis=1)]

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Calculate distances between X and the training data
        distances = euclidean_distances(X, self.X_)
        
        # Sort the distances and get indices of the k-nearest neighbors
        nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        # Initialize the output array
        y_pred = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, neighbors in enumerate(nearest_neighbor_indices):
            for neighbor in neighbors:
                # Calculate the fuzzy membership
                distance = distances[i, neighbor]
                if distance == 0:
                    y_pred[i, :] = 0
                    y_pred[i, self.y_[neighbor]] = 1
                    break
                membership = 1 / (distance ** (2 / (self.m - 1)))
                
                # Update the fuzzy votes
                y_pred[i, self.y_[neighbor]] += membership
        
        # Normalize the fuzzy votes
        y_pred /= np.sum(y_pred, axis=1, keepdims=True)
        
        return y_pred