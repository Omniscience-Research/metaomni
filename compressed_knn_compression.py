import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import euclidean_distances

class CompressedKNNCompressor(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, compression_ratio=0.1):
        self.n_neighbors = n_neighbors
        self.compression_ratio = compression_ratio

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Compress the training data
        n_compressed = int(X.shape[0] * self.compression_ratio)
        compressed_indices = np.random.choice(X.shape[0], n_compressed, replace=False)
        
        self.X_compressed_ = X[compressed_indices]
        self.y_compressed_ = y[compressed_indices]
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Check that the input features match the training data
        if X.shape[1] != self.n_features_in_:
            raise ValueError("The number of features in X does not match the number of features learned during fitting.")

        # Compute distances between X and compressed training data
        distances = euclidean_distances(X, self.X_compressed_)
        
        # Find k-nearest neighbors
        nearest_neighbor_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        
        # Predict the class for each sample
        y_pred = np.zeros(X.shape[0], dtype=self.classes_.dtype)
        for i, neighbors in enumerate(nearest_neighbor_indices):
            neighbor_classes = self.y_compressed_[neighbors]
            y_pred[i] = np.argmax(np.bincount(neighbor_classes))
        
        return y_pred

    def score(self, X, y):
        # Predict and calculate accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)