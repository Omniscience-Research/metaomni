import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityWeightedKNNWithFeatureSets(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, feature_sets=None, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.feature_sets = feature_sets
        self.weights = weights

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the feature sets if not provided
        if self.feature_sets is None:
            self.feature_sets = [list(range(X.shape[1]))]
        
        # Store the training data
        self.X_ = X
        self.y_ = y
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute similarities for each feature set
        similarities = []
        for feature_set in self.feature_sets:
            sim = cosine_similarity(X[:, feature_set], self.X_[:, feature_set])
            similarities.append(sim)

        # Combine similarities
        combined_similarities = np.mean(similarities, axis=0)

        # Get the k nearest neighbors
        nearest_neighbor_indices = np.argsort(combined_similarities, axis=1)[:, -self.n_neighbors:]

        # Predict for each test sample
        y_pred = []
        for i, indices in enumerate(nearest_neighbor_indices):
            if self.weights == 'uniform':
                weights = np.ones(self.n_neighbors)
            elif self.weights == 'distance':
                weights = combined_similarities[i, indices]
            else:
                raise ValueError("weights should be 'uniform' or 'distance'")

            neighbor_labels = self.y_[indices]
            class_weights = np.zeros(len(self.classes_))
            for label, weight in zip(neighbor_labels, weights):
                class_weights[np.where(self.classes_ == label)] += weight

            y_pred.append(self.classes_[np.argmax(class_weights)])

        return np.array(y_pred)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute similarities for each feature set
        similarities = []
        for feature_set in self.feature_sets:
            sim = cosine_similarity(X[:, feature_set], self.X_[:, feature_set])
            similarities.append(sim)

        # Combine similarities
        combined_similarities = np.mean(similarities, axis=0)

        # Get the k nearest neighbors
        nearest_neighbor_indices = np.argsort(combined_similarities, axis=1)[:, -self.n_neighbors:]

        # Compute probabilities for each test sample
        probas = []
        for i, indices in enumerate(nearest_neighbor_indices):
            if self.weights == 'uniform':
                weights = np.ones(self.n_neighbors)
            elif self.weights == 'distance':
                weights = combined_similarities[i, indices]
            else:
                raise ValueError("weights should be 'uniform' or 'distance'")

            neighbor_labels = self.y_[indices]
            class_weights = np.zeros(len(self.classes_))
            for label, weight in zip(neighbor_labels, weights):
                class_weights[np.where(self.classes_ == label)] += weight

            probas.append(class_weights / np.sum(class_weights))

        return np.array(probas)