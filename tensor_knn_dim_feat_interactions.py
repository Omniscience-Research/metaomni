import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class TensorKNNInteractions(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, interaction_degree=2, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.interaction_degree = interaction_degree
        self.weights = weights

    def _generate_interactions(self, X):
        X_tensor = tf.constant(X, dtype=tf.float32)
        interactions = [X_tensor]
        
        for degree in range(2, self.interaction_degree + 1):
            combinations = tf.experimental.numpy.triu_indices(X.shape[1], k=1, m=degree)
            interaction = tf.reduce_prod(tf.gather(X_tensor, combinations, axis=1), axis=2)
            interactions.append(interaction)
        
        return tf.concat(interactions, axis=1)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Generate feature interactions
        self.X_train_interactions_ = self._generate_interactions(X)
        self.y_train_ = tf.constant(y, dtype=tf.int32)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Generate feature interactions for test data
        X_test_interactions = self._generate_interactions(X)
        
        # Compute distances
        distances = tf.reduce_sum(tf.square(tf.expand_dims(X_test_interactions, 1) - 
                                            tf.expand_dims(self.X_train_interactions_, 0)), axis=2)
        
        # Find k-nearest neighbors
        _, indices = tf.nn.top_k(-distances, k=self.n_neighbors)
        
        # Get labels of k-nearest neighbors
        neighbor_labels = tf.gather(self.y_train_, indices)
        
        if self.weights == 'uniform':
            # Majority voting
            predictions = tf.mode(neighbor_labels, axis=1).values
        elif self.weights == 'distance':
            # Distance-weighted voting
            weights = 1.0 / (tf.gather_nd(distances, tf.stack([tf.range(tf.shape(indices)[0])[:, None], indices], axis=2)) + 1e-8)
            weighted_votes = tf.one_hot(neighbor_labels, depth=len(self.classes_)) * weights[..., None]
            predictions = tf.argmax(tf.reduce_sum(weighted_votes, axis=1), axis=1)
        
        return self.classes_[predictions.numpy()]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Generate feature interactions for test data
        X_test_interactions = self._generate_interactions(X)
        
        # Compute distances
        distances = tf.reduce_sum(tf.square(tf.expand_dims(X_test_interactions, 1) - 
                                            tf.expand_dims(self.X_train_interactions_, 0)), axis=2)
        
        # Find k-nearest neighbors
        _, indices = tf.nn.top_k(-distances, k=self.n_neighbors)
        
        # Get labels of k-nearest neighbors
        neighbor_labels = tf.gather(self.y_train_, indices)
        
        if self.weights == 'uniform':
            # Uniform weights
            weights = tf.ones_like(neighbor_labels, dtype=tf.float32)
        elif self.weights == 'distance':
            # Distance-weighted voting
            weights = 1.0 / (tf.gather_nd(distances, tf.stack([tf.range(tf.shape(indices)[0])[:, None], indices], axis=2)) + 1e-8)
        
        # Compute class probabilities
        weighted_votes = tf.one_hot(neighbor_labels, depth=len(self.classes_)) * weights[..., None]
        probabilities = tf.reduce_sum(weighted_votes, axis=1) / tf.reduce_sum(weights, axis=1)[:, None]
        
        return probabilities.numpy()