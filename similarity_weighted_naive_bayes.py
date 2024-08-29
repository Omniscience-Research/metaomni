import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import pairwise_distances

class SimilarityWeightedNB(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, similarity_metric='cosine'):
        self.alpha = alpha
        self.similarity_metric = similarity_metric

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, accept_sparse=True)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_ = X.shape[1]
        
        # Initialize feature probabilities and class priors
        self.feature_prob_ = np.zeros((len(self.classes_), self.n_features_))
        self.class_prior_ = np.zeros(len(self.classes_))
        
        # Compute class priors and feature probabilities
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_prior_[idx] = len(X_c) / len(X)
            self.feature_prob_[idx] = (X_c.sum(axis=0) + self.alpha) / (np.sum(X_c) + self.alpha * self.n_features_)
        
        # Ensure feature probabilities are in correct shape
        if issparse(X):
            self.feature_prob_ = np.array(self.feature_prob_.todense())
        
        self.X_train_ = X
        self.y_train_ = y
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_train_', 'y_train_', 'classes_', 'feature_prob_', 'class_prior_'])

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Compute similarity weights
        similarities = 1 - pairwise_distances(X, self.X_train_, metric=self.similarity_metric)
        
        # Compute weighted log-likelihood for each class
        weighted_ll = np.zeros((X.shape[0], len(self.classes_)))
        for idx, c in enumerate(self.classes_):
            class_mask = (self.y_train_ == c)
            class_similarities = similarities[:, class_mask]
            
            # Compute weighted feature probabilities
            weighted_feature_prob = np.average(self.feature_prob_[idx], 
                                               weights=class_similarities.sum(axis=1),
                                               axis=0)
            
            # Compute log-likelihood
            weighted_ll[:, idx] = np.sum(X * np.log(weighted_feature_prob) + 
                                         (1 - X) * np.log(1 - weighted_feature_prob), axis=1)
            weighted_ll[:, idx] += np.log(self.class_prior_[idx])

        # Return the class with highest log-likelihood
        return self.classes_[np.argmax(weighted_ll, axis=1)]