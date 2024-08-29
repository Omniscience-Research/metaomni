import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class ComplexityPenalizedGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2,
                 min_samples_leaf=1, subsample=1.0, max_features=None, complexity_penalty=0.01):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.complexity_penalty = complexity_penalty
        
    def _initialize_trees(self):
        self.trees_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            self.trees_.append(tree)
    
    def _compute_complexity_penalty(self, tree):
        # Compute the complexity penalty based on the number of nodes in the tree
        return self.complexity_penalty * tree.tree_.node_count
    
    def _negative_gradient(self, y, pred):
        return y - 1 / (1 + np.exp(-pred))
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        # Convert y to {0, 1}
        y = np.where(y == self.classes_[1], 1, 0)
        
        self._initialize_trees()
        
        # Initialize predictions
        F = np.zeros(len(y))
        
        for i, tree in enumerate(self.trees_):
            residuals = self._negative_gradient(y, F)
            
            # Subsample the data if subsample < 1.0
            if self.subsample < 1.0:
                sample_mask = np.random.rand(len(y)) < self.subsample
                X_subset = X[sample_mask]
                residuals_subset = residuals[sample_mask]
            else:
                X_subset = X
                residuals_subset = residuals
            
            # Fit the tree to the residuals
            tree.fit(X_subset, residuals_subset)
            
            # Update predictions
            update = self.learning_rate * tree.predict(X)
            
            # Apply complexity penalty
            complexity_penalty = self._compute_complexity_penalty(tree)
            update -= complexity_penalty
            
            F += update
        
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        # Compute raw predictions
        raw_predictions = np.zeros(X.shape[0])
        for tree in self.trees_:
            raw_predictions += self.learning_rate * tree.predict(X)
        
        # Apply sigmoid function to get probabilities
        proba = 1 / (1 + np.exp(-raw_predictions))
        
        # Return probabilities for both classes
        return np.vstack((1 - proba, proba)).T
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]