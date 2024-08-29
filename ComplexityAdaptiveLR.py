import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class ComplexityAdaptiveLRModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_min_samples_split=2, max_depth=None, random_state=None):
        self.base_min_samples_split = base_min_samples_split
        self.max_depth = max_depth
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Create the base decision tree
        self.tree_ = DecisionTreeClassifier(
            min_samples_split=self.base_min_samples_split,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        # Fit the base tree
        self.tree_.fit(X, y)
        
        # Adjust the learning rate (min_samples_split) based on local complexity
        self._adjust_learning_rate(X, y)
        
        return self
    
    def _adjust_learning_rate(self, X, y):
        # Get the leaf indices for each sample
        leaf_indices = self.tree_.apply(X)
        
        # Calculate the complexity of each leaf
        leaf_complexities = {}
        for leaf_idx in np.unique(leaf_indices):
            leaf_mask = (leaf_indices == leaf_idx)
            leaf_X = X[leaf_mask]
            leaf_y = y[leaf_mask]
            
            # Calculate complexity based on the variance of the target variable
            complexity = np.var(leaf_y)
            leaf_complexities[leaf_idx] = complexity
        
        # Adjust min_samples_split for each leaf based on its complexity
        self.leaf_min_samples_split_ = {}
        for leaf_idx, complexity in leaf_complexities.items():
            # Increase min_samples_split for more complex regions
            adjusted_min_samples_split = max(
                self.base_min_samples_split,
                int(self.base_min_samples_split * (1 + complexity))
            )
            self.leaf_min_samples_split_[leaf_idx] = adjusted_min_samples_split
        
        # Retrain the tree with adjusted min_samples_split
        self.tree_ = DecisionTreeClassifier(
            min_samples_split=self.base_min_samples_split,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.tree_.fit(X, y)
        
        # Apply the adjusted min_samples_split to each node
        self._apply_adjusted_min_samples_split(self.tree_.tree_, 0)
    
    def _apply_adjusted_min_samples_split(self, tree, node_id):
        if tree.feature[node_id] != -2:  # Not a leaf node
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            # Get the leaf indices for the samples in this node
            node_leaf_indices = self.tree_.apply(self.tree_.tree_.value[node_id])
            
            # Calculate the average min_samples_split for the leaves in this node
            avg_min_samples_split = np.mean([
                self.leaf_min_samples_split_[leaf_idx]
                for leaf_idx in np.unique(node_leaf_indices)
            ])
            
            # Apply the adjusted min_samples_split
            tree.min_samples_split[node_id] = int(avg_min_samples_split)
            
            # Recursively apply to child nodes
            self._apply_adjusted_min_samples_split(tree, left_child)
            self._apply_adjusted_min_samples_split(tree, right_child)
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        return self.tree_.predict(X)