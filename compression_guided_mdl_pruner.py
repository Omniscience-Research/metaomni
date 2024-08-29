import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MDLGuidedPruner(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Fit a decision tree
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        self.tree_.fit(X, y)
        
        # Prune the tree using MDL principle
        self._mdl_prune(self.tree_.tree_, X, y)
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['tree_'])
        
        # Input validation
        X = check_array(X)
        
        return self.tree_.predict(X)

    def _mdl_prune(self, tree, X, y, node_id=0):
        if tree.feature[node_id] == -2:  # Leaf node
            return self._calculate_mdl_leaf(y)
        
        left_mask = X[:, tree.feature[node_id]] <= tree.threshold[node_id]
        right_mask = ~left_mask
        
        left_mdl = self._mdl_prune(tree, X[left_mask], y[left_mask], tree.children_left[node_id])
        right_mdl = self._mdl_prune(tree, X[right_mask], y[right_mask], tree.children_right[node_id])
        
        subtree_mdl = left_mdl + right_mdl + self._calculate_mdl_split(tree, node_id)
        leaf_mdl = self._calculate_mdl_leaf(y)
        
        if leaf_mdl <= subtree_mdl:
            # Prune the subtree
            tree.children_left[node_id] = -1
            tree.children_right[node_id] = -1
            tree.feature[node_id] = -2
            return leaf_mdl
        
        return subtree_mdl

    def _calculate_mdl_leaf(self, y):
        n = len(y)
        if n == 0:
            return 0
        
        class_counts = np.bincount(y)
        probs = class_counts[class_counts > 0] / n
        
        # Calculate the MDL for a leaf node
        return -n * np.sum(probs * np.log2(probs))  # Entropy

    def _calculate_mdl_split(self, tree, node_id):
        # Calculate the MDL cost of encoding the split
        return np.log2(tree.n_features)  # Cost of encoding the feature index