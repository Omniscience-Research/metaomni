import numpy as np
from typing import List, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class Node:
    def __init__(self, depth: int = 0):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.depth = depth
        self.value = None

class FeatureInteractionTreeModel(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_features: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        if self.max_features is None:
            self.max_features = self.n_features_
        
        self.tree = self._grow_tree(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.array([self._predict_single(x) for x in X])

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        node = Node(depth)
        
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            node.value = np.argmax(np.bincount(y))
            return node
        
        best_gain = -np.inf
        best_split = None
        
        feature_combinations = self._get_feature_combinations()
        
        for combo in feature_combinations:
            split = self._find_best_split(X, y, combo)
            if split['gain'] > best_gain:
                best_gain = split['gain']
                best_split = split
        
        if best_split is None or best_split['gain'] <= 0:
            node.value = np.argmax(np.bincount(y))
            return node
        
        node.feature = best_split['feature']
        node.threshold = best_split['threshold']
        
        left_mask = X[:, node.feature] <= node.threshold
        right_mask = ~left_mask
        
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node

    def _get_feature_combinations(self) -> List[Tuple[int, ...]]:
        single_features = list(range(self.n_features_))
        feature_pairs = [(i, j) for i in range(self.n_features_) for j in range(i+1, self.n_features_)]
        return single_features + feature_pairs

    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature: Tuple[int, ...]) -> dict:
        if isinstance(feature, int):
            feature_values = X[:, feature]
        else:
            feature_values = X[:, feature[0]] * X[:, feature[1]]  # Interaction term
        
        thresholds = np.unique(feature_values)
        
        best_gain = -np.inf
        best_threshold = None
        
        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                continue
            
            gain = self._calculate_gain(y, y[left_mask], y[right_mask])
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return {
            'feature': feature,
            'threshold': best_threshold,
            'gain': best_gain
        }

    def _calculate_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        def gini(x):
            _, counts = np.unique(x, return_counts=True)
            probabilities = counts / len(x)
            return 1 - np.sum(probabilities ** 2)
        
        n = len(parent)
        n_left, n_right = len(left), len(right)
        
        return gini(parent) - (n_left / n) * gini(left) - (n_right / n) * gini(right)

    def _predict_single(self, x: np.ndarray) -> int:
        node = self.tree
        while node.left:
            if isinstance(node.feature, int):
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                interaction_value = x[node.feature[0]] * x[node.feature[1]]
                if interaction_value <= node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node.value