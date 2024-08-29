import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DirectionalNode:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DirectionalTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.array([self._predict_single(x) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels == 1:
            return DirectionalNode(value=np.argmax(np.bincount(y)))

        best_gain = -1
        best_split = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain, split = self._directional_split(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_idx, threshold, split)

        if best_gain > 0:
            feature_idx, threshold, (left_idx, right_idx) = best_split
            left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
            right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
            return DirectionalNode(feature_idx=feature_idx, threshold=threshold, left=left, right=right)
        else:
            return DirectionalNode(value=np.argmax(np.bincount(y)))

    def _directional_split(self, X, y, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0, ([], [])

        left_y = y[left_mask]
        right_y = y[right_mask]

        # Calculate the directional information gain
        parent_entropy = self._entropy(y)
        left_entropy = self._entropy(left_y)
        right_entropy = self._entropy(right_y)

        # Incorporate directional information
        left_direction = np.mean(X[left_mask, feature_idx])
        right_direction = np.mean(X[right_mask, feature_idx])
        direction_factor = np.abs(left_direction - right_direction) / (np.max(X[:, feature_idx]) - np.min(X[:, feature_idx]))

        information_gain = parent_entropy - (len(left_y) / len(y) * left_entropy + len(right_y) / len(y) * right_entropy)
        directional_gain = information_gain * direction_factor

        return directional_gain, (left_mask, right_mask)

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _predict_single(self, x):
        node = self.tree_
        while node.left:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value