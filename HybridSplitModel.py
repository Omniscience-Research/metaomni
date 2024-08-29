import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class Node:
    def __init__(self, depth=0):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.categorical_split = None
        self.prediction = None
        self.depth = depth

class ContinuousDiscreteHybridSplitModel(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.array([self._predict_single(x) for x in X])

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return self._create_leaf(y)

        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_categorical_split = None

        for feature in range(n_features):
            if self._is_categorical(X[:, feature]):
                gain, threshold = self._find_best_categorical_split(X[:, feature], y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_categorical_split = threshold
            else:
                gain, threshold = self._find_best_continuous_split(X[:, feature], y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain == -np.inf:
            return self._create_leaf(y)

        node = Node(depth)
        node.feature = best_feature

        if best_categorical_split is not None:
            node.categorical_split = best_categorical_split
            left_mask = np.isin(X[:, best_feature], best_categorical_split)
        else:
            node.threshold = best_threshold
            left_mask = X[:, best_feature] <= best_threshold

        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def _is_categorical(self, column):
        return column.dtype.kind in 'OUS'

    def _find_best_categorical_split(self, feature, y):
        unique_values = np.unique(feature)
        best_gain = -np.inf
        best_split = None

        for i in range(1, len(unique_values)):
            left_categories = unique_values[:i]
            gain = self._calculate_information_gain(feature, y, left_categories)
            if gain > best_gain:
                best_gain = gain
                best_split = left_categories

        return best_gain, best_split

    def _find_best_continuous_split(self, feature, y):
        sorted_idx = np.argsort(feature)
        sorted_feature, sorted_y = feature[sorted_idx], y[sorted_idx]

        best_gain = -np.inf
        best_threshold = None

        for i in range(1, len(sorted_feature)):
            if sorted_feature[i] != sorted_feature[i - 1]:
                threshold = (sorted_feature[i] + sorted_feature[i - 1]) / 2
                gain = self._calculate_information_gain(sorted_feature, sorted_y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold

        return best_gain, best_threshold

    def _calculate_information_gain(self, feature, y, split):
        if isinstance(split, (list, np.ndarray)):  # Categorical split
            left_mask = np.isin(feature, split)
        else:  # Continuous split
            left_mask = feature <= split

        left_y, right_y = y[left_mask], y[~left_mask]

        entropy_before = self._calculate_entropy(y)
        entropy_after = (len(left_y) / len(y)) * self._calculate_entropy(left_y) + \
                        (len(right_y) / len(y)) * self._calculate_entropy(right_y)

        return entropy_before - entropy_after

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def _create_leaf(self, y):
        node = Node()
        node.prediction = np.argmax(np.bincount(y))
        return node

    def _predict_single(self, x):
        node = self.tree_
        while node.left:
            if node.categorical_split is not None:
                if x[node.feature] in node.categorical_split:
                    node = node.left
                else:
                    node = node.right
            else:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node.prediction