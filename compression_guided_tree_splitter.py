import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class CompressionGuidedTreeSplitter(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'class': np.argmax(np.bincount(y))}
        
        best_gain = -np.inf
        best_split = None
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self._compression_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, threshold)
        
        if best_split is None:
            return {'leaf': True, 'class': np.argmax(np.bincount(y))}
        
        feature, threshold = best_split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        return {
            'leaf': False,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _compression_gain(self, parent, left, right):
        def description_length(y):
            probs = np.bincount(y) / len(y)
            return entropy(probs, base=2) * len(y)
        
        parent_dl = description_length(parent)
        left_dl = description_length(left)
        right_dl = description_length(right)
        
        return parent_dl - (left_dl + right_dl)

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])