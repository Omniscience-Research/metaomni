import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier

class SimilaritySplitSelector(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Build the decision tree
        self.tree_ = self._build_tree(X, y, depth=0)
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['tree_'])

        # Input validation
        X = check_array(X)

        # Predict for each sample
        return np.array([self._predict_sample(sample) for sample in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (n_samples < self.min_samples_split) or \
           (n_samples < 2 * self.min_samples_leaf) or \
           (len(np.unique(y)) == 1):
            return {'leaf': True, 'class': stats.mode(y)[0][0]}

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return {'leaf': True, 'class': stats.mode(y)[0][0]}

        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Create child nodes
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_child,
            'right': right_child
        }

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_similarity = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)[1:-1]  # Exclude min and max

            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                similarity = self._calculate_similarity(feature_values[left_mask], feature_values[right_mask])

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_similarity(self, left_values, right_values):
        # Use Kolmogorov-Smirnov test as similarity measure
        statistic, _ = stats.ks_2samp(left_values, right_values)
        return -statistic  # Negative because lower K-S statistic means more similar distributions

    def _predict_sample(self, sample):
        node = self.tree_
        while not node['leaf']:
            if sample[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['class']