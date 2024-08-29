import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

class DimAwareBranchingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None, min_impurity_decrease=0.0,
                 min_dim_ratio=0.1, max_branching_factor=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_dim_ratio = min_dim_ratio
        self.max_branching_factor = max_branching_factor

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Create the root node
        self.tree_ = self._build_tree(X, y, depth=0)
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Ensure X has the same number of features as during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Number of features in X does not match training data.")

        # Traverse the tree for each sample
        return np.array([self._traverse_tree(x, self.tree_) for x in X])

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (depth == self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'leaf': True, 'class': np.argmax(np.bincount(y))}
        
        # Compute local dimensionality
        pca = PCA().fit(X)
        local_dim = np.sum(np.cumsum(pca.explained_variance_ratio_) < (1 - self.min_dim_ratio))
        
        # Adjust branching factor based on local dimensionality
        branching_factor = min(max(2, int(local_dim)), self.max_branching_factor)
        
        # Create decision tree with adjusted max_leaf_nodes
        dt = DecisionTreeClassifier(
            max_depth=1,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            min_impurity_decrease=self.min_impurity_decrease,
            max_leaf_nodes=branching_factor
        )
        dt.fit(X, y)
        
        # Create node
        node = {
            'leaf': False,
            'feature': dt.tree_.feature[0],
            'threshold': dt.tree_.threshold[0],
            'children': []
        }
        
        # Recursively build child nodes
        for i in range(dt.tree_.n_outputs):
            mask = X[:, node['feature']] <= node['threshold']
            node['children'].append(self._build_tree(X[mask], y[mask], depth + 1))
            node['children'].append(self._build_tree(X[~mask], y[~mask], depth + 1))
        
        return node

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['children'][0])
        else:
            return self._traverse_tree(x, node['children'][1])