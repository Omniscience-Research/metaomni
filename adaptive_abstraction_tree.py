import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

class AdaptiveAbstractionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Create the root node
        self.tree_ = self._build_tree(X, y, depth=0)
        
        return self
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf or
            np.all(y == y[0])):
            return Node(y=y)
        
        # Determine the level of abstraction based on local data distribution
        abstraction_level = self._determine_abstraction_level(X, y)
        
        if abstraction_level == 'detailed':
            # Use a decision tree for detailed splitting
            clf = DecisionTreeClassifier(max_depth=1, 
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features,
                                         random_state=self.random_state)
            clf.fit(X, y)
            
            feature = clf.tree_.feature[0]
            threshold = clf.tree_.threshold[0]
            
            mask = X[:, feature] <= threshold
            
            left = self._build_tree(X[mask], y[mask], depth + 1)
            right = self._build_tree(X[~mask], y[~mask], depth + 1)
            
            return Node(feature=feature, threshold=threshold, left=left, right=right)
        
        else:  # abstraction_level == 'abstract'
            # Use K-means clustering for abstract splitting
            kmeans = KMeans(n_clusters=2, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(X)
            
            mask = cluster_labels == 0
            
            left = self._build_tree(X[mask], y[mask], depth + 1)
            right = self._build_tree(X[~mask], y[~mask], depth + 1)
            
            return Node(kmeans=kmeans, left=left, right=right)
    
    def _determine_abstraction_level(self, X, y):
        # This is a simple heuristic. You might want to use a more sophisticated method.
        if len(np.unique(y)) > 1 and X.shape[0] > 100:
            return 'detailed'
        else:
            return 'abstract'
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        return np.array([self._traverse_tree(x, self.tree_) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return np.argmax(np.bincount(node.y))
        
        if hasattr(node, 'feature'):
            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            cluster = node.kmeans.predict([x])[0]
            if cluster == 0:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

class Node:
    def __init__(self, feature=None, threshold=None, kmeans=None, left=None, right=None, y=None):
        self.feature = feature
        self.threshold = threshold
        self.kmeans = kmeans
        self.left = left
        self.right = right
        self.y = y
    
    def is_leaf(self):
        return self.y is not None