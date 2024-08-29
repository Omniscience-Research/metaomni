import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class HybridContinuousDiscreteBooster(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, hidden_layer_sizes=(10,)):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.hidden_layer_sizes = hidden_layer_sizes
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Initialize estimators
        self.estimators_ = []
        
        # Initialize predictions
        F = np.zeros((X.shape[0], len(self.classes_)))
        
        for _ in range(self.n_estimators):
            # Compute negative gradient
            neg_gradient = y - self._sigmoid(F)
            
            # Fit decision tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, neg_gradient)
            
            # Fit neural network
            nn = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=1000)
            nn.fit(X, neg_gradient)
            
            # Update predictions
            tree_pred = tree.predict(X)
            nn_pred = nn.predict(X)
            F += self.learning_rate * (tree_pred[:, np.newaxis] + nn_pred[:, np.newaxis]) / 2
            
            # Store estimators
            self.estimators_.append((tree, nn))
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Initialize predictions
        F = np.zeros((X.shape[0], len(self.classes_)))
        
        # Iterate through all estimators
        for tree, nn in self.estimators_:
            tree_pred = tree.predict(X)
            nn_pred = nn.predict(X)
            F += self.learning_rate * (tree_pred[:, np.newaxis] + nn_pred[:, np.newaxis]) / 2
        
        # Return class labels
        return self.classes_[np.argmax(F, axis=1)]
    
    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Initialize predictions
        F = np.zeros((X.shape[0], len(self.classes_)))
        
        # Iterate through all estimators
        for tree, nn in self.estimators_:
            tree_pred = tree.predict(X)
            nn_pred = nn.predict(X)
            F += self.learning_rate * (tree_pred[:, np.newaxis] + nn_pred[:, np.newaxis]) / 2
        
        # Return probabilities
        return self._sigmoid(F)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))