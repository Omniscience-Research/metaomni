import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class BiasVarianceBalancedTree(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, 
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Create the ensemble of trees
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Create a decision tree with varying depth to balance bias and variance
            depth = np.random.randint(1, self.max_depth + 1) if self.max_depth else None
            
            tree = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
            
            # Bootstrap sampling if specified
            if self.bootstrap:
                n_samples = X.shape[0]
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
                tree.fit(X_bootstrap, y_bootstrap)
            else:
                tree.fit(X, y)
            
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions with each tree and take the majority vote
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        maj_vote = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), 
            axis=0, 
            arr=predictions
        )
        
        return self.classes_[maj_vote]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions with each tree
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        
        # Average the probabilities
        avg_proba = np.mean(all_proba, axis=0)
        
        return avg_proba