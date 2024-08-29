import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class BiasVarianceBalancedTreeEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, min_depth=1, max_depth=10, cv=5, random_state=None):
        self.n_estimators = n_estimators
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.cv = cv
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        self.estimators_ = []
        self.estimator_depths_ = []
        
        for i in range(self.n_estimators):
            best_depth = None
            best_score = -np.inf
            
            for depth in range(self.min_depth, self.max_depth + 1):
                clf = DecisionTreeClassifier(max_depth=depth, random_state=self.random_state)
                scores = cross_val_score(clf, X, y, cv=self.cv)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_depth = depth
            
            # Train the best tree and add it to the ensemble
            best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=self.random_state)
            best_tree.fit(X, y)
            self.estimators_.append(best_tree)
            self.estimator_depths_.append(best_depth)
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Input validation
        X = check_array(X)
        
        # Collect predictions from all estimators
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Majority voting
        maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        
        # Return prediction
        return self.classes_[maj]
    
    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Input validation
        X = check_array(X)
        
        # Collect probability predictions from all estimators
        all_proba = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        
        # Average probabilities across estimators
        avg_proba = np.mean(all_proba, axis=0)
        
        return avg_proba