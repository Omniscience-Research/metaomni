import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class AdaptiveBiasVarianceSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C_range=np.logspace(-3, 3, 7), gamma_range=np.logspace(-3, 3, 7), cv=5):
        self.C_range = C_range
        self.gamma_range = gamma_range
        self.cv = cv
        self.best_C = None
        self.best_gamma = None
        self.svm = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        best_score = -np.inf
        for C in self.C_range:
            for gamma in self.gamma_range:
                svm = SVC(C=C, gamma=gamma, kernel='rbf')
                scores = cross_val_score(svm, X, y, cv=self.cv)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_C = C
                    self.best_gamma = gamma
        
        # Train the final model with the best parameters
        self.svm = SVC(C=self.best_C, gamma=self.best_gamma, kernel='rbf')
        self.svm.fit(X, y)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        return self.svm.predict(X)

    def decision_function(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        return self.svm.decision_function(X)

    def score(self, X, y):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X, y = check_X_y(X, y)
        
        return self.svm.score(X, y)