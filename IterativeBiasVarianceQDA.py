import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class IterativeBiasVarianceQDA(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iterations=10, cv=5, tol=1e-4):
        self.max_iterations = max_iterations
        self.cv = cv
        self.tol = tol
        self.best_model = None
        self.best_score = -np.inf
        self.complexity_param = 0.0

    def fit(self, X, y):
        n_features = X.shape[1]
        
        for iteration in range(self.max_iterations):
            # Increase complexity gradually
            self.complexity_param = iteration / (self.max_iterations - 1)
            
            # Create QDA model with current complexity
            current_model = QuadraticDiscriminantAnalysis(
                reg_param=1 - self.complexity_param,
                store_covariance=True
            )
            
            # Perform cross-validation
            scores = cross_val_score(current_model, X, y, cv=self.cv)
            mean_score = np.mean(scores)
            
            # Check if the current model is better
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_model = current_model
            
            # Check for convergence
            if iteration > 0 and (mean_score - prev_score) < self.tol:
                break
            
            prev_score = mean_score
        
        # Fit the best model on the entire dataset
        self.best_model.fit(X, y)
        return self

    def predict(self, X):
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.best_model.predict_proba(X)

    def score(self, X, y):
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.best_model.score(X, y)