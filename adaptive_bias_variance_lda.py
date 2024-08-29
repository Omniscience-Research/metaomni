import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

class AdaptiveBiasVarianceLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=None, shrinkage='auto', solver='svd', 
                 store_covariance=False, tol=1e-4, cv=5):
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.solver = solver
        self.store_covariance = store_covariance
        self.tol = tol
        self.cv = cv
        self.lda = None
        self.best_alpha = None

    def fit(self, X_train, y_train):
        # Define a range of alpha values to try
        alphas = np.logspace(-4, 0, 50)
        
        best_score = -np.inf
        best_alpha = None
        
        # Perform cross-validation to find the best alpha
        for alpha in alphas:
            lda = LinearDiscriminantAnalysis(
                n_components=self.n_components,
                shrinkage=alpha,
                solver=self.solver,
                store_covariance=self.store_covariance,
                tol=self.tol
            )
            scores = cross_val_score(lda, X_train, y_train, cv=self.cv)
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
        
        # Train the final model with the best alpha
        self.best_alpha = best_alpha
        self.lda = LinearDiscriminantAnalysis(
            n_components=self.n_components,
            shrinkage=self.best_alpha,
            solver=self.solver,
            store_covariance=self.store_covariance,
            tol=self.tol
        )
        self.lda.fit(X_train, y_train)
        
        return self

    def predict(self, X_test):
        if self.lda is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before using 'predict'.")
        return self.lda.predict(X_test)

    def predict_proba(self, X_test):
        if self.lda is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before using 'predict_proba'.")
        return self.lda.predict_proba(X_test)

    def score(self, X, y):
        if self.lda is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before using 'score'.")
        return self.lda.score(X, y)

    def get_params(self, deep=True):
        return {
            "n_components": self.n_components,
            "shrinkage": self.shrinkage,
            "solver": self.solver,
            "store_covariance": self.store_covariance,
            "tol": self.tol,
            "cv": self.cv
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self