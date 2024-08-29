import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

class DimAdaptiveLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=100, tol=1e-4, random_state=None):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.logistic_regression = None
        self.best_k = None

    def fit(self, X_train, y_train):
        # Scale the features
        X_scaled = self.scaler.fit_transform(X_train)

        # Determine the optimal number of features
        n_features = X_scaled.shape[1]
        k_range = range(1, n_features + 1)
        best_score = -np.inf

        for k in k_range:
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(X_scaled, y_train)
            
            lr = LogisticRegression(C=self.C, max_iter=self.max_iter, 
                                    tol=self.tol, random_state=self.random_state)
            
            scores = cross_val_score(lr, X_selected, y_train, cv=5)
            mean_score = np.mean(scores)

            if mean_score > best_score:
                best_score = mean_score
                self.best_k = k

        # Select the best features
        self.feature_selector = SelectKBest(f_classif, k=self.best_k)
        X_best = self.feature_selector.fit_transform(X_scaled, y_train)

        # Train the final logistic regression model
        self.logistic_regression = LogisticRegression(C=self.C, max_iter=self.max_iter, 
                                                      tol=self.tol, random_state=self.random_state)
        self.logistic_regression.fit(X_best, y_train)

        return self

    def predict(self, X_test):
        # Scale the features
        X_scaled = self.scaler.transform(X_test)

        # Select the best features
        X_best = self.feature_selector.transform(X_scaled)

        # Make predictions
        return self.logistic_regression.predict(X_best)

    def predict_proba(self, X_test):
        # Scale the features
        X_scaled = self.scaler.transform(X_test)

        # Select the best features
        X_best = self.feature_selector.transform(X_scaled)

        # Make probability predictions
        return self.logistic_regression.predict_proba(X_best)