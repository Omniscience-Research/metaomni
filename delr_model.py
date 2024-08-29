import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DirectionalEnsembleLogisticRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, C=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.C = C
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        if len(self.classes_) != 2:
            raise ValueError("DirectionalEnsembleLogisticRegressor only supports binary classification.")
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize the ensemble
        self.estimators_ = []
        
        np.random.seed(self.random_state)
        
        for _ in range(self.n_estimators):
            # Generate random direction
            direction = np.random.randn(X.shape[1])
            direction /= np.linalg.norm(direction)
            
            # Project data onto the random direction
            X_proj = X.dot(direction)
            
            # Fit logistic regression on the projected data
            lr = LogisticRegression(C=self.C, random_state=self.random_state)
            lr.fit(X_proj.reshape(-1, 1), y)
            
            # Store the estimator and its direction
            self.estimators_.append((lr, direction))
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but DirectionalEnsembleLogisticRegressor is expecting {self.n_features_in_} features as input.")
        
        # Collect predictions from all estimators
        predictions = []
        for lr, direction in self.estimators_:
            X_proj = X.dot(direction)
            pred = lr.predict(X_proj.reshape(-1, 1))
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0,
            arr=predictions
        )
        
        return self.classes_[final_predictions]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but DirectionalEnsembleLogisticRegressor is expecting {self.n_features_in_} features as input.")
        
        # Collect probabilities from all estimators
        probas = []
        for lr, direction in self.estimators_:
            X_proj = X.dot(direction)
            prob = lr.predict_proba(X_proj.reshape(-1, 1))
            probas.append(prob)
        
        # Average probabilities
        final_probas = np.mean(probas, axis=0)
        
        return final_probas