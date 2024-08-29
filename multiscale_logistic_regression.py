import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MultiScaleLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, n_scales=3, C=1.0, max_iter=100, tol=1e-4, random_state=None):
        self.n_scales = n_scales
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Initialize scalers and classifiers for each scale
        self.scalers_ = []
        self.classifiers_ = []
        
        for scale in range(self.n_scales):
            # Create and fit scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers_.append(scaler)
            
            # Create and fit classifier
            clf = LogisticRegression(
                C=self.C * (2 ** scale),
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )
            clf.fit(X_scaled, y)
            self.classifiers_.append(clf)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Combine predictions from all scales
        predictions = []
        for scaler, clf in zip(self.scalers_, self.classifiers_):
            X_scaled = scaler.transform(X)
            predictions.append(clf.predict_proba(X_scaled))
        
        # Average predictions across scales
        avg_predictions = np.mean(predictions, axis=0)
        
        # Return class labels
        return self.classes_[np.argmax(avg_predictions, axis=1)]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Combine predictions from all scales
        predictions = []
        for scaler, clf in zip(self.scalers_, self.classifiers_):
            X_scaled = scaler.transform(X)
            predictions.append(clf.predict_proba(X_scaled))
        
        # Average predictions across scales
        avg_predictions = np.mean(predictions, axis=0)
        
        return avg_predictions