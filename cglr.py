import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class CompressionGuidedLogisticRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, max_iter=100, tol=1e-4, random_state=None, compression_ratio=0.5):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.compression_ratio = compression_ratio

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Split the data for compression guidance
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_scaled, y, test_size=self.compression_ratio, random_state=self.random_state
        )
        
        # Initialize and fit the base logistic regression model
        self.base_model_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state
        )
        self.base_model_.fit(X_train, y_train)
        
        # Get predictions on the validation set
        y_pred_valid = self.base_model_.predict(X_valid)
        
        # Identify misclassified samples
        misclassified = X_valid[y_pred_valid != y_valid]
        
        # Combine original training data with misclassified samples
        X_combined = np.vstack((X_train, misclassified))
        y_combined = np.hstack((y_train, y_valid[y_pred_valid != y_valid]))
        
        # Fit the final model on the combined dataset
        self.final_model_ = LogisticRegression(
            C=self.C, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state
        )
        self.final_model_.fit(X_combined, y_combined)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['base_model_', 'final_model_', 'scaler_'])

        # Input validation
        X = check_array(X)
        
        # Standardize features
        X_scaled = self.scaler_.transform(X)
        
        # Make predictions using the final model
        return self.final_model_.predict(X_scaled)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['base_model_', 'final_model_', 'scaler_'])

        # Input validation
        X = check_array(X)
        
        # Standardize features
        X_scaled = self.scaler_.transform(X)
        
        # Return probability estimates
        return self.final_model_.predict_proba(X_scaled)