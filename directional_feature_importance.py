import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier

class DirectionalFeatureImportance(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=100, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_ = X.shape[1]
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(self.n_features_)
        self.feature_directions_ = np.zeros(self.n_features_)
        
        # Use DecisionTreeClassifier as the base estimator if not specified
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(random_state=self.random_state)
        
        # Fit multiple estimators and calculate directional feature importance
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Fit the base estimator
            self.base_estimator.fit(X_sample, y_sample)
            
            # Calculate feature importances
            importances = self.base_estimator.feature_importances_
            
            # Calculate feature directions
            directions = self._calculate_feature_directions(X_sample, y_sample)
            
            # Update feature importances and directions
            self.feature_importances_ += importances
            self.feature_directions_ += directions
        
        # Normalize feature importances and directions
        self.feature_importances_ /= self.n_estimators
        self.feature_directions_ /= self.n_estimators
        
        # Store the fitted attributes
        self.X_ = X
        self.y_ = y
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions using the base estimator
        return self.base_estimator.predict(X)

    def _calculate_feature_directions(self, X, y):
        directions = np.zeros(self.n_features_)
        
        for feature in range(self.n_features_):
            # Calculate correlation between feature and target
            correlation = np.corrcoef(X[:, feature], y)[0, 1]
            
            # Determine direction based on correlation
            if correlation > 0:
                directions[feature] = 1
            elif correlation < 0:
                directions[feature] = -1
        
        return directions

    def get_directional_feature_importance(self):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Combine feature importances and directions
        directional_importance = self.feature_importances_ * self.feature_directions_
        
        return directional_importance