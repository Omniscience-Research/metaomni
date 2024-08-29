import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class FractalQuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize the scaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Build the fractal QDA tree
        self.root_ = self._build_tree(X_scaled, y, depth=0)
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Scale the input
        X_scaled = self.scaler_.transform(X)

        # Predict using the fractal QDA tree
        return np.array([self._predict_tree(self.root_, x) for x in X_scaled])

    def _build_tree(self, X, y, depth):
        n_samples = X.shape[0]
        
        # Base case: maximum depth reached or not enough samples to split
        if depth == self.max_depth or n_samples < self.min_samples_split:
            return QuadraticDiscriminantAnalysis().fit(X, y)
        
        # Fit a QDA model at this level
        qda = QuadraticDiscriminantAnalysis().fit(X, y)
        
        # Split the data based on QDA predictions
        y_pred = qda.predict(X)
        
        # Recursively build subtrees
        subtrees = {}
        for class_label in self.classes_:
            mask = y_pred == class_label
            if np.sum(mask) >= self.min_samples_split:
                subtrees[class_label] = self._build_tree(X[mask], y[mask], depth + 1)
        
        return {'qda': qda, 'subtrees': subtrees}

    def _predict_tree(self, node, x):
        if isinstance(node, QuadraticDiscriminantAnalysis):
            return node.predict([x])[0]
        
        qda_pred = node['qda'].predict([x])[0]
        
        if qda_pred in node['subtrees']:
            return self._predict_tree(node['subtrees'][qda_pred], x)
        else:
            return qda_pred