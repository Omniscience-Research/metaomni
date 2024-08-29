import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MultiScaleKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, n_scales=3, weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_scales = n_scales
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Create a list to store KNN classifiers for each scale
        self.knn_classifiers_ = []
        
        # Create a scaler for standardization
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Create KNN classifiers for different scales
        for scale in range(self.n_scales):
            knn = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs
            )
            
            # Fit the KNN classifier on the current scale
            knn.fit(X_scaled[:, :X.shape[1] // (2 ** scale)], y)
            
            self.knn_classifiers_.append(knn)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Scale the input data
        X_scaled = self.scaler_.transform(X)
        
        # Make predictions for each scale
        predictions = []
        for scale, knn in enumerate(self.knn_classifiers_):
            pred = knn.predict(X_scaled[:, :X.shape[1] // (2 ** scale)])
            predictions.append(pred)
        
        # Combine predictions from different scales (majority voting)
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0,
            arr=np.array(predictions)
        )
        
        return final_predictions

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        
        # Scale the input data
        X_scaled = self.scaler_.transform(X)
        
        # Make probability predictions for each scale
        probabilities = []
        for scale, knn in enumerate(self.knn_classifiers_):
            prob = knn.predict_proba(X_scaled[:, :X.shape[1] // (2 ** scale)])
            probabilities.append(prob)
        
        # Average probabilities from different scales
        final_probabilities = np.mean(probabilities, axis=0)
        
        return final_probabilities