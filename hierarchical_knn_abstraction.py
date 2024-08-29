import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

class HierarchicalKNNAbstractor(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, n_levels=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski'):
        self.n_neighbors = n_neighbors
        self.n_levels = n_levels
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.label_encoders = []
        self.classifiers = []

    def fit(self, X_train, y_train):
        # Ensure y_train is a 2D array with n_levels columns
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        if y_train.shape[1] != self.n_levels:
            raise ValueError(f"y_train should have {self.n_levels} columns, got {y_train.shape[1]}")

        # Initialize label encoders and classifiers for each level
        self.label_encoders = [LabelEncoder() for _ in range(self.n_levels)]
        self.classifiers = []

        # Fit classifiers for each level
        for level in range(self.n_levels):
            # Encode labels for the current level
            y_encoded = self.label_encoders[level].fit_transform(y_train[:, level])

            # Create and fit KNN classifier for the current level
            knn = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                p=self.p,
                metric=self.metric
            )
            knn.fit(X_train, y_encoded)
            self.classifiers.append(knn)

        return self

    def predict(self, X_test):
        if not self.classifiers:
            raise ValueError("The model has not been fitted yet. Call 'fit' before using 'predict'.")

        predictions = []

        for level, (knn, le) in enumerate(zip(self.classifiers, self.label_encoders)):
            # Predict labels for the current level
            y_pred_encoded = knn.predict(X_test)
            
            # Decode predictions
            y_pred = le.inverse_transform(y_pred_encoded)
            predictions.append(y_pred)

        # Stack predictions for all levels
        return np.column_stack(predictions)

    def get_params(self, deep=True):
        return {
            "n_neighbors": self.n_neighbors,
            "n_levels": self.n_levels,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            "p": self.p,
            "metric": self.metric
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self