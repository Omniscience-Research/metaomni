import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class AdaptiveK(BaseEstimator, ClassifierMixin):
    def __init__(self, k_min=1, k_max=20, step=1):
        self.k_min = k_min
        self.k_max = k_max
        self.step = step

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]

        # Split the training data for internal validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        best_k = self.k_min
        best_accuracy = 0

        for k in range(self.k_min, self.k_max + 1, self.step):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        self.best_k_ = best_k
        self.knn_ = KNeighborsClassifier(n_neighbors=best_k)
        self.knn_.fit(X, y)

        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Check that the input features match the number of features seen during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError("The number of features in predict is different from the number of features in fit.")

        return self.knn_.predict(X)

    def get_params(self, deep=True):
        return {"k_min": self.k_min, "k_max": self.k_max, "step": self.step}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self