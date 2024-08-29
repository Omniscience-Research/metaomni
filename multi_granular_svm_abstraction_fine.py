import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class MultiGranularSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_levels=3, C=1.0, kernel='rbf', random_state=None):
        self.n_levels = n_levels
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.svms = []
        self.label_encoders = []

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Create label encoders for each granularity level
        self.label_encoders = [LabelEncoder() for _ in range(self.n_levels)]

        # Fit SVMs for each granularity level
        for level in range(self.n_levels):
            # Encode labels for current granularity level
            y_encoded = self.label_encoders[level].fit_transform(
                ['.'.join(label.split('.')[:level+1]) for label in y]
            )

            # Create and fit SVM for current level
            svm = SVC(C=self.C, kernel=self.kernel, random_state=self.random_state)
            svm.fit(X, y_encoded)
            self.svms.append(svm)

        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        predictions = []
        for i in range(len(X)):
            sample = X[i].reshape(1, -1)
            prediction = []
            
            for level in range(self.n_levels):
                # Predict at current granularity level
                level_pred = self.svms[level].predict(sample)[0]
                
                # Decode prediction
                decoded_pred = self.label_encoders[level].inverse_transform([level_pred])[0]
                
                # Add to prediction list
                prediction.append(decoded_pred)
                
                # If prediction is not full (contains all levels), break
                if len(decoded_pred.split('.')) < self.n_levels:
                    break
            
            predictions.append('.'.join(prediction))

        return np.array(predictions)

    def get_params(self, deep=True):
        return {
            "n_levels": self.n_levels,
            "C": self.C,
            "kernel": self.kernel,
            "random_state": self.random_state
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self