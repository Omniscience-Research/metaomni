import numpy as np
from scipy.stats import multivariate_normal, multinomial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder

class HybridDiscreteContQDA(BaseEstimator, ClassifierMixin):
    def __init__(self, discrete_columns=None):
        self.discrete_columns = discrete_columns

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        if self.discrete_columns is None:
            self.discrete_columns = []
        self.continuous_columns = [i for i in range(self.n_features_) if i not in self.discrete_columns]

        # Use LabelEncoder for discrete features
        self.label_encoders_ = [LabelEncoder() for _ in self.discrete_columns]
        for i, le in zip(self.discrete_columns, self.label_encoders_):
            X[:, i] = le.fit_transform(X[:, i])

        # Compute class priors
        self.priors_ = np.bincount(y) / len(y)

        # Compute means and covariances for continuous features
        self.means_ = []
        self.covariances_ = []
        
        # Compute probabilities for discrete features
        self.discrete_probs_ = []

        for c in self.classes_:
            X_c = X[y == c]
            
            # Continuous features
            X_c_cont = X_c[:, self.continuous_columns]
            self.means_.append(np.mean(X_c_cont, axis=0))
            self.covariances_.append(np.cov(X_c_cont, rowvar=False))
            
            # Discrete features
            X_c_disc = X_c[:, self.discrete_columns]
            disc_probs = []
            for j in range(X_c_disc.shape[1]):
                unique, counts = np.unique(X_c_disc[:, j], return_counts=True)
                probs = np.zeros(len(self.label_encoders_[j].classes_))
                probs[unique] = counts / len(X_c_disc)
                disc_probs.append(probs)
            self.discrete_probs_.append(disc_probs)

        self.means_ = np.array(self.means_)
        self.covariances_ = np.array(self.covariances_)
        self.discrete_probs_ = np.array(self.discrete_probs_)

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # Transform discrete features
        for i, le in zip(self.discrete_columns, self.label_encoders_):
            X[:, i] = le.transform(X[:, i])

        proba = np.zeros((X.shape[0], self.n_classes_))

        for i, c in enumerate(self.classes_):
            # Continuous part
            X_cont = X[:, self.continuous_columns]
            proba[:, i] = multivariate_normal.pdf(X_cont, mean=self.means_[i], cov=self.covariances_[i])
            
            # Discrete part
            X_disc = X[:, self.discrete_columns]
            for j in range(X_disc.shape[1]):
                proba[:, i] *= self.discrete_probs_[i, j][X_disc[:, j].astype(int)]
            
            # Multiply by prior
            proba[:, i] *= self.priors_[i]

        # Normalize probabilities
        proba /= proba.sum(axis=1)[:, np.newaxis]
        return proba

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]