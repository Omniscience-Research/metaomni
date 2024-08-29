import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

class DimensionalityAdaptiveQDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=10, min_dim=2, max_dim=None, priors=None, reg_param=0.0):
        self.n_neighbors = n_neighbors
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.priors = priors
        self.reg_param = reg_param
        self.classes_ = None
        self.qda_models_ = {}
        self.pca_models_ = {}
        self.nn_model_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        
        if self.max_dim is None:
            self.max_dim = n_features

        # Fit nearest neighbors model for local dimensionality estimation
        self.nn_model_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn_model_.fit(X)

        # Fit QDA and PCA models for each possible dimensionality
        for dim in range(self.min_dim, self.max_dim + 1):
            pca = PCA(n_components=dim)
            X_pca = pca.fit_transform(X)
            
            qda = QuadraticDiscriminantAnalysis(priors=self.priors, reg_param=self.reg_param)
            qda.fit(X_pca, y)
            
            self.qda_models_[dim] = qda
            self.pca_models_[dim] = pca

        return self

    def predict(self, X):
        if self.classes_ is None:
            raise ValueError("Classifier not fitted. Call 'fit' first.")

        predictions = np.zeros(X.shape[0], dtype=self.classes_.dtype)

        for i, x in enumerate(X):
            # Estimate local dimensionality
            _, indices = self.nn_model_.kneighbors([x])
            local_dim = self._estimate_local_dimensionality(X[indices[0]])
            
            # Clip dimensionality to allowed range
            local_dim = max(self.min_dim, min(local_dim, self.max_dim))
            
            # Apply PCA and QDA for the estimated dimensionality
            x_pca = self.pca_models_[local_dim].transform([x])
            predictions[i] = self.qda_models_[local_dim].predict(x_pca)[0]

        return predictions

    def _estimate_local_dimensionality(self, X_local):
        # Estimate local dimensionality using PCA
        pca = PCA()
        pca.fit(X_local)
        
        # Find number of components that explain 95% of variance
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        return np.argmax(cumulative_variance_ratio >= 0.95) + 1

    def predict_proba(self, X):
        if self.classes_ is None:
            raise ValueError("Classifier not fitted. Call 'fit' first.")

        probas = np.zeros((X.shape[0], len(self.classes_)))

        for i, x in enumerate(X):
            _, indices = self.nn_model_.kneighbors([x])
            local_dim = self._estimate_local_dimensionality(X[indices[0]])
            local_dim = max(self.min_dim, min(local_dim, self.max_dim))
            
            x_pca = self.pca_models_[local_dim].transform([x])
            probas[i] = self.qda_models_[local_dim].predict_proba(x_pca)[0]

        return probas