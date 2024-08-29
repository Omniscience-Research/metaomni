import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

class FractalLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=None, fractal_levels=3):
        self.n_components = n_components
        self.fractal_levels = fractal_levels

    def _compute_fractal_features(self, X):
        n_samples, n_features = X.shape
        fractal_features = []

        for level in range(self.fractal_levels):
            step = 2 ** level
            for i in range(0, n_features, step):
                end = min(i + step, n_features)
                fractal_features.append(np.mean(X[:, i:end], axis=1))
                fractal_features.append(np.std(X[:, i:end], axis=1))

        return np.column_stack(fractal_features)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if self.n_components is None:
            self.n_components = min(n_classes - 1, X.shape[1])

        # Compute fractal features
        X_fractal = self._compute_fractal_features(X)

        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_fractal)

        # Compute class means and overall mean
        class_means = []
        overall_mean = np.mean(X_scaled, axis=0)

        for c in self.classes_:
            class_means.append(np.mean(X_scaled[y == c], axis=0))

        # Compute between-class and within-class scatter matrices
        S_b = np.zeros((X_scaled.shape[1], X_scaled.shape[1]))
        S_w = np.zeros((X_scaled.shape[1], X_scaled.shape[1]))

        for i, mean in enumerate(class_means):
            class_samples = X_scaled[y == self.classes_[i]]
            n_samples = len(class_samples)

            S_b += n_samples * np.outer(mean - overall_mean, mean - overall_mean)
            S_w += np.cov(class_samples.T) * (n_samples - 1)

        # Solve the generalized eigenvalue problem
        eig_vals, eig_vecs = eigh(S_b, S_w)

        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eig_vals)[::-1]
        self.eig_vecs_ = eig_vecs[:, idx][:, :self.n_components]

        # Compute projections of class means
        self.means_ = np.dot(class_means, self.eig_vecs_)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # Compute fractal features
        X_fractal = self._compute_fractal_features(X)

        # Standardize features
        X_scaled = self.scaler_.transform(X_fractal)

        # Project data onto LDA space
        X_lda = np.dot(X_scaled, self.eig_vecs_)

        # Compute distances to class means
        distances = np.array([np.sum((X_lda - mean) ** 2, axis=1) for mean in self.means_])

        # Predict the class with the smallest distance
        return self.classes_[np.argmin(distances, axis=0)]