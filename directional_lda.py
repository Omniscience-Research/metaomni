import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder

class DirectionalLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_components is None:
            self.n_components_ = min(X.shape[1], self.n_classes_ - 1)
        else:
            self.n_components_ = min(X.shape[1], self.n_classes_ - 1, self.n_components)
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Compute class means
        class_means = []
        for cls in range(self.n_classes_):
            class_means.append(np.mean(X[y_encoded == cls], axis=0))
        
        # Compute within-class scatter matrix
        S_w = np.zeros((X.shape[1], X.shape[1]))
        for cls in range(self.n_classes_):
            class_scatter = np.cov(X[y_encoded == cls].T)
            S_w += class_scatter * np.sum(y_encoded == cls)
        
        # Compute between-class scatter matrix
        overall_mean = np.mean(X, axis=0)
        S_b = np.zeros((X.shape[1], X.shape[1]))
        for cls in range(self.n_classes_):
            n_samples = np.sum(y_encoded == cls)
            mean_diff = class_means[cls] - overall_mean
            S_b += n_samples * np.outer(mean_diff, mean_diff)
        
        # Solve the generalized eigenvalue problem
        eig_vals, eig_vecs = eigh(S_b, S_w)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, idx]
        
        # Select the top n_components eigenvectors
        self.components_ = eig_vecs[:, :self.n_components_]
        
        # Project class means
        self.means_ = np.dot(class_means, self.components_)
        
        # Store X and y for prediction
        self.X_ = X
        self.y_ = y
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Project the test data
        X_transformed = np.dot(X, self.components_)
        
        # Compute distances to class means
        distances = np.zeros((X.shape[0], self.n_classes_))
        for cls in range(self.n_classes_):
            distances[:, cls] = np.sum((X_transformed - self.means_[cls])**2, axis=1)
        
        # Predict the class with minimum distance
        y_pred = self.classes_[np.argmin(distances, axis=1)]
        
        return y_pred