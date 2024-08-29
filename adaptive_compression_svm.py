import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class AdaptiveCompressionSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, degree_of_compression=0.5, kernel='rbf', C=1.0, random_state=None):
        self.degree_of_compression = degree_of_compression
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
        
    def _compress_data(self, X):
        n_features = X.shape[1]
        n_compressed_features = max(1, int(n_features * (1 - self.degree_of_compression)))
        
        # Perform PCA to compress the data
        cov_matrix = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top n_compressed_features eigenvectors
        compression_matrix = eigenvectors[:, :n_compressed_features]
        
        # Compress the data
        X_compressed = np.dot(X, compression_matrix)
        
        return X_compressed, compression_matrix
    
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Compress the data
        self.X_compressed_, self.compression_matrix_ = self._compress_data(X)
        
        # Scale the compressed data
        self.scaler_ = StandardScaler()
        X_compressed_scaled = self.scaler_.fit_transform(self.X_compressed_)
        
        # Fit the SVM on the compressed data
        self.svm_ = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        self.svm_.fit(X_compressed_scaled, y)
        
        # Return the classifier
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['X_compressed_', 'compression_matrix_', 'svm_', 'scaler_'])
        
        # Input validation
        X = check_array(X)
        
        # Compress the input data
        X_compressed = np.dot(X, self.compression_matrix_)
        
        # Scale the compressed data
        X_compressed_scaled = self.scaler_.transform(X_compressed)
        
        # Predict using the SVM
        return self.svm_.predict(X_compressed_scaled)