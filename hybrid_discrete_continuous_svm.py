import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class HybridDiscreteContSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', 
                 coef0=0.0, shrinking=True, probability=False, 
                 tol=1e-3, cache_size=200, class_weight=None, 
                 verbose=False, max_iter=-1, decision_function_shape='ovr', 
                 break_ties=False, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        
        self.svm = None
        self.scaler = StandardScaler()
        self.discrete_indices = None
        self.continuous_indices = None

    def fit(self, X, y):
        # Identify discrete and continuous features
        self.discrete_indices = np.where(np.all(X.astype(int) == X, axis=0))[0]
        self.continuous_indices = np.setdiff1d(np.arange(X.shape[1]), self.discrete_indices)

        # Separate discrete and continuous features
        X_discrete = X[:, self.discrete_indices]
        X_continuous = X[:, self.continuous_indices]

        # Scale continuous features
        X_continuous_scaled = self.scaler.fit_transform(X_continuous)

        # Combine discrete and scaled continuous features
        X_combined = np.hstack((X_discrete, X_continuous_scaled))

        # Create and fit the SVM
        self.svm = SVC(
            C=self.C, kernel=self.kernel, degree=self.degree, 
            gamma=self.gamma, coef0=self.coef0, shrinking=self.shrinking, 
            probability=self.probability, tol=self.tol, 
            cache_size=self.cache_size, class_weight=self.class_weight, 
            verbose=self.verbose, max_iter=self.max_iter, 
            decision_function_shape=self.decision_function_shape, 
            break_ties=self.break_ties, random_state=self.random_state
        )
        self.svm.fit(X_combined, y)

        return self

    def predict(self, X):
        # Separate discrete and continuous features
        X_discrete = X[:, self.discrete_indices]
        X_continuous = X[:, self.continuous_indices]

        # Scale continuous features
        X_continuous_scaled = self.scaler.transform(X_continuous)

        # Combine discrete and scaled continuous features
        X_combined = np.hstack((X_discrete, X_continuous_scaled))

        # Make predictions
        return self.svm.predict(X_combined)

    def predict_proba(self, X):
        if not self.probability:
            raise AttributeError("Probabilities are not available. Set probability=True during initialization.")
        
        # Separate discrete and continuous features
        X_discrete = X[:, self.discrete_indices]
        X_continuous = X[:, self.continuous_indices]

        # Scale continuous features
        X_continuous_scaled = self.scaler.transform(X_continuous)

        # Combine discrete and scaled continuous features
        X_combined = np.hstack((X_discrete, X_continuous_scaled))

        # Make probability predictions
        return self.svm.predict_proba(X_combined)