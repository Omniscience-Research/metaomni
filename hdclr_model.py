import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

class HybridDiscreteContLR(BaseEstimator, ClassifierMixin):
    def __init__(self, discrete_cols=None, continuous_cols=None, l2_penalty=0.01, max_iter=1000, tol=1e-4):
        self.discrete_cols = discrete_cols
        self.continuous_cols = continuous_cols
        self.l2_penalty = l2_penalty
        self.max_iter = max_iter
        self.tol = tol
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _loss(self, params, X, y):
        n_samples, n_features = X.shape
        w = params[:-1]
        b = params[-1]
        
        z = np.dot(X, w) + b
        h = self._sigmoid(z)
        
        loss = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        reg_term = 0.5 * self.l2_penalty * np.sum(w**2)
        
        return loss + reg_term
    
    def _gradient(self, params, X, y):
        n_samples, n_features = X.shape
        w = params[:-1]
        b = params[-1]
        
        z = np.dot(X, w) + b
        h = self._sigmoid(z)
        
        dw = np.dot(X.T, (h - y)) / n_samples + self.l2_penalty * w
        db = np.mean(h - y)
        
        return np.concatenate([dw, [db]])
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        if self.discrete_cols is None or self.continuous_cols is None:
            raise ValueError("Both discrete_cols and continuous_cols must be specified.")
        
        self.discrete_cols_ = np.array(self.discrete_cols)
        self.continuous_cols_ = np.array(self.continuous_cols)
        
        X_discrete = X[:, self.discrete_cols_]
        X_continuous = X[:, self.continuous_cols_]
        
        self.scaler_ = StandardScaler()
        X_continuous_scaled = self.scaler_.fit_transform(X_continuous)
        
        X_combined = np.hstack((X_discrete, X_continuous_scaled))
        
        n_features = X_combined.shape[1]
        initial_params = np.zeros(n_features + 1)
        
        res = minimize(
            fun=self._loss,
            x0=initial_params,
            args=(X_combined, y),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        self.coef_ = res.x[:-1]
        self.intercept_ = res.x[-1]
        
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        X_discrete = X[:, self.discrete_cols_]
        X_continuous = X[:, self.continuous_cols_]
        X_continuous_scaled = self.scaler_.transform(X_continuous)
        
        X_combined = np.hstack((X_discrete, X_continuous_scaled))
        
        z = np.dot(X_combined, self.coef_) + self.intercept_
        proba = self._sigmoid(z)
        
        return np.column_stack((1 - proba, proba))
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)