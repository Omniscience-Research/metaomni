import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

class AdaptiveComplexityLogisticRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter=1000, tol=1e-4, learning_rate=0.01, 
                 complexity_penalty=0.01, adaptive_rate=0.1):
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.complexity_penalty = complexity_penalty
        self.adaptive_rate = adaptive_rate
        self.coef_ = None
        self.intercept_ = None
        self.scaler = StandardScaler()
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, X, y, coef, intercept):
        z = np.dot(X, coef) + intercept
        predictions = self._sigmoid(z)
        loss = log_loss(y, predictions)
        complexity = np.sum(np.abs(coef))
        return loss + self.complexity_penalty * complexity
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        n_samples, n_features = X_train.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        best_val_loss = np.inf
        no_improvement_count = 0
        
        for _ in range(self.max_iter):
            z = np.dot(X_train, self.coef_) + self.intercept_
            predictions = self._sigmoid(z)
            
            # Compute gradients
            d_coef = np.dot(X_train.T, predictions - y_train) / n_samples + self.complexity_penalty * np.sign(self.coef_)
            d_intercept = np.mean(predictions - y_train)
            
            # Update parameters
            self.coef_ -= self.learning_rate * d_coef
            self.intercept_ -= self.learning_rate * d_intercept
            
            # Compute validation loss
            val_loss = self._compute_loss(X_val, y_val, self.coef_, self.intercept_)
            
            # Check for convergence and adapt complexity
            if val_loss < best_val_loss - self.tol:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= 5:
                self.complexity_penalty *= (1 + self.adaptive_rate)
                no_improvement_count = 0
            
            if np.abs(val_loss - best_val_loss) < self.tol:
                break
        
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        z = np.dot(X_scaled, self.coef_) + self.intercept_
        probabilities = self._sigmoid(z)
        return np.column_stack((1 - probabilities, probabilities))
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)