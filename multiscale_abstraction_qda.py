import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MultiScaleAbstractionQDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_scales=3, aggregation_method='mean'):
        self.n_scales = n_scales
        self.aggregation_method = aggregation_method
        self.qda_models = []
        self.scalers = []
    
    def _aggregate_features(self, X, scale):
        if scale == 1:
            return X
        
        n_features = X.shape[1]
        n_aggregated = n_features // scale
        
        if self.aggregation_method == 'mean':
            return np.array([X[:, i:i+scale].mean(axis=1) for i in range(0, n_features, scale)]).T
        elif self.aggregation_method == 'max':
            return np.array([X[:, i:i+scale].max(axis=1) for i in range(0, n_features, scale)]).T
        else:
            raise ValueError("Unsupported aggregation method. Use 'mean' or 'max'.")
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        
        for scale in range(1, self.n_scales + 1):
            X_scaled = self._aggregate_features(X, scale)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_scaled)
            self.scalers.append(scaler)
            
            qda = QuadraticDiscriminantAnalysis()
            qda.fit(X_scaled, y)
            self.qda_models.append(qda)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = []
        
        for scale, (qda, scaler) in enumerate(zip(self.qda_models, self.scalers), 1):
            X_scaled = self._aggregate_features(X, scale)
            X_scaled = scaler.transform(X_scaled)
            predictions.append(qda.predict_proba(X_scaled))
        
        # Combine predictions from all scales
        combined_predictions = np.mean(predictions, axis=0)
        return self.classes_[np.argmax(combined_predictions, axis=1)]

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = []
        
        for scale, (qda, scaler) in enumerate(zip(self.qda_models, self.scalers), 1):
            X_scaled = self._aggregate_features(X, scale)
            X_scaled = scaler.transform(X_scaled)
            predictions.append(qda.predict_proba(X_scaled))
        
        # Combine predictions from all scales
        return np.mean(predictions, axis=0)