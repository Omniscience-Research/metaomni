import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

class MultiScaleLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_scales=3, scale_factor=2):
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.lda_models = []
        self.scalers = []

    def fit(self, X_train, y_train):
        n_features = X_train.shape[1]
        
        for scale in range(self.n_scales):
            # Calculate the number of features for this scale
            n_features_scale = max(1, n_features // (self.scale_factor ** scale))
            
            # Create a scaler for this scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train[:, :n_features_scale])
            
            # Create and fit LDA model for this scale
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_scaled, y_train)
            
            # Store the scaler and LDA model
            self.scalers.append(scaler)
            self.lda_models.append(lda)
        
        return self

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        predictions = np.zeros((n_samples, len(self.lda_models)))
        
        for i, (scaler, lda) in enumerate(zip(self.scalers, self.lda_models)):
            # Scale the features for this scale
            n_features_scale = scaler.n_features_in_
            X_scaled = scaler.transform(X_test[:, :n_features_scale])
            
            # Make predictions using the LDA model for this scale
            predictions[:, i] = lda.predict(X_scaled)
        
        # Combine predictions from all scales (majority voting)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=1,
            arr=predictions
        )
        
        return final_predictions

    def predict_proba(self, X_test):
        n_samples = X_test.shape[0]
        n_classes = len(self.lda_models[0].classes_)
        probas = np.zeros((n_samples, n_classes))
        
        for i, (scaler, lda) in enumerate(zip(self.scalers, self.lda_models)):
            # Scale the features for this scale
            n_features_scale = scaler.n_features_in_
            X_scaled = scaler.transform(X_test[:, :n_features_scale])
            
            # Get probabilities using the LDA model for this scale
            probas += lda.predict_proba(X_scaled)
        
        # Average probabilities across all scales
        probas /= len(self.lda_models)
        
        return probas