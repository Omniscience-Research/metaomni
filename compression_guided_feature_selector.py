import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

class CompressionGuidedFeatureSelector(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier=SVC(), n_features_to_select=10, compression_ratio=0.5):
        self.base_classifier = base_classifier
        self.n_features_to_select = n_features_to_select
        self.compression_ratio = compression_ratio

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate mutual information between features and target
        mi_scores = mutual_info_classif(X_scaled, y)
        
        # Sort features by mutual information scores
        sorted_indices = np.argsort(mi_scores)[::-1]
        
        # Initialize selected features
        selected_features = []
        
        # Compression-guided feature selection
        for i in range(self.n_features_to_select):
            candidate_features = selected_features + [sorted_indices[i]]
            X_candidate = X_scaled[:, candidate_features]
            
            # Calculate compression ratio
            compression_ratio = len(candidate_features) / self.n_features_in_
            
            if compression_ratio <= self.compression_ratio:
                # Evaluate performance with cross-validation
                scores = cross_val_score(self.base_classifier, X_candidate, y, cv=5)
                mean_score = np.mean(scores)
                
                if len(selected_features) == 0 or mean_score > self.best_score_:
                    selected_features = candidate_features
                    self.best_score_ = mean_score
            else:
                break
        
        # Store selected feature indices
        self.selected_features_ = selected_features
        
        # Fit the base classifier with selected features
        X_selected = X_scaled[:, self.selected_features_]
        self.base_classifier.fit(X_selected, y)
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Select features
        X_selected = X_scaled[:, self.selected_features_]
        
        # Predict using the base classifier
        return self.base_classifier.predict(X_selected)