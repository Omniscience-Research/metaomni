import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import entropy

class EntropyGuidedRandomSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', n_subsets=10, subset_size=0.8, random_state=None):
        self.C = C
        self.kernel = kernel
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.random_state = random_state
        self.svms = []
        self.feature_importances_ = None

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Create multiple SVM models on random subsets
        np.random.seed(self.random_state)
        n_samples = int(X.shape[0] * self.subset_size)
        
        for _ in range(self.n_subsets):
            subset_indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_subset, y_subset = X[subset_indices], y[subset_indices]
            
            svm = SVC(C=self.C, kernel=self.kernel, probability=True)
            svm.fit(X_subset, y_subset)
            self.svms.append(svm)
        
        # Calculate feature importances based on entropy
        self.calculate_feature_importances(X)
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions using all SVM models
        predictions = np.array([svm.predict(X) for svm in self.svms])
        
        # Use majority voting for final prediction
        final_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)
        
        return final_predictions

    def calculate_feature_importances(self, X):
        feature_probs = []
        
        for feature in range(X.shape[1]):
            feature_values = X[:, feature]
            _, counts = np.unique(feature_values, return_counts=True)
            probs = counts / len(feature_values)
            feature_probs.append(probs)
        
        entropies = [entropy(probs) for probs in feature_probs]
        self.feature_importances_ = entropies / np.sum(entropies)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make probability predictions using all SVM models
        probas = np.array([svm.predict_proba(X) for svm in self.svms])
        
        # Average probabilities across all models
        final_probas = np.mean(probas, axis=0)
        
        return final_probas