import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy

class RandomnessAdaptiveTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None, max_randomness=0.5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_randomness = max_randomness

    def _calculate_compressibility(self, X):
        # Calculate compressibility using entropy
        entropies = np.apply_along_axis(lambda x: entropy(x), 0, X)
        return 1 - np.mean(entropies) / np.log2(X.shape[0])

    def _inject_randomness(self, X, y, compressibility):
        # Inject randomness inversely proportional to compressibility
        randomness = self.max_randomness * (1 - compressibility)
        
        # Add random noise to features
        X_noisy = X + np.random.normal(0, randomness, X.shape)
        
        # Randomly flip some labels
        flip_mask = np.random.random(y.shape) < randomness
        y_noisy = np.where(flip_mask, 1 - y, y)
        
        return X_noisy, y_noisy

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Calculate compressibility
        compressibility = self._calculate_compressibility(X)
        
        # Inject randomness
        X_noisy, y_noisy = self._inject_randomness(X, y, compressibility)
        
        # Create and fit the decision tree
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        self.tree_.fit(X_noisy, y_noisy)
        
        # Store compressibility for later use
        self.compressibility_ = compressibility
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Inject randomness (but less than during training)
        X_noisy, _ = self._inject_randomness(X, np.zeros(X.shape[0]), self.compressibility_ / 2)
        
        return self.tree_.predict(X_noisy)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Inject randomness (but less than during training)
        X_noisy, _ = self._inject_randomness(X, np.zeros(X.shape[0]), self.compressibility_ / 2)
        
        return self.tree_.predict_proba(X_noisy)