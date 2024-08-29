import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class WeakLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=1):
        self.tree = DecisionTreeClassifier(max_depth=max_depth)
        self.feature_idx = None
        self.abstraction_level = None

    def fit(self, X, y, sample_weight=None):
        self.feature_idx = np.random.randint(X.shape[1])
        self.abstraction_level = np.random.randint(1, 11)  # 1 to 10 levels
        X_abstracted = self._abstract_feature(X[:, self.feature_idx])
        self.tree.fit(X_abstracted.reshape(-1, 1), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        X_abstracted = self._abstract_feature(X[:, self.feature_idx])
        return self.tree.predict(X_abstracted.reshape(-1, 1))

    def _abstract_feature(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(-1, 1))
        kmeans = KMeans(n_clusters=self.abstraction_level)
        return kmeans.fit_predict(X_scaled)

class AdaptiveAbstractionBooster(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        
        # Initialize weights
        sample_weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            # Train weak learner
            weak_learner = WeakLearner()
            weak_learner.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Make predictions
            y_pred = weak_learner.predict(X_train)
            
            # Calculate error
            err = np.sum(sample_weights * (y_pred != y_train)) / np.sum(sample_weights)
            
            # Calculate estimator weight
            estimator_weight = self.learning_rate * np.log((1 - err) / err)
            
            # Update sample weights
            sample_weights *= np.exp(estimator_weight * (y_pred != y_train))
            sample_weights /= np.sum(sample_weights)
            
            # Store the weak learner and its weight
            self.estimators.append(weak_learner)
            self.estimator_weights.append(estimator_weight)
        
        return self

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        y_pred = np.zeros(n_samples)
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            y_pred += weight * estimator.predict(X_test)
        
        return np.sign(y_pred)

    def feature_importances(self):
        feature_importances = np.zeros(self.n_features_)
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            feature_importances[estimator.feature_idx] += weight
        return feature_importances / np.sum(feature_importances)