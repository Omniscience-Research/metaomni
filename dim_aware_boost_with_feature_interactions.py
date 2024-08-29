import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DimAwareInteractionBooster(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1, 
                 interaction_threshold=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.interaction_threshold = interaction_threshold
        self.random_state = random_state

    def _detect_interactions(self, X, y):
        n_features = X.shape[1]
        interactions = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_score = self._calculate_interaction_score(X[:, i], X[:, j], y)
                if interaction_score > self.interaction_threshold:
                    interactions.append((i, j))

        return interactions

    def _calculate_interaction_score(self, feature1, feature2, y):
        # This is a simplified interaction score calculation
        # You may want to implement a more sophisticated method
        combined = feature1 * feature2
        correlation = np.corrcoef(combined, y)[0, 1]
        return abs(correlation)

    def _create_interaction_features(self, X, interactions):
        X_with_interactions = X.copy()
        for i, j in interactions:
            new_feature = X[:, i] * X[:, j]
            X_with_interactions = np.column_stack((X_with_interactions, new_feature))
        return X_with_interactions

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        # Detect feature interactions
        self.interactions_ = self._detect_interactions(X, y)

        # Create new features based on detected interactions
        X_with_interactions = self._create_interaction_features(X, self.interactions_)

        # Initialize the ensemble
        self.estimators_ = []
        self.feature_importances_ = np.zeros(X_with_interactions.shape[1])

        # Initialize predictions
        F = np.zeros((X.shape[0], self.n_classes_))

        for _ in range(self.n_estimators):
            # Calculate negative gradient
            negative_gradient = y - self._sigmoid(F)

            # Fit a new tree to the negative gradient
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_leaf=self.min_samples_leaf,
                                          random_state=self.random_state)
            tree.fit(X_with_interactions, negative_gradient)

            # Update predictions
            F += self.learning_rate * tree.predict(X_with_interactions)

            # Store the tree and update feature importances
            self.estimators_.append(tree)
            self.feature_importances_ += tree.feature_importances_

        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        # Create interaction features for prediction
        X_with_interactions = self._create_interaction_features(X, self.interactions_)

        # Calculate final predictions
        F = np.zeros((X.shape[0], self.n_classes_))
        for estimator in self.estimators_:
            F += self.learning_rate * estimator.predict(X_with_interactions)

        return self._sigmoid(F)

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))