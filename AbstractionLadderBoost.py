import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class AbstractionLadderBooster(BaseEstimator, ClassifierMixin):
    def __init__(self, n_levels=3, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_levels = n_levels
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.transformers = []

    def _create_abstraction_level(self, X, level):
        if level == 0:
            return X
        elif level == 1:
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        elif level == 2:
            pca = PCA(n_components=min(X.shape[1], 10))
            return pca.fit_transform(X)
        else:
            k = max(1, X.shape[1] // 2)
            selector = SelectKBest(f_classif, k=k)
            return selector.fit_transform(X, self.y_train)

    def fit(self, X_train, y_train):
        self.y_train = y_train
        X_abstracted = X_train.copy()

        for level in range(self.n_levels):
            # Create abstraction for this level
            X_level = self._create_abstraction_level(X_abstracted, level)
            self.transformers.append(X_level)

            # Train a GradientBoostingClassifier on this abstraction level
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42
            )
            model.fit(X_level, y_train)
            self.models.append(model)

            # Update X_abstracted for the next level
            if level < self.n_levels - 1:
                X_abstracted = model.apply(X_level)[:, :, 0]

        return self

    def predict(self, X_test):
        predictions = np.zeros((X_test.shape[0], len(self.models)))

        for level, (model, transformer) in enumerate(zip(self.models, self.transformers)):
            if level == 0:
                X_level = X_test
            else:
                X_level = transformer.transform(X_test)
            
            predictions[:, level] = model.predict(X_level)

        # Majority voting
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x.astype(int))),
            axis=1,
            arr=predictions
        )

        return final_predictions

    def predict_proba(self, X_test):
        probas = np.zeros((X_test.shape[0], len(self.models), 2))

        for level, (model, transformer) in enumerate(zip(self.models, self.transformers)):
            if level == 0:
                X_level = X_test
            else:
                X_level = transformer.transform(X_test)
            
            probas[:, level, :] = model.predict_proba(X_level)

        # Average probabilities across all levels
        final_probas = np.mean(probas, axis=1)

        return final_probas