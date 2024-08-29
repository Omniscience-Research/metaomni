import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

class HierarchicalNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.hierarchy = {}
        self.label_encoders = {}
        self.classifiers = {}

    def _build_hierarchy(self, y):
        hierarchy = {}
        for label in y:
            parts = label.split('/')
            current = hierarchy
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]
        return hierarchy

    def _train_level(self, X, y, level, prefix=''):
        unique_labels = np.unique([label.split('/')[level] for label in y if len(label.split('/')) > level])
        
        if len(unique_labels) <= 1:
            return

        le = LabelEncoder()
        y_encoded = le.fit_transform([label.split('/')[level] for label in y if len(label.split('/')) > level])
        
        clf = MultinomialNB(alpha=self.alpha)
        clf.fit(X, y_encoded)
        
        self.label_encoders[prefix] = le
        self.classifiers[prefix] = clf

        for label in unique_labels:
            new_prefix = f"{prefix}/{label}" if prefix else label
            mask = np.array([l.startswith(new_prefix) for l in y])
            self._train_level(X[mask], y[mask], level + 1, new_prefix)

    def fit(self, X, y):
        self.hierarchy = self._build_hierarchy(y)
        self._train_level(X, y, 0)
        return self

    def _predict_level(self, X, level, prefix=''):
        if prefix not in self.classifiers:
            return np.array([prefix] * len(X))

        clf = self.classifiers[prefix]
        le = self.label_encoders[prefix]
        
        y_pred = clf.predict(X)
        labels = le.inverse_transform(y_pred)
        
        results = []
        for i, label in enumerate(labels):
            new_prefix = f"{prefix}/{label}" if prefix else label
            results.append(self._predict_level(X[i:i+1], level + 1, new_prefix)[0])
        
        return np.array(results)

    def predict(self, X):
        return self._predict_level(X, 0)