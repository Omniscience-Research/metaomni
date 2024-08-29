import numpy as np
from scipy.stats import norm
from collections import defaultdict

class HybridNaiveBayes:
    def __init__(self, continuous_features=None):
        self.continuous_features = continuous_features or []
        self.class_priors = {}
        self.feature_params = defaultdict(lambda: defaultdict(dict))
        self.classes = None

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        n_samples, n_features = X_train.shape

        # Calculate class priors
        for c in self.classes:
            self.class_priors[c] = np.sum(y_train == c) / n_samples

        # Calculate feature parameters
        for feature in range(n_features):
            if feature in self.continuous_features:
                for c in self.classes:
                    feature_values = X_train[y_train == c, feature]
                    self.feature_params[feature][c] = {
                        'mean': np.mean(feature_values),
                        'std': np.std(feature_values)
                    }
            else:
                for c in self.classes:
                    feature_values = X_train[y_train == c, feature]
                    unique_values, counts = np.unique(feature_values, return_counts=True)
                    probabilities = counts / len(feature_values)
                    self.feature_params[feature][c] = dict(zip(unique_values, probabilities))

        return self

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            class_scores = {}
            for c in self.classes:
                class_score = np.log(self.class_priors[c])
                for feature, value in enumerate(sample):
                    if feature in self.continuous_features:
                        mean = self.feature_params[feature][c]['mean']
                        std = self.feature_params[feature][c]['std']
                        if std > 0:
                            class_score += np.log(norm.pdf(value, mean, std))
                    else:
                        prob = self.feature_params[feature][c].get(value, 1e-10)  # Laplace smoothing
                        class_score += np.log(prob)
                class_scores[c] = class_score
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)