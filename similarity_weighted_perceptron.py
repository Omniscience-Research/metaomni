import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class SimilarityWeightedPerceptron(BaseEstimator, ClassifierMixin):
    """
    Similarity-Weighted Perceptron (SWP) classifier.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for weight updates.
    n_iterations : int, default=1000
        The number of iterations over the training data.
    similarity_threshold : float, default=0.5
        The threshold for considering two samples similar.

    Attributes:
    -----------
    classes_ : array, shape (n_classes,)
        The classes labels.
    weights_ : array, shape (n_features,)
        The weight vector.
    bias_ : float
        The bias term.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, similarity_threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.similarity_threshold = similarity_threshold

    def fit(self, X, y):
        """
        Fit the SWP model according to the given training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns:
        --------
        self : object
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        n_samples, n_features = X.shape

        self.weights_ = np.zeros(n_features)
        self.bias_ = 0

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                y_pred = self._predict_sample(X[i])
                error = y[i] - y_pred

                if error != 0:
                    similarity_weights = self._compute_similarity_weights(X, X[i])
                    self.weights_ += self.learning_rate * error * np.dot(similarity_weights, X)
                    self.bias_ += self.learning_rate * error

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y : array, shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)

        return np.array([self._predict_sample(x) for x in X])

    def _predict_sample(self, x):
        """Predict the class label for a single sample."""
        activation = np.dot(self.weights_, x) + self.bias_
        return 1 if activation >= 0 else 0

    def _compute_similarity_weights(self, X, x):
        """Compute similarity weights for a given sample."""
        similarities = np.sum(X * x, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(x))
        return np.where(similarities >= self.similarity_threshold, 1, 0)