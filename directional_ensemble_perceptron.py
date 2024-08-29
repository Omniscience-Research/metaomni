import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DirectionalEnsemblePerceptron(BaseEstimator, ClassifierMixin):
    """
    Directional Ensemble Perceptron (DEP) classifier.

    Parameters:
    -----------
    n_estimators : int, default=10
        The number of perceptrons in the ensemble.
    learning_rate : float, default=0.01
        The learning rate for weight updates.
    max_iter : int, default=1000
        Maximum number of iterations for training.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes:
    -----------
    classes_ : array, shape (n_classes,)
        The classes labels.
    n_classes_ : int
        The number of classes.
    weights_ : array, shape (n_estimators, n_features + 1)
        The weights for each perceptron in the ensemble.
    """

    def __init__(self, n_estimators=10, learning_rate=0.01, max_iter=1000, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the DEP model according to the given training data.

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
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ != 2:
            raise ValueError("DirectionalEnsemblePerceptron only supports binary classification.")

        n_samples, n_features = X.shape

        # Initialize weights
        rng = np.random.RandomState(self.random_state)
        self.weights_ = rng.randn(self.n_estimators, n_features + 1)

        # Add bias term to X
        X = np.hstack((X, np.ones((n_samples, 1))))

        # Convert y to {-1, 1}
        y = np.where(y == self.classes_[0], -1, 1)

        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                ensemble_output = np.sign(np.sum(np.sign(np.dot(self.weights_, xi))))
                if yi * ensemble_output <= 0:
                    errors += 1
                    update = self.learning_rate * yi * xi
                    for j in range(self.n_estimators):
                        if yi * np.dot(self.weights_[j], xi) <= 0:
                            self.weights_[j] += update

            if errors == 0:
                break

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
        X = np.hstack((X, np.ones((X.shape[0], 1))))  # Add bias term

        ensemble_output = np.sign(np.sum(np.sign(np.dot(self.weights_, X.T)), axis=0))
        return np.where(ensemble_output == -1, self.classes_[0], self.classes_[1])