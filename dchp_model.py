import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import StandardScaler

class DCHybridPerceptron(BaseEstimator, ClassifierMixin):
    """
    Discrete-Continuous Hybrid Perceptron (DCHP) classifier.

    Parameters:
    -----------
    learning_rate : float, default=0.01
        The learning rate for weight updates.
    n_iterations : int, default=1000
        The number of iterations for training.
    random_state : int, default=None
        Seed for random number generation.

    Attributes:
    -----------
    w_ : array, shape (n_features,)
        Weights after fitting.
    b_ : float
        Bias after fitting.
    losses_ : list
        Training loss history.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the DCHP model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w_ = np.random.randn(n_features)
        self.b_ = 0
        self.losses_ = []

        # Separate discrete and continuous features
        discrete_mask = np.all(np.mod(X, 1) == 0, axis=0)
        continuous_mask = ~discrete_mask

        # Standardize continuous features
        self.scaler_ = StandardScaler()
        X_continuous = self.scaler_.fit_transform(X[:, continuous_mask])
        X_discrete = X[:, discrete_mask]

        # Combine preprocessed features
        X_processed = np.hstack((X_discrete, X_continuous))

        for _ in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X_processed, self.w_) + self.b_
            predictions = self._step_function(linear_output)

            # Compute loss
            loss = np.mean((y - predictions) ** 2)
            self.losses_.append(loss)

            # Backward pass
            error = y - predictions
            self.w_ += self.learning_rate * np.dot(X_processed.T, error)
            self.b_ += self.learning_rate * np.sum(error)

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
        check_is_fitted(self, ['w_', 'b_'])
        X = check_array(X)

        # Separate discrete and continuous features
        discrete_mask = np.all(np.mod(X, 1) == 0, axis=0)
        continuous_mask = ~discrete_mask

        # Preprocess features
        X_continuous = self.scaler_.transform(X[:, continuous_mask])
        X_discrete = X[:, discrete_mask]
        X_processed = np.hstack((X_discrete, X_continuous))

        linear_output = np.dot(X_processed, self.w_) + self.b_
        return self._step_function(linear_output)

    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)