import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel

class SimilarityWeightedLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma='scale', max_iter=100, tol=1e-4):
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol

    def _similarity_weights(self, X, X_train):
        if self.gamma == 'scale':
            gamma = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            gamma = 1.0 / X.shape[1]
        else:
            gamma = self.gamma
        
        return rbf_kernel(X, X_train, gamma=gamma)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, w, X, y, S):
        z = X.dot(w)
        log_likelihood = np.sum(S * (y * np.log(self._sigmoid(z)) + (1 - y) * np.log(1 - self._sigmoid(z))))
        regularization = 0.5 * self.C * np.sum(w ** 2)
        return -log_likelihood + regularization

    def _gradient(self, w, X, y, S):
        z = X.dot(w)
        gradient = X.T.dot(S * (self._sigmoid(z) - y)) + self.C * w
        return gradient

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("SimilarityWeightedLogisticRegression only supports binary classification.")

        self.X_ = X
        self.y_ = y

        n_samples, n_features = X.shape
        self.S_ = self._similarity_weights(X, X)

        initial_w = np.zeros(n_features)
        
        res = minimize(
            fun=self._loss,
            x0=initial_w,
            args=(X, y, self.S_),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )

        self.coef_ = res.x
        self.intercept_ = 0  # For simplicity, we're not using a separate intercept term

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        S_test = self._similarity_weights(X, self.X_)
        z = X.dot(self.coef_)
        
        weighted_probs = S_test.dot(self._sigmoid(self.X_.dot(self.coef_)))
        normalized_probs = weighted_probs / np.sum(S_test, axis=1)[:, np.newaxis]
        
        return np.column_stack((1 - normalized_probs, normalized_probs))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]