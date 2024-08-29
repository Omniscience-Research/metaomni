import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class FractalLogisticRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, fractal_dim=1.5, max_iter=1000, tol=1e-4):
        self.fractal_dim = fractal_dim
        self.max_iter = max_iter
        self.tol = tol

    def _fractal_sigmoid(self, z):
        return 1 / (1 + np.exp(-np.power(np.abs(z), self.fractal_dim - 1) * np.sign(z)))

    def _fractal_log_likelihood(self, w, X, y):
        z = np.dot(X, w)
        log_likelihood = np.sum(y * np.log(self._fractal_sigmoid(z)) + 
                                (1 - y) * np.log(1 - self._fractal_sigmoid(z)))
        return -log_likelihood

    def _fractal_log_likelihood_gradient(self, w, X, y):
        z = np.dot(X, w)
        gradient = np.dot(X.T, (self._fractal_sigmoid(z) - y))
        return gradient

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("FractalLogisticRegressor only supports binary classification.")

        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)

        # Add intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Optimize using L-BFGS-B algorithm
        result = minimize(
            fun=self._fractal_log_likelihood,
            x0=self.coef_,
            args=(X, y),
            method='L-BFGS-B',
            jac=self._fractal_log_likelihood_gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )

        self.coef_ = result.x
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        # Add intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        z = np.dot(X, np.hstack((self.intercept_, self.coef_)))
        probas = self._fractal_sigmoid(z)
        return np.vstack((1 - probas, probas)).T

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]