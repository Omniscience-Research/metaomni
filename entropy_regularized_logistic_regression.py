import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class EntropyRegLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, lambda_=1.0, max_iter=1000, tol=1e-4):
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, w, X, y):
        m = X.shape[0]
        z = np.dot(X, w)
        h = self._sigmoid(z)
        log_likelihood = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        entropy_reg = -self.lambda_ * np.sum(w * np.log(np.abs(w) + 1e-8))
        return log_likelihood + entropy_reg

    def _gradient(self, w, X, y):
        m = X.shape[0]
        z = np.dot(X, w)
        h = self._sigmoid(z)
        grad_log_likelihood = np.dot(X.T, (h - y)) / m
        grad_entropy_reg = -self.lambda_ * (np.log(np.abs(w) + 1e-8) + 1)
        return grad_log_likelihood + grad_entropy_reg

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)

        if len(self.classes_) != 2:
            raise ValueError("EntropyRegLogisticRegression only supports binary classification.")

        n_features = X.shape[1]
        self.coef_ = np.zeros(n_features)

        opt_result = minimize(
            fun=self._loss,
            x0=self.coef_,
            args=(X, y),
            method='L-BFGS-B',
            jac=self._gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )

        self.coef_ = opt_result.x
        self.intercept_ = 0.0  # For compatibility with scikit-learn
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        z = np.dot(X, self.coef_)
        proba = self._sigmoid(z)
        return np.column_stack([1 - proba, proba])

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)