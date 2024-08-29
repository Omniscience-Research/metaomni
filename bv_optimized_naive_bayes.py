import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score

class OptimizedNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, class_prior=None, fit_prior=True, var_smoothing=1e-9):
        self.alpha = alpha
        self.class_prior = class_prior
        self.fit_prior = fit_prior
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_ = X.shape[1]
        
        # Initialize parameters
        self.class_count_ = np.zeros(len(self.classes_), dtype=np.float64)
        self.feature_count_ = np.zeros((len(self.classes_), self.n_features_), dtype=np.float64)
        self.feature_log_prob_ = np.zeros((len(self.classes_), self.n_features_), dtype=np.float64)
        self.class_log_prior_ = np.zeros(len(self.classes_), dtype=np.float64)
        
        # Count occurrences of classes and features
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_count_[i] = X_c.shape[0]
            self.feature_count_[i] = np.sum(X_c, axis=0) + self.alpha
        
        # Compute class priors
        if self.fit_prior:
            self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
        elif self.class_prior is not None:
            self.class_log_prior_ = np.log(self.class_prior)
        else:
            self.class_log_prior_ = np.zeros(len(self.classes_))
        
        # Compute feature log probabilities
        smoothed_fc = self.feature_count_ + self.var_smoothing
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        
        # Optimize bias-variance tradeoff
        self._optimize_bias_variance(X, y)
        
        return self

    def _optimize_bias_variance(self, X, y):
        # Perform cross-validation to find optimal alpha and var_smoothing
        alphas = np.logspace(-3, 3, 7)
        var_smoothings = np.logspace(-12, -6, 7)
        
        best_score = -np.inf
        best_alpha = self.alpha
        best_var_smoothing = self.var_smoothing
        
        for alpha in alphas:
            for var_smoothing in var_smoothings:
                self.alpha = alpha
                self.var_smoothing = var_smoothing
                scores = cross_val_score(self, X, y, cv=5)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
                    best_var_smoothing = var_smoothing
        
        self.alpha = best_alpha
        self.var_smoothing = best_var_smoothing
        
        # Refit the model with optimized parameters
        self.fit(X, y)

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute log probabilities
        jll = self._joint_log_likelihood(X)
        
        # Return predictions
        return self.classes_[np.argmax(jll, axis=1)]

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(len(self.classes_)):
            jointi = np.log(self.class_count_[i]) + self.class_log_prior_[i]
            n_ij = X * self.feature_log_prob_[i]
            joint_log_likelihood.append(jointi + np.sum(n_ij, axis=1))
        
        return np.array(joint_log_likelihood).T