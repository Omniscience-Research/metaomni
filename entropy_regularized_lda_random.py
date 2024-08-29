import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class EntropyRegLDARandomness(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, entropy_reg=0.1, random_state=None):
        self.alpha = alpha
        self.entropy_reg = entropy_reg
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]

        # Initialize random number generator
        self.rng_ = np.random.RandomState(self.random_state)

        # Compute class means and covariance
        self.class_means_ = []
        self.class_cov_ = np.zeros((self.n_features_, self.n_features_))

        for c in self.classes_:
            X_c = X[y == c]
            self.class_means_.append(np.mean(X_c, axis=0))
            self.class_cov_ += np.cov(X_c, rowvar=False)

        self.class_means_ = np.array(self.class_means_)
        self.class_cov_ /= self.n_classes_

        # Add regularization to covariance matrix
        self.class_cov_ += self.alpha * np.eye(self.n_features_)

        # Compute inverse of covariance matrix
        self.inv_cov_ = np.linalg.inv(self.class_cov_)

        # Compute class priors
        self.class_priors_ = np.bincount(y) / len(y)

        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute discriminant function for each class
        discriminant = []
        for i, mean in enumerate(self.class_means_):
            d = np.dot(X, np.dot(self.inv_cov_, mean)) - 0.5 * np.dot(mean, np.dot(self.inv_cov_, mean))
            d += np.log(self.class_priors_[i])
            discriminant.append(d)

        discriminant = np.array(discriminant).T

        # Apply entropy regularization
        probs = softmax(discriminant / self.entropy_reg, axis=1)

        # Add randomness
        random_probs = self.rng_.random(probs.shape)
        random_probs /= np.sum(random_probs, axis=1, keepdims=True)

        # Combine probabilities
        final_probs = (1 - self.entropy_reg) * probs + self.entropy_reg * random_probs

        # Return class with highest probability
        return self.classes_[np.argmax(final_probs, axis=1)]

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute discriminant function for each class
        discriminant = []
        for i, mean in enumerate(self.class_means_):
            d = np.dot(X, np.dot(self.inv_cov_, mean)) - 0.5 * np.dot(mean, np.dot(self.inv_cov_, mean))
            d += np.log(self.class_priors_[i])
            discriminant.append(d)

        discriminant = np.array(discriminant).T

        # Apply entropy regularization
        probs = softmax(discriminant / self.entropy_reg, axis=1)

        # Add randomness
        random_probs = self.rng_.random(probs.shape)
        random_probs /= np.sum(random_probs, axis=1, keepdims=True)

        # Combine probabilities
        final_probs = (1 - self.entropy_reg) * probs + self.entropy_reg * random_probs

        return final_probs