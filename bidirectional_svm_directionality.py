import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class BidirectionalSVMDirectional(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, 
                 probability=False, tol=1e-3, cache_size=200, class_weight=None, verbose=False, 
                 max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Create two SVM classifiers
        self.svm_forward_ = SVC(
            C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            shrinking=self.shrinking, probability=self.probability, tol=self.tol, 
            cache_size=self.cache_size, class_weight=self.class_weight, verbose=self.verbose,
            max_iter=self.max_iter, decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties, random_state=self.random_state
        )
        
        self.svm_backward_ = SVC(
            C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma, coef0=self.coef0,
            shrinking=self.shrinking, probability=self.probability, tol=self.tol, 
            cache_size=self.cache_size, class_weight=self.class_weight, verbose=self.verbose,
            max_iter=self.max_iter, decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties, random_state=self.random_state
        )

        # Fit the forward SVM
        self.svm_forward_.fit(X, y)

        # Fit the backward SVM with reversed input
        X_reversed = X[:, ::-1]
        self.svm_backward_.fit(X_reversed, y)

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Make predictions using both SVMs
        forward_pred = self.svm_forward_.predict(X)
        backward_pred = self.svm_backward_.predict(X[:, ::-1])

        # Combine predictions (you can modify this part based on your specific requirements)
        final_pred = np.where(forward_pred == backward_pred, forward_pred, self.classes_[0])

        return final_pred

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Get probabilities from both SVMs
        forward_proba = self.svm_forward_.predict_proba(X)
        backward_proba = self.svm_backward_.predict_proba(X[:, ::-1])

        # Combine probabilities (average in this case, but you can modify based on your needs)
        final_proba = (forward_proba + backward_proba) / 2

        return final_proba