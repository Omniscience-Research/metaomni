import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

class SimilarityWeightedSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', similarity_threshold=0.5):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.similarity_threshold = similarity_threshold
        self.svm = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True)
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.svm.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before using 'predict'.")

        similarities = rbf_kernel(X_test, self.X_train, gamma=self.svm._gamma)
        
        weighted_predictions = []
        for i, x in enumerate(X_test):
            similar_indices = np.where(similarities[i] >= self.similarity_threshold)[0]
            
            if len(similar_indices) > 0:
                similar_X = self.X_train[similar_indices]
                similar_y = self.y_train[similar_indices]
                weights = similarities[i, similar_indices]
                
                local_svm = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True)
                local_svm.fit(similar_X, similar_y, sample_weight=weights)
                
                weighted_pred = local_svm.predict([x])[0]
            else:
                weighted_pred = self.svm.predict([x])[0]
            
            weighted_predictions.append(weighted_pred)

        return np.array(weighted_predictions)

    def predict_proba(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before using 'predict_proba'.")

        similarities = rbf_kernel(X_test, self.X_train, gamma=self.svm._gamma)
        
        weighted_probas = []
        for i, x in enumerate(X_test):
            similar_indices = np.where(similarities[i] >= self.similarity_threshold)[0]
            
            if len(similar_indices) > 0:
                similar_X = self.X_train[similar_indices]
                similar_y = self.y_train[similar_indices]
                weights = similarities[i, similar_indices]
                
                local_svm = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True)
                local_svm.fit(similar_X, similar_y, sample_weight=weights)
                
                weighted_proba = local_svm.predict_proba([x])[0]
            else:
                weighted_proba = self.svm.predict_proba([x])[0]
            
            weighted_probas.append(weighted_proba)

        return np.array(weighted_probas)