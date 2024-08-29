import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SimilarityBasedWeakLearnerSelector(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=3, similarity_threshold=0.7):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.similarity_threshold = similarity_threshold
        self.weak_learners = []
        self.selected_learners = []
        
    def _create_weak_learners(self):
        for _ in range(self.n_estimators):
            weak_learner = DecisionTreeClassifier(max_depth=self.max_depth)
            self.weak_learners.append(weak_learner)
    
    def _compute_similarity(self, X1, X2):
        # Compute cosine similarity between two datasets
        X1_norm = np.linalg.norm(X1, axis=1)
        X2_norm = np.linalg.norm(X2, axis=1)
        similarity = np.dot(X1, X2.T) / (X1_norm[:, np.newaxis] * X2_norm[np.newaxis, :])
        return np.mean(similarity)
    
    def _select_weak_learners(self, X, y):
        self.selected_learners = []
        for learner in self.weak_learners:
            # Split the data to compute similarity
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the learner on a subset of data
            learner.fit(X_train, y_train)
            
            # Compute similarity between training and validation sets
            similarity = self._compute_similarity(X_train, X_val)
            
            # If similarity is above threshold, add to selected learners
            if similarity >= self.similarity_threshold:
                self.selected_learners.append(learner)
        
        # If no learners are selected, use all weak learners
        if not self.selected_learners:
            self.selected_learners = self.weak_learners
    
    def fit(self, X, y):
        # Create weak learners if not already created
        if not self.weak_learners:
            self._create_weak_learners()
        
        # Select weak learners based on similarity
        self._select_weak_learners(X, y)
        
        # Train selected weak learners
        for learner in self.selected_learners:
            learner.fit(X, y)
        
        return self
    
    def predict(self, X):
        # Collect predictions from all selected weak learners
        predictions = np.array([learner.predict(X) for learner in self.selected_learners])
        
        # Return the majority vote
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate a random dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = SimilarityBasedWeakLearnerSelector(n_estimators=20, max_depth=3, similarity_threshold=0.7)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")