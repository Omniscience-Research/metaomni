import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class MultiScaleSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, n_scales=3, C=1.0, kernel='rbf', random_state=None):
        self.n_scales = n_scales
        self.C = C
        self.kernel = kernel
        self.random_state = random_state
        self.scalers = []
        self.classifiers = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        for scale in range(self.n_scales):
            # Create a scaler for this scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers.append(scaler)

            # Create and train an SVM classifier for this scale
            clf = SVC(C=self.C, kernel=self.kernel, random_state=self.random_state)
            clf.fit(X_scaled, y)
            self.classifiers.append(clf)

            # Reduce the scale (increase complexity) for the next iteration
            X = np.column_stack([X, X[:, :X.shape[1]//2]])

        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = []

        for scale, (scaler, clf) in enumerate(zip(self.scalers, self.classifiers)):
            X_scaled = scaler.transform(X)
            pred = clf.predict(X_scaled)
            predictions.append(pred)

            # Expand X for the next scale, if not the last scale
            if scale < self.n_scales - 1:
                X = np.column_stack([X, X[:, :X.shape[1]//2]])

        # Combine predictions from all scales (majority voting)
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0,
            arr=np.array(predictions)
        )

        return final_predictions

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate a sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_classes=2, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Multi-scale SVM
    multi_svm = MultiScaleSVM(n_scales=3, C=1.0, kernel='rbf', random_state=42)
    multi_svm.fit(X_train, y_train)

    # Make predictions
    y_pred = multi_svm.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")