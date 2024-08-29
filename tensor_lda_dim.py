import numpy as np
import tensorflow as tf

class TensorLDADim:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.means_ = None
        self.priors_ = None
        self.covariance_ = None
        self.scaling_ = None

    def fit(self, X_train, y_train):
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

        n_samples, n_features = X_train.shape
        classes = tf.unique(y_train)[0]
        n_classes = len(classes)

        if self.n_components is None:
            self.n_components = min(n_features, n_classes - 1)

        # Compute class means
        self.means_ = tf.stack([tf.reduce_mean(X_train[y_train == c], axis=0) for c in classes])

        # Compute class priors
        self.priors_ = tf.stack([tf.reduce_sum(tf.cast(y_train == c, tf.float32)) / n_samples for c in classes])

        # Compute within-class covariance
        X_centered = X_train - tf.gather(self.means_, y_train)
        self.covariance_ = tf.reduce_mean(tf.matmul(X_centered[:, :, tf.newaxis], X_centered[:, tf.newaxis, :]), axis=0)

        # Compute between-class covariance
        overall_mean = tf.reduce_mean(X_train, axis=0)
        B = tf.reduce_sum(self.priors_[:, tf.newaxis, tf.newaxis] * 
                          tf.matmul((self.means_ - overall_mean)[:, :, tf.newaxis],
                                    (self.means_ - overall_mean)[:, tf.newaxis, :]), axis=0)

        # Solve the generalized eigenvalue problem
        eigvals, eigvecs = tf.linalg.eigh(B, self.covariance_)

        # Sort eigenvectors by decreasing eigenvalues
        idx = tf.argsort(eigvals, direction='DESCENDING')
        eigvecs = tf.gather(eigvecs, idx, axis=1)

        # Select the top n_components eigenvectors
        self.scaling_ = eigvecs[:, :self.n_components]

        return self

    def predict(self, X_test):
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

        # Project the data
        X_transformed = tf.matmul(X_test, self.scaling_)

        # Compute distances to class means in the transformed space
        means_transformed = tf.matmul(self.means_, self.scaling_)
        distances = tf.sqrt(tf.reduce_sum((X_transformed[:, tf.newaxis, :] - means_transformed[tf.newaxis, :, :]) ** 2, axis=2))

        # Predict the class with the smallest distance
        predictions = tf.argmin(distances, axis=1)

        return predictions.numpy()

# Example usage:
if __name__ == "__main__":
    # Generate some random data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 3, 100)

    # Split the data into train and test sets
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Create and fit the TensorLDADim model
    lda = TensorLDADim(n_components=2)
    lda.fit(X_train, y_train)

    # Make predictions
    y_pred = lda.predict(X_test)

    # Print the results
    print("True labels:", y_test)
    print("Predicted labels:", y_pred)