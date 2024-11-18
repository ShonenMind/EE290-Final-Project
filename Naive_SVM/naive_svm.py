import numpy as np
import matplotlib.pyplot as plt

class NaiveSVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iters=1000):
        self.learning_rate = learning_rate  # Step size for gradient descent
        self.lambda_param = lambda_param    # Regularization parameter
        self.num_iters = num_iters         # Number of iterations
        self.w = None                      # Weights
        self.b = 0                         # Bias term

    def fit(self, X, y):
        """Train the SVM using gradient descent."""
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)

        loss_history = []

        for i in range(self.num_iters):
            # Calculate margin
            margin = y * (np.dot(X, self.w) + self.b)
            
            # Compute gradient for weights and bias
            dw = self.lambda_param * self.w - (np.mean((margin < 1)[:, None] * y[:, None] * X, axis=0))
            db = -np.mean((margin < 1) * y)

            # Gradient descent update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Compute hinge loss for tracking
            loss = self._compute_loss(X, y)
            loss_history.append(loss)

            # Print loss periodically
            if i % (self.num_iters // 10) == 0 or i == self.num_iters - 1:
                print(f"Iteration {i + 1}/{self.num_iters}: Loss = {loss:.4f}")

        return loss_history

    def predict(self, X):
        """Make predictions using the trained SVM."""
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)

    def _compute_loss(self, X, y):
        """Compute hinge loss."""
        hinge_loss = np.maximum(0, 1 - y * (np.dot(X, self.w) + self.b))
        return np.mean(hinge_loss) + 0.5 * self.lambda_param * np.sum(self.w ** 2)

# Training script
def main():
    # Load the dataset
    dataset_path = 'Naive_SVM/mmwave_dataset.npy'
    data = np.load(dataset_path, allow_pickle=True).item()
    X = data['features']
    y = data['labels']

    # Preprocess labels for binary classification
    unique_labels = np.unique(y)
    mid_index = len(unique_labels) // 2
    y_binary = np.where(y < unique_labels[mid_index], 1, -1)

    # Train the Naïve SVM
    svm = NaiveSVM(learning_rate=0.001, lambda_param=0.01, num_iters=10000)
    print("Training Naïve SVM...")
    loss_history = svm.fit(X, y_binary)
    print("Training complete.")

    # Evaluate on the training set
    predictions = svm.predict(X)
    accuracy = np.mean(predictions == y_binary) * 100
    print(f"Training Accuracy: {accuracy:.2f}%")

    # Plot the loss over iterations
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label="Hinge Loss")
    plt.title("Hinge Loss During Training")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
