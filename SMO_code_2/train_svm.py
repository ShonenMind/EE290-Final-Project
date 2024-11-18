import numpy as np
import matplotlib.pyplot as plt
from SMO import SVM  # Assuming the SVM class is saved in a file named `svm.py`

def load_mmwave_dataset(filepath):
    """Load the dataset from the .npy file."""
    data = np.load(filepath, allow_pickle=True).item()
    features = data['features']
    labels = data['labels']
    return features, labels

def preprocess_labels(labels):
    """
    Convert the beam indices into a binary classification problem for SVM.
    For simplicity, classify beams into two classes: class 1 (beam index < NC/2) 
    and class -1 (beam index >= NC/2).
    """
    unique_labels = np.unique(labels)
    mid_index = len(unique_labels) // 2
    binary_labels = np.where(labels < unique_labels[mid_index], 1, -1)
    return binary_labels

def calculate_accuracy(predictions, true_labels):
    """Calculate accuracy as the percentage of correct predictions."""
    return np.mean(predictions == true_labels) * 100

def main():
    # Load the dataset
    dataset_path = 'SMO_code_2/mmwave_dataset.npy'
    X, y = load_mmwave_dataset(dataset_path)
    print(f"Dataset loaded. Feature shape: {X.shape}, Label shape: {y.shape}")

    # Preprocess labels for binary classification
    y_binary = preprocess_labels(y)
    print(f"Labels converted to binary classification. Unique labels: {np.unique(y_binary)}")

    # Initialize and train the SVM
    svm = SVM(X, y_binary, C=1, kernel='linear', max_iter=300)
    print("Training the SVM...")

    loss_history = []
    accuracy_history = []

    for iteration in range(svm.max_iter):
        # Perform one iteration of SMO
        svm.fit()

        # Calculate loss (L2 regularization term)
        loss = 0.5 * np.sum(svm.w ** 2) if svm.is_linear_kernel else 0.0
        loss_history.append(loss)

        # Predictions on the training data
        predictions = np.sign([svm.predict(x) for x in X])

        # Calculate accuracy
        accuracy = calculate_accuracy(predictions, y_binary)
        accuracy_history.append(accuracy)

        # Print metrics for the current iteration
        print(f"Iteration {iteration + 1}/{svm.max_iter}: Loss = {loss:.4f}, Accuracy = {accuracy:.2f}%")

    print("Training complete.")

    # Plot the metrics
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Loss")
    plt.plot(accuracy_history, label="Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
