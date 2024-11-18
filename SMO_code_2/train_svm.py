import numpy as np
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
    svm.fit()
    print("Training complete.")

    # Test predictions
    sample_idx = np.random.randint(0, len(X))
    sample_feature = X[sample_idx]
    sample_label = y_binary[sample_idx]
    prediction = svm.predict(sample_feature)
    print(f"Sample prediction: {prediction}, Ground truth: {sample_label}")

if __name__ == "__main__":
    main()
