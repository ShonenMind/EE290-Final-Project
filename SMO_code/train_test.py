import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SMO import IterativeSMOTrainer
import ast

class ComplexDataProcessor:
    def __init__(self):
        pass  # No scalers needed here

    def process_array_string(self, arr_string):
        """Convert string representation of array to numpy array"""
        arr = np.array(ast.literal_eval(arr_string))
        return arr.flatten()

    def process_data(self, df):
        """Process the complex data and create feature vector and target variable"""
        n_samples = len(df)

        d_S_real = np.array([self.process_array_string(x) for x in df['d_S_real']])
        d_S_imag = np.array([self.process_array_string(x) for x in df['d_S_imag']])

        g_MT_real = np.array([self.process_array_string(x) for x in df['g_MT_real']])
        g_MT_imag = np.array([self.process_array_string(x) for x in df['g_MT_imag']])

        theta = np.array([self.process_array_string(x) for x in df['theta']])

        phi_MT = np.array([self.process_array_string(x).flatten() for x in df['phi_MT']])
        phi_S = np.array([self.process_array_string(x).flatten() for x in df['phi_S']])

        phi_MT_sin = np.sin(phi_MT)
        phi_MT_cos = np.cos(phi_MT)
        phi_S_sin = np.sin(phi_S)
        phi_S_cos = np.cos(phi_S)

        alpha_S_real = np.array([self.process_array_string(x).flatten() for x in df['alpha_S_real']])
        alpha_S_imag = np.array([self.process_array_string(x).flatten() for x in df['alpha_S_imag']])

        # Process H_S_norms
        H_S_norms = np.array([self.process_array_string(x) for x in df['H_S_norms']])

        # Process H_S_singular_values
        H_S_singular_values = np.array([self.process_array_string(x).flatten() for x in df['H_S_singular_values']])

        # Process H_S_eigenvalues
        H_S_eigenvalues = np.array([self.process_array_string(x).flatten() for x in df['H_S_eigenvalues']])

        # Process sinr_values
        sinr_values = np.array([self.process_array_string(x) for x in df['sinr_values']])

        # Process asr
        asr = df['asr'].values.reshape(-1, 1)  # Reshape to (n_samples, 1)

        # Process scalar features
        scalar_features = df[['P_S_dBm', 'lambda_S', 'noise_power']].values

        # Concatenate all features
        features = np.hstack([
            d_S_real, d_S_imag,
            g_MT_real, g_MT_imag,
            theta,
            phi_MT, phi_S,
            phi_MT_sin, phi_MT_cos,
            phi_S_sin, phi_S_cos,
            alpha_S_real, alpha_S_imag,
            H_S_norms,
            H_S_singular_values,
            H_S_eigenvalues,
            sinr_values,
            asr,
            scalar_features,
        ])

        # Process target variable - optimal beams
        optimal_beams = df['optimal_beams'].apply(lambda x: self.process_array_string(x).astype(int))
        targets = np.array([x for x in optimal_beams])

        return features, targets

def load_and_preprocess_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the mmWave dataset"""
    # Load data
    df = pd.read_csv(file_path)

    # Process the data using ComplexDataProcessor
    processor = ComplexDataProcessor()
    X, y = processor.process_data(df)

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Since y contains optimal beams for multiple SBSs,
    # we'll consider only the first SBS for simplicity
    y = y[:, 0]  # Take the first optimal beam index for each sample

    return X, y

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate performance metrics"""
    accuracy = np.mean(y_true == y_pred)
    misclassification = 1 - accuracy
    
    classes = np.unique(y_true)
    per_class_accuracy = {}
    for c in classes:
        mask = y_true == c
        per_class_accuracy[f'class_{c}'] = np.mean(y_pred[mask] == y_true[mask])
    
    return {
        'accuracy': accuracy,
        'misclassification_rate': misclassification,
        'per_class_accuracy': per_class_accuracy
    }

def main():
    # Load and preprocess data
    X, y = load_and_preprocess_data('mmwave_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = IterativeSMOTrainer(C=1.0, tol=1e-3, max_passes=10)
    NC = len(np.unique(y))  # Number of beam choices
    hyperplanes = model.train(X_train, y_train, NC)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    metrics = evaluate_model(y_test, y_pred)
    
    # Print results
    print("\nModel Performance:")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Misclassification Rate: {metrics['misclassification_rate']:.4f}")
    print("\nPer-class Accuracy:")
    for class_name, accuracy in metrics['per_class_accuracy'].items():
        print(f"{class_name}: {accuracy:.4f}")

if __name__ == "__main__":
    main()