import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm

class IterativeSMOTrainer:
    def __init__(self, C: float = 1.0, tol: float = 1e-3, max_passes: int = 10):
        """
        Initialize SMO trainer for beam selection
        
        Args:
            C: Regularization parameter
            tol: Tolerance for KKT conditions
            max_passes: Maximum number of iterations without alpha changes
        """
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.alphas = None
        self.b = 0
        self.hyperplanes = {}
        
    def _calculate_kernel(self, X: np.ndarray) -> np.ndarray:
        """Precompute the linear kernel matrix"""
        return X @ X.T
    
    def _select_second_alpha(self, i: int, E_i: float, E: np.ndarray) -> int:
        """
        Select second alpha that maximizes |E_i - E_j|
        """
        non_zero_C = np.where((self.alphas != 0) & (self.alphas != self.C))[0]
        if len(non_zero_C) > 1:
            # Choose the index j that maximizes |E_i - E_j|
            delta_E = np.abs(E_i - E[non_zero_C])
            j = non_zero_C[np.argmax(delta_E)]
            return j if j != i else (i + 1) % len(self.alphas)
        else:
            # If no suitable j found, choose a random index not equal to i
            j = np.random.randint(0, len(self.alphas))
            while j == i:
                j = np.random.randint(0, len(self.alphas))
            return j
    def train_single_hyperplane(self, X: np.ndarray, y: np.ndarray, 
                                m: int, n: int) -> Tuple[np.ndarray, float]:
        """Train a single hyperplane between codewords m and n"""
        num_samples = X.shape[0]
        self.alphas = np.zeros(num_samples)
        self.b = 0
        
        # Precompute the kernel matrix
        K = self._calculate_kernel(X)
        
        # Initialize the error cache with float64 type
        E = -y.astype(float)  # Ensure E is of type float64
        
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(num_samples):
                E_i = (np.dot((self.alphas * y), K[:, i]) + self.b) - y[i]
                
                # Check KKT conditions
                if ((y[i] * E_i < -self.tol and self.alphas[i] < self.C) or 
                    (y[i] * E_i > self.tol and self.alphas[i] > 0)):
                    
                    # Select second alpha
                    j = self._select_second_alpha(i, E_i, E)
                    E_j = (np.dot((self.alphas * y), K[:, j]) + self.b) - y[j]
                    
                    alpha_i_old = self.alphas[i].copy()
                    alpha_j_old = self.alphas[j].copy()
                    
                    # Compute L and H
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alphas[j] -= y[j] * (E_i - E_j) / eta
                    # Clip alpha_j
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # Compute b1 and b2
                    b1 = self.b - E_i - y[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - \
                        y[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - \
                        y[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    
                    # Update b
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    
                    # Update error cache
                    delta_alpha_i = self.alphas[i] - alpha_i_old
                    delta_alpha_j = self.alphas[j] - alpha_j_old
                    E += (y[i] * delta_alpha_i) * K[:, i] + (y[j] * delta_alpha_j) * K[:, j]
                    
                    num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        return self.alphas, self.b


    def train(self, X: np.ndarray, y: np.ndarray, NC: int) -> Dict:
        """
        Train all hyperplanes for multiclass SVM.

        Args:
            X: Training features
            y: Training labels
            NC: Number of classes (beam indices)

        Returns:
            Dict containing hyperplanes for all class pairs.
        """
        self.hyperplanes = {}
        for m in range(NC):
            for n in range(m + 1, NC):
                # Select data for current pair of classes
                mask = np.logical_or(y == m, y == n)
                X_sub = X[mask]
                y_sub = y[mask]
                # Convert labels to binary
                y_sub = np.where(y_sub == m, 1, -1)

                # Train hyperplane for the pair
                alphas, b = self.train_single_hyperplane(X_sub, y_sub, m, n)
                sv_mask = alphas > 1e-5
                self.hyperplanes[(m, n)] = {
                    'alphas': alphas[sv_mask],
                    'b': b,
                    'support_vectors': X_sub[sv_mask],
                    'support_vector_labels': y_sub[sv_mask]
                }
        return self.hyperplanes
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class for new samples"""
        if not self.hyperplanes:
            raise ValueError("Model not trained yet")
        
        num_samples = X.shape[0]
        num_classes = max(max(k) for k in self.hyperplanes.keys()) + 1
        votes = np.zeros((num_samples, num_classes))
        
        # Each hyperplane votes
        for (m, n), hyperplane in self.hyperplanes.items():
            sv = hyperplane['support_vectors']  # Shape: (n_sv, n_features)
            sv_y = hyperplane['support_vector_labels']  # Shape: (n_sv,)
            alphas = hyperplane['alphas']  # Shape: (n_sv,)
            b = hyperplane['b']
            
            # Compute decision function
            # Compute kernel between X and support vectors
            # K_test: Shape (num_samples, n_sv)
            K_test = X @ sv.T  # Matrix multiplication
            
            # Compute the decision values
            decision = (K_test @ (alphas * sv_y)) + b  # Shape: (num_samples,)
            
            # Vote based on decision
            votes[decision >= 0, m] += 1
            votes[decision < 0, n] += 1
        
        return np.argmax(votes, axis=1)
