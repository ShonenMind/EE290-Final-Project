import numpy as np
import pandas as pd

def select_optimal_beams(H_S, g_MT, codebook, P_S, noise_power, c_S_indices):
    N_S = len(H_S)  # Number of SBSs
    optimal_beam_indices = []
    sinr_values = []

    # Beams used by other SBSs
    beams_for_interference = [codebook[c_S_indices[j]] for j in range(N_S)]

    for k in range(N_S):
        best_sinr = -np.inf
        best_beam_index = c_S_indices[k]

        for i, c in enumerate(codebook):
            signal_power = np.abs(g_MT @ H_S[k] @ c)**2 * P_S

            interference_power = 0
            for j in range(N_S):
                if j != k:
                    c_j = beams_for_interference[j]
                    interference_power += np.abs(g_MT @ H_S[j] @ c_j)**2 * P_S

            sinr = signal_power / (interference_power + noise_power)

            if sinr > best_sinr:
                best_sinr = sinr
                best_beam_index = i

        optimal_beam_indices.append(best_beam_index)
        sinr_values.append(best_sinr)

    return optimal_beam_indices, sinr_values

def generate_codebook(N_C, N_SBS, D_S, sigma):
    codebook = []
    codebook_angles = np.linspace(-np.pi/2, np.pi/2, N_C)
    for angle in codebook_angles:
        steering_vector = np.exp(1j * sigma * D_S * np.sin(angle) * np.arange(N_SBS)) / np.sqrt(N_SBS)
        codebook.append(steering_vector)
    return codebook, codebook_angles

def generate_mmwave_data(num_samples, filename='mmwave_data.csv'):
    # Constants and Parameters
    c = 3e8
    frequency = 28e9
    wavelength = c / frequency
    sigma = 2 * np.pi / wavelength
    D_MT = D_S = wavelength / 2
    L = 2
    N_C = 8
    N_SBS = 32
    N_MT = 2
    P_S_dBm = 20
    P_S = 10 ** ((P_S_dBm - 30) / 10)
    noise_power = 1e-10

    lambda_S = 1e-4
    R = 100

    N_S = int(np.floor(lambda_S * np.pi * R**2))
    N_S = max(N_S, 1)

    # Generate codebook once for all samples
    codebook, codebook_angles = generate_codebook(N_C, N_SBS, D_S, sigma)
    
    data = []

    for _ in range(num_samples):
        d_S = (np.random.randn(N_S) + 1j * np.random.randn(N_S)) / np.sqrt(2)

        c_S_indices = [np.random.randint(0, N_C) for _ in range(N_S)]
        c_S = [codebook[i] for i in c_S_indices]

        alpha_S = (np.random.randn(N_S, L) + 1j * np.random.randn(N_S, L)) / np.sqrt(2)
        phi_MT = np.random.uniform(-np.pi/2, np.pi/2, (N_S, L))
        phi_S = np.random.uniform(-np.pi/2, np.pi/2, (N_S, L))

        gamma = np.sqrt(N_SBS * N_MT / L)
        H_S = []
        for k in range(N_S):
            H_S_k = np.zeros((N_MT, N_SBS), dtype=complex)
            for l in range(L):
                a_MT = np.exp(1j * sigma * D_MT * np.sin(phi_MT[k, l]) * np.arange(N_MT)) / np.sqrt(N_MT)
                a_S = np.exp(1j * sigma * D_S * np.sin(phi_S[k, l]) * np.arange(N_SBS)) / np.sqrt(N_SBS)
                H_S_k += alpha_S[k, l] * np.outer(a_MT, np.conj(a_S))
            H_S_k *= gamma
            H_S.append(H_S_k)

        H_S_norms = np.array([np.linalg.norm(H_S_k, 'fro') for H_S_k in H_S])
        H_S_singular_values = np.array([np.linalg.svd(H_S_k, compute_uv=False) for H_S_k in H_S])
        H_S_eigenvalues = np.array([np.linalg.eigvalsh(H_S_k.conj().T @ H_S_k) for H_S_k in H_S])

        theta = np.random.uniform(0, 2 * np.pi, N_MT)
        g_MT = np.exp(1j * theta)

        optimal_beams, sinr_values = select_optimal_beams(H_S, g_MT, codebook, P_S, noise_power, c_S_indices)
        asr = np.sum(np.log2(1 + np.array(sinr_values)))

        sample = {
            'd_S_real': np.real(d_S).tolist(),
            'd_S_imag': np.imag(d_S).tolist(),
            'H_S_norms': H_S_norms.tolist(),
            'H_S_singular_values': H_S_singular_values.tolist(),
            'H_S_eigenvalues': H_S_eigenvalues.tolist(),
            'N_S': N_S,
            'P_S_dBm': P_S_dBm,
            'lambda_S': lambda_S,
            'R': R,
            'L': L,
            'N_MT': N_MT,
            'N_SBS': N_SBS,
            'N_C': N_C,
            'noise_power': noise_power,
            'g_MT_real': np.real(g_MT).tolist(),
            'g_MT_imag': np.imag(g_MT).tolist(),
            'theta': theta.tolist(),
            'phi_MT': phi_MT.tolist(),
            'phi_S': phi_S.tolist(),
            'alpha_S_real': np.real(alpha_S).tolist(),
            'alpha_S_imag': np.imag(alpha_S).tolist(),
            'optimal_beams': optimal_beams,
            'sinr_values': sinr_values,
            'asr': asr,
        }
        data.append(sample)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f'Dataset saved to {filename}')

# Example usage:
generate_mmwave_data(num_samples=30000)