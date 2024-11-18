import numpy as np
import pandas as pd

def select_optimal_beams(H_S, g_MT, codebook, P_S, noise_power):
    N_S = len(H_S)  # Number of SBSs
    optimal_beam_indices = []
    sinr_values = []

    for k in range(N_S):
        best_sinr = -np.inf
        best_beam_index = None
        
        for i, c in enumerate(codebook):
            # Compute Signal Power
            signal_power = np.abs(g_MT @ H_S[k] @ c)**2 * P_S
            
            # Compute Interference Power
            interference_power = 0
            for j in range(N_S):
                if j != k:
                    interference_power += np.abs(g_MT @ H_S[j] @ c)**2 * P_S
            
            # Compute SINR
            sinr = signal_power / (interference_power + noise_power)
            
            # Update best beam if SINR is higher
            if sinr > best_sinr:
                best_sinr = sinr
                best_beam_index = i  # Store the index of the beam
        
        optimal_beam_indices.append(best_beam_index)
        sinr_values.append(best_sinr)
    
    return sinr_values

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
        # Generate data streams d_S,k
        d_S = (np.random.randn(N_S) + 1j * np.random.randn(N_S)) / np.sqrt(2)
        d_S_abs = np.abs(d_S)
        d_S_phase = np.angle(d_S)

        c_S = [codebook[np.random.randint(0, N_C)] for _ in range(N_S)]

        # Generate positions of SBSs
        SBS_positions = np.random.uniform(-R, R, (N_S, 2))
        distances = np.linalg.norm(SBS_positions, axis=1)

        # Calculate path loss
        path_loss_exponent = 2  # Free-space path loss
        path_loss = (4 * np.pi * distances / wavelength) ** path_loss_exponent

        # Generate random AoD angles for each SBS
        phi_S = np.random.uniform(-np.pi/2, np.pi/2, N_S)
        # Repeat phi_S for each path
        phi_S = np.tile(phi_S[:, np.newaxis], (1, L))

        # Generate random AoA angles for each SBS
        phi_MT = np.random.uniform(-np.pi/2, np.pi/2, (N_S, L))

        # Generate complex gains Î±_S,k,l
        alpha_S = (np.random.randn(N_S, L) + 1j * np.random.randn(N_S, L)) / np.sqrt(2)
        alpha_S_abs = np.abs(alpha_S)
        alpha_S_phase = np.angle(alpha_S)

        gamma = np.sqrt(N_SBS * N_MT / L)
        H_S = []
        a_S_list = []
        a_MT_list = []
        for k in range(N_S):
            H_S_k = np.zeros((N_MT, N_SBS), dtype=complex)
            a_S_paths = []
            a_MT_paths = []
            for l in range(L):
                a_MT = np.exp(1j * sigma * D_MT * np.sin(phi_MT[k, l]) * np.arange(N_MT)) / np.sqrt(N_MT)
                a_S = np.exp(1j * sigma * D_S * np.sin(phi_S[k, l]) * np.arange(N_SBS)) / np.sqrt(N_SBS)
                H_S_k += alpha_S[k, l] * np.outer(a_MT, np.conj(a_S))
                a_MT_paths.append(a_MT)
                a_S_paths.append(a_S)
            H_S_k *= gamma
            H_S.append(H_S_k)
            a_S_list.append(a_S_paths)
            a_MT_list.append(a_MT_paths)

        # Compute channel matrix features
        H_S_norms = np.array([np.linalg.norm(H_S_k, 'fro') for H_S_k in H_S])
        H_S_singular_values = np.array([np.linalg.svd(H_S_k, compute_uv=False) for H_S_k in H_S])
        H_S_eigenvalues = np.array([np.linalg.eigvalsh(H_S_k.conj().T @ H_S_k) for H_S_k in H_S])

        theta = np.random.uniform(0, 2 * np.pi, N_MT)
        g_MT = np.exp(1j * theta)
        g_MT_abs = np.abs(g_MT)
        g_MT_phase = np.angle(g_MT)

        # Beamforming gains
        beamforming_gains = []
        for k in range(N_S):
            gain = np.abs(c_S[k].conj().T @ a_S_list[k][0])  # Assuming first path
            beamforming_gains.append(gain)

        # Interference indicators
        interference_indicators = []
        for k in range(N_S):
            interference = sum([np.abs(g_MT @ H_S[j] @ c_S[j])**2 for j in range(N_S) if j != k])
            interference_indicators.append(interference)

        # Angle differences
        angle_differences = np.abs(phi_S[:, 0][:, np.newaxis] - codebook_angles[np.newaxis, :])  # Use first path AoD

        # Optimal beam selection based on closest angle
        optimal_beams = []
        for k in range(N_S):
            # Find the codebook angle closest to phi_S[k, 0]
            angle_diff = np.abs(codebook_angles - phi_S[k, 0])
            best_beam_index = np.argmin(angle_diff)
            optimal_beams.append(best_beam_index)

        # Collect codebook angles (same for all samples)
        sample_codebook_angles = codebook_angles.tolist()

        # Compute received signal y_MT
        y_MT = np.zeros(N_MT, dtype=complex)
        for k in range(N_S):
            y_MT += g_MT @ H_S[k] @ c_S[k] * d_S[k] * np.sqrt(P_S)

        # Add noise
        noise = np.sqrt(noise_power / 2) * (np.random.randn(N_MT) + 1j * np.random.randn(N_MT))
        y_MT += noise

        # Compute Zero Forcing equalized signal y_MT_ZF
        G = np.zeros((N_MT, N_S), dtype=complex)
        for k in range(N_S):
            G[:, k] = H_S[k] @ c_S[k] * np.sqrt(P_S)
        W_ZF = np.linalg.pinv(G)
        y_MT_ZF = W_ZF @ y_MT
        y_MT_ZF_real = np.real(y_MT_ZF)
        y_MT_ZF_imag = np.imag(y_MT_ZF)

        # asr = np.sum(np.log2(1 + np.array(sinr_values)))

        # Collect data for this sample
        sample = {
            'd_S_real': np.real(d_S).tolist(),
            'd_S_imag': np.imag(d_S).tolist(),
            'd_S_abs': d_S_abs.tolist(),
            'd_S_phase': d_S_phase.tolist(),
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
            'g_MT_abs': g_MT_abs.tolist(),
            'g_MT_phase': g_MT_phase.tolist(),
            'theta': theta.tolist(),
            'phi_MT': phi_MT.tolist(),
            'phi_S': phi_S.tolist(),
            'phi_MT_sin': np.sin(phi_MT).tolist(),
            'phi_MT_cos': np.cos(phi_MT).tolist(),
            'phi_S_sin': np.sin(phi_S).tolist(),
            'phi_S_cos': np.cos(phi_S).tolist(),
            'alpha_S_real': np.real(alpha_S).tolist(),
            'alpha_S_imag': np.imag(alpha_S).tolist(),
            'alpha_S_abs': alpha_S_abs.tolist(),
            'alpha_S_phase': alpha_S_phase.tolist(),
            'H_S_norms': H_S_norms.tolist(),
            'H_S_singular_values': H_S_singular_values.tolist(),
            'H_S_eigenvalues': H_S_eigenvalues.tolist(),
            'beamforming_gains': beamforming_gains,
            'interference_indicators': interference_indicators,
            'distances': distances.tolist(),
            'path_loss': path_loss.tolist(),
            'angle_differences': angle_differences.tolist(),
            'codebook_angles': sample_codebook_angles,
            'y_MT_ZF_real': y_MT_ZF_real.tolist(),
            'y_MT_ZF_imag': y_MT_ZF_imag.tolist(),
            'optimal_beams': optimal_beams,
            # 'asr': asr,
        }
        data.append(sample)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f'Dataset saved to {filename}')

# Example usage:
generate_mmwave_data(num_samples=30000)