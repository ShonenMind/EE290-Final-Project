import numpy as np
from scipy import constants

class MmWaveDatasetGenerator:
    def __init__(self, num_samples=30000):
        # System parameters as given in Table I
        self.sbs_density = 1e-4        # λS: 1 × 10−4m−2
        self.sbs_power_db = 20         # PS: 20 dBm
        self.num_paths = 2             # L: 2 propagation paths
        self.num_mt_antennas = 2       # NMT: 2 MT antennas
        self.num_sbs_antennas = 32     # NSBS: 32 SBS antennas
        self.radius = 100              # R: 100m radius
        self.num_codebook_beams = 8    # NC: 8 candidate vectors
        
        self.sbs_power = 10 ** ((self.sbs_power_db - 30) / 10)  # Convert dBm to Watts
        
        self.num_sbs = int(np.floor(self.sbs_density * np.pi * self.radius**2))
        
        self.frequency = 28e9          # 28 GHz (mmWave frequency)
        self.wavelength = constants.c / self.frequency
        self.k = 2 * np.pi / self.wavelength
        self.d_mt = self.wavelength / 2    # Half-wavelength antenna spacing
        self.d_sbs = self.wavelength / 2
        
        self.codebook = self._generate_codebook()
        
        self.num_samples = num_samples
        
    def _generate_codebook(self):
        """Generate DFT-based codebook with NC=8 candidate vectors"""
        codebook = np.zeros((self.num_sbs_antennas, self.num_codebook_beams), dtype=complex)
        angles = np.linspace(-np.pi/2, np.pi/2, self.num_codebook_beams, endpoint=False)
        
        for i, angle in enumerate(angles):
            array_response = np.exp(1j * self.k * self.d_sbs * 
                                  np.arange(self.num_sbs_antennas) * np.sin(angle))
            codebook[:, i] = array_response / np.sqrt(self.num_sbs_antennas)
        
        return codebook

    def _generate_channel(self):
        """Generate channel matrix H_S,k based on Saleh-Valenzuela model"""
        aoa = np.random.uniform(-np.pi/2, np.pi/2, self.num_paths)
        aod = np.random.uniform(-np.pi/2, np.pi/2, self.num_paths)
        
        alpha = (np.random.normal(0, 1/np.sqrt(2), self.num_paths) + 
                1j * np.random.normal(0, 1/np.sqrt(2), self.num_paths))
        
        a_mt = np.zeros((self.num_mt_antennas, self.num_paths), dtype=complex)
        a_sbs = np.zeros((self.num_sbs_antennas, self.num_paths), dtype=complex)
        
        for l in range(self.num_paths):
            a_mt[:, l] = np.exp(1j * self.k * self.d_mt * 
                               np.arange(self.num_mt_antennas) * np.sin(aoa[l])) / \
                               np.sqrt(self.num_mt_antennas)
            
            a_sbs[:, l] = np.exp(1j * self.k * self.d_sbs * 
                                np.arange(self.num_sbs_antennas) * np.sin(aod[l])) / \
                                np.sqrt(self.num_sbs_antennas)
        
        gamma = np.sqrt(self.num_sbs_antennas * self.num_mt_antennas / self.num_paths)
        H = gamma * sum(alpha[l] * np.outer(a_mt[:, l], a_sbs[:, l].conj()) 
                       for l in range(self.num_paths))
        
        return H, aoa, aod, alpha

    def _calculate_snr(self, H, beam):
        """Calculate SNR for given channel and beam"""
        noise_power = 1e-13
        H_reshaped = H.reshape(self.num_mt_antennas, self.num_sbs_antennas)
        received_signal = H_reshaped @ beam
        received_power = np.sum(np.abs(received_signal) ** 2) * self.sbs_power
        return received_power / noise_power

    def _find_optimal_beam(self, H):
        """Find optimal beam from codebook that maximizes SNR"""
        snr_values = np.zeros(self.num_codebook_beams)
        
        for i in range(self.num_codebook_beams):
            beam = self.codebook[:, i].reshape(-1, 1)
            snr_values[i] = self._calculate_snr(H, beam)
            
        return np.argmax(snr_values)

    def generate_dataset(self):
        """Generate dataset with features and labels"""
        features = []
        labels = []
        
        for _ in range(self.num_samples):
            H, aoa, aod, alpha = self._generate_channel()
            
            feature = np.concatenate([
                aoa.real,              # AoA for each path
                aod.real,              # AoD for each path
                alpha.real,            # Real part of path gains
                alpha.imag,            # Imaginary part of path gains
                H.reshape(-1).real,    # Real part of channel matrix
                H.reshape(-1).imag     # Imaginary part of channel matrix
            ])
            
            optimal_beam = self._find_optimal_beam(H)
            
            features.append(feature)
            labels.append(optimal_beam)
        
        return np.array(features), np.array(labels)

def main():
    generator = MmWaveDatasetGenerator(num_samples=30000)
    
    X, y = generator.generate_dataset()
    
    print("DEBUGGING")
    print(f"Dataset generated:")
    print(f"Feature shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Number of unique beam indices: {len(np.unique(y))}")
    
    np.save('mmwave_features.npy', X)
    np.save('mmwave_labels.npy', y)

if __name__ == "__main__":
    main()