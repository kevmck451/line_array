import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def extract_and_plot_pca(wav_path):
    """
    Loads a WAV file, computes its spectrogram, performs PCA to extract
    the top 10 components, and visualizes them in a scatter plot
    (all against the 1st component).
    """

    # 1. Load audio (sr=None to preserve original sample rate)
    y, sr = librosa.load(wav_path, sr=None)

    # 2. Compute spectrogram (STFT) -> shape: (freq_bins, time_frames)
    stft_result = librosa.stft(y, n_fft=2**12)
    spectrogram = np.abs(stft_result)

    # 3. Prepare data for PCA: transpose so shape is (time_frames, freq_bins)
    data_for_pca = spectrogram.T

    # 4. Perform PCA for top 10 components
    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(data_for_pca)
    # pca_result shape: (time_frames, 10)
    # Each column = one principal component

    # 5. Scatter plot: PC2..PC10 vs PC1
    plt.figure(figsize=(8, 6))
    x_vals = pca_result[:, 0]  # First principal component

    # Choose colors for 9 components (PC2..PC10)
    colors = plt.cm.tab10(np.linspace(0, 1, 9))

    for i in range(1, 10):
        plt.scatter(x_vals, pca_result[:, i],
                    alpha=0.25,
                    color=colors[i - 1],
                    label=f'PC{i + 1}')

    plt.title("Top 10 Principal Components of Spectrogram\n(PC1 vs. Others)")
    plt.xlabel("PC1")
    plt.ylabel("PC2..PC10 (overlaid)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Set your .wav file path here:
    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29'
    filename_calibration = 'calibration bf-12m'
    filepath_calibration = f'{base_path}/Calibration/{filename_calibration}.wav'
    extract_and_plot_pca(filepath_calibration)
