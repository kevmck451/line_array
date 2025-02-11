import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import soundfile as sf  # for saving .wav files


def reconstruct_single_component(
        original_spectrogram_magnitude,
        pca_obj,
        pca_scores,
        component_index
):
    """
    Reconstructs the spectrogram magnitude using only ONE component
    from the PCA decomposition.

    Parameters
    ----------
    original_spectrogram_magnitude : np.ndarray, shape=(freq_bins, frames)
        The original magnitude spectrogram (for reference to get shape).

    pca_obj : PCA object (from sklearn.decomposition)
        Fitted PCA object containing the `components_` and `mean_`.

    pca_scores : np.ndarray, shape=(frames, n_components)
        The projected data in the PCA space (scores).

    component_index : int
        Which single PCA component to reconstruct from (0-based).

    Returns
    -------
    recon_magnitude : np.ndarray, shape=(freq_bins, frames)
        Reconstructed magnitude spectrogram using only the selected component.
    """

    # Zero out all components except the one we want
    pca_only_one = np.zeros_like(pca_scores)  # shape = (frames, n_components)
    pca_only_one[:, component_index] = pca_scores[:, component_index]

    # Use PCA's inverse transform to get back to the original dimension
    recon_data = pca_obj.inverse_transform(pca_only_one)  # shape=(frames, freq_bins)
    # Transpose to (freq_bins, frames) to match original spectrogram shape
    recon_magnitude = recon_data.T

    # In principle, some values can be negative; clamp to zero for magnitude
    recon_magnitude = np.maximum(recon_magnitude, 0.0)

    return recon_magnitude


def process_wav_and_save_single_components(wav_path):
    """
    1) Loads a wav file
    2) Computes the STFT (magnitude & phase)
    3) Performs PCA on the magnitude
    4) Reconstructs audio from each of the top 10 PCA components (one at a time)
    5) Saves each reconstruction as a separate .wav file
    """

    # -------------------------
    # 1. Load the WAV
    # -------------------------
    y, sr = librosa.load(wav_path, sr=None)

    # -------------------------
    # 2. Compute STFT
    # -------------------------
    # shape = (freq_bins, frames)
    stft_result = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude, phase = np.abs(stft_result), np.exp(1j * np.angle(stft_result))

    # -------------------------
    # 3. PCA on magnitude
    # -------------------------
    # Flatten as (frames, freq_bins) for PCA
    data_for_pca = magnitude.T  # shape=(frames, freq_bins)

    pca = PCA(n_components=10)
    pca_scores = pca.fit_transform(data_for_pca)  # shape=(frames, 10)

    # Print explained variance for curiosity:
    print("Explained variance ratio (top 10):", pca.explained_variance_ratio_)

    # -------------------------
    # 4. Loop over top 10 components
    # -------------------------
    for k in range(10):
        # Reconstruct magnitude from the k-th component only
        recon_mag_k = reconstruct_single_component(magnitude, pca, pca_scores, k)

        # Combine the reconstructed magnitude with the original phase
        recon_stft_k = recon_mag_k * phase  # shape=(freq_bins, frames)

        # -------------------------
        # 5. iSTFT to go back to time domain
        # -------------------------
        # Use the same hop_length, window, etc. as your forward STFT
        y_recon_k = librosa.istft(recon_stft_k, hop_length=512)

        # -------------------------
        # 6. Save the result
        # -------------------------
        out_filename = f"component_{k + 1}.wav"
        sf.write(out_filename, y_recon_k, sr)
        print(f"Saved single-component reconstruction to {out_filename}")

    # Optionally, return the PCA object and scores if needed
    return pca, pca_scores


# ------------------------------
# Example usage in an IDE/script
# ------------------------------
if __name__ == "__main__":
    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29'
    filename_calibration = 'calibration bf-12m'
    filepath_calibration = f'{base_path}/Calibration/{filename_calibration}.wav'
    pca_obj, pca_scores = process_wav_and_save_single_components(filepath_calibration)
