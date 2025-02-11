import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import PCA


def remove_pca_component(wav_path, n_components=10, component_to_remove=1, output_path=None, ):
    """
    Loads a WAV file, performs PCA on the magnitude of its spectrogram,
    zeroes out the specified principal component, and reconstructs an
    audio file with that component removed. Saves the new audio to disk.

    Parameters
    ----------
    wav_path : str
        Path to the input WAV file.
    component_to_remove : int
        1-based index of the principal component to remove.
        E.g. 1 means "remove PC1," 2 means "remove PC2," etc.
    output_path : str, optional
        Where to save the modified audio file.
        If None, it saves under the same directory with "_removedPCx.wav" suffix.
    """

    # 1. Load the audio
    y, sr = librosa.load(wav_path, sr=None)

    # 2. Compute STFT -> complex matrix D, shape: (frequency_bins, time_frames)
    D = librosa.stft(y)

    # Separate magnitude and phase
    magnitude = np.abs(D)
    phase = np.angle(D)

    # 3. Prepare data for PCA:
    # Each time frame is a row in `data_for_pca`, so shape = (time_frames, frequency_bins).
    data_for_pca = magnitude.T  # shape = (time_frames, freq_bins)

    # 4. Fit PCA on the magnitude
    pca = PCA(n_components=n_components)  # By default, it keeps min(n_samples, n_features) components
    pca_result = pca.fit_transform(data_for_pca)
    # pca_result shape = (time_frames, num_components)

    # Ensure the requested component is valid (1-based index)
    max_components = pca_result.shape[1]
    if not (1 <= component_to_remove <= max_components):
        raise ValueError(f"component_to_remove must be between 1 and {max_components}.")

    # 5. Zero out the chosen component
    pca_result_modified = pca_result.copy()
    pca_result_modified[:, component_to_remove - 1] = 0.0  # 0-based indexing

    # 6. Inverse transform to get back modified magnitude
    magnitude_modified = pca.inverse_transform(pca_result_modified)
    # Transpose back to (freq_bins, time_frames)
    magnitude_modified = magnitude_modified.T

    # 7. Reconstruct the complex spectrogram using the original phase
    D_modified = magnitude_modified * np.exp(1j * phase)

    # 8. Inverse STFT to recover the time-domain signal
    y_modified = librosa.istft(D_modified)

    # 9. Save the new audio file
    if output_path is None:
        # Auto-generate an output path in the same directory
        # E.g., if wav_path = "my_audio.wav" and component_to_remove=1,
        # output = "my_audio_removedPC1.wav"
        import os
        base, ext = os.path.splitext(wav_path)
        output_path = f"{base}_removedPC{component_to_remove}{ext}"

    sf.write(output_path, y_modified, sr)
    print(f"Modified audio saved to: {output_path}")


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29'
    filename_calibration = 'calibration bf-12m'
    filepath_calibration = f'{base_path}/Calibration/{filename_calibration}.wav'
    # Try removing the 1st principal component:
    num_comps = 10
    for i in range(2, num_comps):
        remove_pca_component(filepath_calibration, num_comps, component_to_remove=i)

    # Or remove the 2nd principa
