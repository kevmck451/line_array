
from generate_fir_coeffs import generate_fir_coeffs


from scipy.signal import convolve
import numpy as np



def generate_beamformed_audio_iterative(mapped_audio_data, thetas, temp_F, mic_coords):
    beamformed_audio_data = np.zeros((len(thetas), mapped_audio_data.shape[2]+200))
    for i, theta in enumerate(thetas):
        beamformed_audio_data[i, :] = generate_beamformed_audio(mapped_audio_data, theta, temp_F, mic_coords)
        print(f'Completed BF Data for {theta}')
        print('-' * 40)

    return beamformed_audio_data



def generate_beamformed_audio(mapped_audio_data, theta, temp_F, mic_coords):
    print(f'generating fir coefficients: {theta}')
    fir_coeffs = generate_fir_coeffs(mic_coords, theta, temp_F)
    print('beamforming audio')
    print('-' * 40)

    return beamform(mapped_audio_data, fir_coeffs)



def beamform(audio_data, fir_coeffs):
    rows, cols, num_samples = audio_data.shape
    num_coeffs = fir_coeffs.shape[2]

    assert rows * cols == fir_coeffs.shape[0] * fir_coeffs.shape[1], "Mismatch between audio data and FIR coefficients shape."

    beamformed_data = np.zeros(num_samples + num_coeffs - 1)

    for row_index in range(rows):
        for col_index in range(cols):
            filtered_signal = convolve(audio_data[row_index, col_index, :], fir_coeffs[row_index, col_index, :], mode='full')
            beamformed_data += filtered_signal

    return beamformed_data