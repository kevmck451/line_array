

from audio import Audio
from save_to_wav import save_to_wav

import noisereduce as nr
from pathlib import Path
import numpy as np


def noise_reduction_filter(audio_object, std_threshold=1.5):

    reduced_noise_data = np.zeros_like(audio_object.data)

    if audio_object.num_channels == 1:
        reduced_noise_data = nr.reduce_noise(
            y=audio_object.data,
            sr=audio_object.sample_rate,
            stationary=True,  # stationary noise reduction
            freq_mask_smooth_hz=2000,  # default is 500Hz
            time_mask_smooth_ms=1000,  # default is 50ms
            use_tqdm=True,  # show terminal progress bar
            n_std_thresh_stationary=std_threshold,  # default is 1.5
            n_jobs=-1  # use all available cores
        )
        reduced_noise_data = reduced_noise_data[50000:-50000]

    else:
        for channel in range(audio_object.data.shape[0]):
            reduced_noise_data[channel, :] = nr.reduce_noise(
                y=audio_object.data[channel, :],
                sr=audio_object.sample_rate,
                stationary=True, # stationary noise reduction
                freq_mask_smooth_hz=2000, # default is 500Hz
                time_mask_smooth_ms=1000, # default is 50ms
                use_tqdm=True, # show terminal progress bar
                n_std_thresh_stationary = std_threshold, # default is 1.5
                n_jobs = -1 # use all available cores
            )
        reduced_noise_data = reduced_noise_data[:, 50000:-50000]

    reduced_noise_data = np.clip(reduced_noise_data, -1.0, 1.0)


    return reduced_noise_data


if __name__ == '__main__':

    # filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/2 FOSSN/Data/Tests/5_outdoor_testing/07-12-2024_02-49-21_chunk_1.wav'
    # audio = Audio(filepath=filepath, num_channels=48)

    # filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Angel_Mount_Data/2 Flights/A6_Flight_2.wav'
    # audio = Audio(filepath=filepath, num_channels=4)

    filepath = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29/Test1_B12.wav'
    audio = Audio(filepath=filepath, num_channels=1)

    trim_size = 50 * 50000
    front, middle, end = audio.data[:trim_size], audio.data[trim_size:-trim_size], audio.data[-trim_size:]

    audio.data = middle
    std_threshold = 1
    filtered_audio = noise_reduction_filter(audio, std_threshold=std_threshold)

    repackaged_data = np.concatenate([front, filtered_audio.data, end])

    # print(f'Max: {np.max(filtered_data)}')
    # print(f'Min: {np.min(filtered_data)}')

    # Ensure the filtered data shape matches the original
    # assert filtered_data.shape == audio.data.shape, "Filtered data shape does not match original data shape"

    # Create the new filename with "_HPF" suffix
    original_path = Path(filepath)
    new_filename = original_path.stem + f"_NR{std_threshold}" + original_path.suffix
    new_filepath = str(original_path.parent / new_filename)

    # Save the filtered audio to the new file
    save_to_wav(repackaged_data, audio.sample_rate, audio.num_channels, new_filepath)
