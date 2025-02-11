
from audio import Audio
from save_to_wav import save_to_wav

from acoustic.noise_reduction import noise_reduction_filter
from acoustic.low_pass import low_pass_filter
from acoustic.normalize import normalize
import numpy as np
from pathlib import Path



if __name__ == '__main__':

    filepath = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29/Test1_B12.wav'
    audio = Audio(filepath=filepath, num_channels=1)

    trim_size = 45 * 50000
    trim_size_end = 100 * 50000
    front, middle, end = audio.data[:trim_size], audio.data[trim_size:-trim_size_end], audio.data[-trim_size_end:]

    audio.data = middle

    print(f'Max: {np.max(audio.data)}')
    print(f'Min: {np.min(audio.data)}')
    audio.data = normalize(audio)
    print(f'Max: {np.max(audio.data)}')
    print(f'Min: {np.min(audio.data)}')
    audio.data = low_pass_filter(audio, cutoff_freq=1000)
    print(f'Max: {np.max(audio.data)}')
    print(f'Min: {np.min(audio.data)}')
    audio.data = normalize(audio)
    print(f'Max: {np.max(audio.data)}')
    print(f'Min: {np.min(audio.data)}')

    std_threshold = 2
    audio.data = noise_reduction_filter(audio, std_threshold=std_threshold)
    print(f'Max: {np.max(audio.data)}')
    print(f'Min: {np.min(audio.data)}')
    audio.data = normalize(audio)
    print(f'Max: {np.max(audio.data)}')
    print(f'Min: {np.min(audio.data)}')

    audio.data = np.concatenate([front, audio.data, end])

    original_path = Path(filepath)
    new_filename = original_path.stem + f"_NR{std_threshold}" + original_path.suffix
    new_filepath = str(original_path.parent / new_filename)

    # Save the filtered audio to the new file
    save_to_wav(audio.data, audio.sample_rate, audio.num_channels, new_filepath)