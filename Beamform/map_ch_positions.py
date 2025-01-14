



import numpy as np


def map_channels_to_positions(audio_data, array_configuration):
    num_samples = audio_data.shape[1]
    mapped_data = np.zeros((array_configuration.rows, array_configuration.cols, num_samples))

    for ch_index in range(array_configuration.num_mics):
        mic_x, mic_y = array_configuration.mic_positions[ch_index]
        mapped_data[mic_x, ch_index, :] = audio_data[mic_y, :]

    return mapped_data



if __name__ == '__main__':
    map_channels_to_positions()