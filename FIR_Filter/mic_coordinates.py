

import numpy as np



def generate_mic_coordinates(config):
    '''
        these grid coordinate are from the perspective of in front the array with (0,0) at the top left
        the reference is from the center of the array
    '''
    mic_coords = np.zeros((config.num_mics, 3))  # Initialize coordinates array

    idx = 0

    for idx, col in enumerate(range(config.cols)):
        # Calculate the position centered around (0, 0)
        x = (col - (config.cols - 1) / 2) * config.spacing
        y = 0  # For a 1D array, y is always 0
        z = 0  # For a 1D & 2D array, z is always 0
        mic_coords[idx] = [x, y, z]

    return mic_coords


if __name__ == '__main__':
    pass