
import array_config as array_config

import numpy as np


# Array Configuration
rows = array_config.rows
cols = array_config.cols
mic_spacing = array_config.mic_spacing  # meters - based on center freq
num_mics = array_config.num_mics

def generate_mic_coordinates():
    '''
        these grid coordinate are from the perspective of in front the array with (0,0) at the top left
        the reference is from the center of the array
    '''
    mic_coords = np.zeros((num_mics, 3))  # Initialize coordinates array

    idx = 0
    for row in range(rows):
        for col in range(cols):
            # Calculate the position centered around (0, 0)
            x = (col - (cols - 1) / 2) * mic_spacing
            y = ((rows - 1) / 2 - row) * mic_spacing  # Flip the sign to correct y-coordinate
            z = 0  # For a 2D array, z is always 0
            mic_coords[idx] = [x, y, z]
            idx += 1

    # print(mic_coords)
    return mic_coords




def generate_mic_coordinates_arrays(config):
    '''
        these grid coordinate are from the perspective of in front the array with (0,0) at the top left
        the reference is from the center of the array
    '''
    mic_coords = np.zeros((config.num_mics, 3))  # Initialize coordinates array

    idx = 0

    for idx, col in enumerate(range(config.cols)):
        # Calculate the position centered around (0, 0)
        x = (col - (cols - 1) / 2) * mic_spacing
        y = 0  # For a 1D array, y is always 0
        z = 0  # For a 1D & 2D array, z is always 0
        mic_coords[idx] = [x, y, z]

    return mic_coords


if __name__ == '__main__':
    mic_coordinates = generate_mic_coordinates()

    # Print the coordinates as a 4x12 grid
    print("Microphone coordinates in a 1x12 grid:")
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            x, y, z = mic_coordinates[idx]
            print(f'({x:.3f}, {y:.3f})', end='\t ')
        print()