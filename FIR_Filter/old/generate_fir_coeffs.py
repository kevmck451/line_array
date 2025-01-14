from old.mic_coordinates import generate_mic_coordinates
from old.time_delays import calculate_time_delays
from old.fir_filter import create_fir_filters


def generate_fir_coeffs(mic_coords, theta, phi, temp_F, sample_rate=48000, num_taps=201):
    phi = 0
    time_delays = calculate_time_delays(mic_coords, theta, phi, temp_F, sample_rate)
    fir_filter = create_fir_filters(time_delays, num_taps)

    return fir_filter


if __name__ == '__main__':
    # angle is from behind array looking forward. center is (0, 0)
    theta = 0  # elevation angle
    phi = 0  # azimuth angle
    temperature_F = 70  # temperature in Fahrenheit
    sample_rate = 48000

    # uses info from array_config
    mic_coordinates = generate_mic_coordinates()
    print(mic_coordinates.shape)

    time_delays = calculate_time_delays(mic_coordinates, theta, phi, temperature_F, sample_rate)
    print(time_delays.shape)
    num_taps = 201 # should be odd

    fir_filter = create_fir_filters(time_delays, num_taps)
    print(fir_filter.shape)
    # print(fir_filter)

    print(fir_filter)




