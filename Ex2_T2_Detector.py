from acoustic.audio import Audio
from process import average_spectrum
from Detection.pca import PCA_Calculator
from Detection.detector import Detector
from noise_reduction import noise_reduction_filter
from high_pass import high_pass_filter
from low_pass import low_pass_filter
from normalize import normalize
from down_sample import downsample
from save_to_wav import save_to_wav


import matplotlib.pyplot as plt
import numpy as np
import pickle
import os




def process_audio(audio, processes):
    # std_threshold = 0.5
    # top_cutoff_freq = 170
    # bottom_cutoff_freq = 100
    normalize_percentage = 50
    # new_sample_rate = 2000


    for key, value in processes.items():  # Extract key-value dynamically
        if key == 'lp':
            for i in range(5):
                print('Pass Low Freq')
                audio.data = low_pass_filter(audio, value, order=8)

        if key == 'nr':
            print('Reducing Noise')
            audio.data = noise_reduction_filter(audio, value)

        # print('Normalizing')
        # audio.data = normalize(audio, normalize_percentage)

    # print('Passing High Freq')
    # audio.data = high_pass_filter(audio, bottom_cutoff_freq)

    # print('Reducing Noise')
    # audio.data = noise_reduction_filter(audio)
    # print('Down Sampling')
    # audio.data = downsample(audio, new_sample_rate)
    # audio.sample_rate = new_sample_rate



    return audio

def calculate_anomalies(filename, process, processes):
    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 2 2-24/Test Files'

    filepath = f'{base_path}/{filename}.wav'
    audio_calibration = Audio(filepath=filepath, num_channels=1)
    # audio_calibration.data = audio_calibration.data[1610*audio_calibration.sample_rate:2103*audio_calibration.sample_rate]
    audio_calibration.data = audio_calibration.data[587 * audio_calibration.sample_rate:648 * audio_calibration.sample_rate]

    if process:
        audio_calibration = process_audio(audio_calibration, processes)

    pca_detector = PCA_Calculator(num_components=6)
    detector = Detector()

    chunk_size = 1 * audio_calibration.sample_rate
    overlap = chunk_size #// 2  # 50% overlap
    start = 0

    while start < len(audio_calibration.data):
        end = start + chunk_size
        components = pca_detector.process_chunk(audio_calibration.data[start:end])
        detector.calculate_baseline(components)
        start += overlap

    detector.baseline_calculated = True

    filepath = f'{base_path}/{filename}.wav'
    audio = Audio(filepath=filepath, num_channels=1)

    if process:
        audio = process_audio(audio, processes)

    anomalies = []

    overlap = chunk_size
    start = 0

    detector.anomaly_threshold = 1

    while start < len(audio.data):
        end = start + chunk_size
        components = pca_detector.process_chunk(audio.data[start:end])
        anomalies.append(detector.detect_anomalies(components))
        start += overlap

    return anomalies

def create_plot(filename, plot_title, save_tag, threshold, process, processes, recalculate):
    SAVE_FILE = f"anomaly_files/anomalies_{save_tag}.pkl"
    RECALCULATE = recalculate  # Change to True if you need to redo calculations

    if os.path.exists(SAVE_FILE) and not RECALCULATE:
        with open(SAVE_FILE, "rb") as f:
            anomalies = pickle.load(f)
        print("loaded saved anomalies")
    else:
        anomalies = calculate_anomalies(filename, process, processes)
        with open(SAVE_FILE, "wb") as f:
            pickle.dump(anomalies, f)
        print("calculated and saved anomalies")

    targets = [
        82, 126, 170, 213, 256,  # tank
        300, 343, 388, 432, 477,  # truck
        653, 697, 741, 785, 828,
        871, 915, 959, 1003, 1047,
        1178, 1221, 1265, 1310, 1354,
        1398, 1441, 1485, 1529, 1573,
        1705, 1749, 1793, 1837, 1881,
        1925, 1968, 2012, 2056, 2099
    ]

    targets_2 = [521, 1090, 1617, 2141]

    colors = ['purple', 'blue', 'green', 'teal']

    flight_time = (50, 2175)
    threshold = threshold

    plt.figure(figsize=(24, 4))
    plt.title(f'PCA Detector: {plot_title} - Experiment 2: 70, 80, 90, 100m Altitude')
    vehicle_labels = {0: 'Vehicles 70m', 1: 'Vehicles 80m', 2: 'Vehicles 90m', 3: 'Vehicles 100m'}
    added_labels = set()

    plt.plot(anomalies[flight_time[0]:flight_time[1]], label='Anomalies')

    for i, target in enumerate(targets):
        color_idx = (i // 10) % len(colors)  # Ensure correct mapping to colors
        color = colors[color_idx]

        label = vehicle_labels.get(color_idx, 'Vehicles') if vehicle_labels.get(color_idx, 'Vehicles') not in added_labels else None
        if label: added_labels.add(label)

        x_pos = target - flight_time[0]
        plt.axvspan(x_pos - 10, x_pos + 10, color=color, alpha=0.2, label=label)
        plt.axvline(x=x_pos, color=color, linestyle=':', alpha=0.5) # , label=label

    for i, target in enumerate(targets_2):
        x_pos = target - flight_time[0]

        label = 'Gas Generator' if 'Gas Generator' not in added_labels else None
        if label: added_labels.add(label)
        plt.axvspan(x_pos - 10, x_pos + 10, color='gray', alpha=0.2, label=label)
        plt.axvline(x=x_pos, color='gray', linestyle=':', alpha=0.5) # , label=label

    plt.axhline(y=threshold, color='red', linestyle=':', alpha=0.8, label='Threshold')


    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Remove duplicates
    plt.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    plt.xlabel("time")
    plt.ylabel("anomalies")
    plt.tight_layout(pad=1)

    # plt.show()
    plt.savefig(f"PCA Detector {save_tag}.png", dpi=500)


if __name__ == '__main__':

    filters = [180]
    std = 0.5

    recalculate = False

    for filter in filters:

        process = True
        processes = {'lp': filter}

        filenames = ['E2_T2_Full_Raw_Ch1', 'E2_T2_Full_BF12', 'E2_T2_Full_BM'] #
        plot_titles = [f'Raw Ch1 - LP: {filter}_5x',
                       f'Beamed 12 Mics - LP: {filter}_5x',
                       f'Beam Mixture - LP: {filter}_5x']
        save_tags = [f'RawCh1_LP{filter}_5x', f'BF12m_LP{filter}_5x', f'BMix_LP{filter}_5x']
        thresholds = [19, 18, 17]

        for filename, plot_title, save_tag, threshold in zip(filenames, plot_titles, save_tags, thresholds):
            create_plot(filename, plot_title, save_tag, threshold, process, processes, recalculate)














