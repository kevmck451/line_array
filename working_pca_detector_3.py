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
    normalize_percentage = 100
    # new_sample_rate = 2000

    for key, value in processes.items():  # Extract key-value dynamically
        if key == 'lp':
            print('Pass Low Freq')
            audio.data = low_pass_filter(audio, value, order=8)

            # print('Normalizing')
            # audio.data = normalize(audio, normalize_percentage)

    # print('Reducing Noise')
    # audio.data = noise_reduction_filter(audio, std_threshold)
    # print('Passing High Freq')
    # audio.data = high_pass_filter(audio, bottom_cutoff_freq)

    # print('Reducing Noise')
    # audio.data = noise_reduction_filter(audio)
    # print('Down Sampling')
    # audio.data = downsample(audio, new_sample_rate)
    # audio.sample_rate = new_sample_rate



    return audio

def calculate_anomalies(filename, process, processes):
    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29/Test Files'

    # filename = 'Test1_RAW-ch1'
    # filename = 'Test1_B12'
    # filename = 'Test1_BEns'
    # filename = 'Test1_B12 Pro_1'


    filepath = f'{base_path}/{filename}.wav'
    audio_calibration = Audio(filepath=filepath, num_channels=1)
    audio_calibration.data = audio_calibration.data[1610*audio_calibration.sample_rate:2103*audio_calibration.sample_rate]
    # print(audio_calibration)
    # audio_calibration.waveform_rms_overlay(display=True)
    # average_spectrum(audio_calibration, display=True)

    # pca_detector = PCA_Calculator(nperseg=2**12, num_components=10)
    # components = pca_detector.process_chunk(audio_calibration.data)
    # print(components.shape)
    # for i in range(1, components.shape[0]):
    #     plt.scatter(components[0], components[i], alpha=0.1)
    #
    # plt.show()

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
    # print(detector.baseline_means)
    # print(detector.baseline_stds)
    # plt.plot(detector.baseline_means)
    # plt.show()


    filepath = f'{base_path}/{filename}.wav'
    audio = Audio(filepath=filepath, num_channels=1)
    # print(audio)
    # audio.waveform_rms_overlay(display=True)
    # average_spectrum(audio, display=True)

    # pca_detector = PCA_Calculator(nperseg=32768, num_components=10)
    # components = pca_detector.process_chunk(audio.data[(20*50000):-(600*50000)])
    # print(components.shape)
    # plt.plot(components)
    # plt.show()

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

def create_plot(filename, plot_title, save_tag, threshold, process, processes):
    SAVE_FILE = f"anomaly_files/anomalies_{save_tag}.pkl"
    RECALCULATE = False  # Change to True if you need to redo calculations

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
        75, 118, 161, 203, 246, # truck
        290, 333, 377, 421, 464, # tank
        637, 681, 724, 767, 810,
        853, 896, 939, 982, 1025,
        1156, 1200, 1244, 1287, 1330,
        1374, 1417, 1460, 1503, 1546]

    targets_2 = [ 507, 551, 1069, 1113, 1590, 1632 ]

    colors = ['purple', 'blue', 'green']

    flight_time = (50, 2000)
    threshold = threshold

    plt.figure(figsize=(24, 4))
    plt.title(f'PCA Detector: {plot_title} - Experiment 1: 30, 40, 50m Altitude')
    vehicle_labels = {0: 'Vehicles 10m', 1: 'Vehicles 20m', 2: 'Vehicles 30m'}
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

        label = 'TS & WN' if 'TS & WN' not in added_labels else None
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

    process = True
    processes = {'lp': 130}

    filenames = ['Test1_RAW-ch1', 'Test1_B12', 'Test1_BEns']
    plot_titles = ['Raw Ch1 - LP: 130', 'Beamed 12 Mics - LP: 130', 'Beam Mixture - LP: 130']
    save_tags = ['RawCh1_LP130', 'BF12m_LP130', 'BMix_LP130']
    thresholds = [17.5, 18.5, 20]


    # filename = 'Test1_B12 Pro_1'

    for filename, plot_title, save_tag, threshold in zip(filenames, plot_titles, save_tags, thresholds):
        create_plot(filename, plot_title, save_tag, threshold, process, processes)














