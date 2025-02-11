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



def open_anomalies(save_tag):
    SAVE_FILE = f"anomaly_files/anomalies_{save_tag}.pkl"
    RECALCULATE = False  # Change to True if you need to redo calculations

    if os.path.exists(SAVE_FILE) and not RECALCULATE:
        with open(SAVE_FILE, "rb") as f:
            anomalies = pickle.load(f)
        print("loaded saved anomalies")
        return anomalies
    # else:
    #     anomalies = calculate_anomalies(filename, process, processes, num_comps)
    #     with open(SAVE_FILE, "wb") as f:
    #         pickle.dump(anomalies, f)
    #     print("calculated and saved anomalies")




def create_plot(anomalies, plot_title, save_tag, threshold):


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

    plt.show()
    # plt.savefig(f"PCA Detector {save_tag}.png", dpi=500)





if __name__ == '__main__':

    anomalies = []

    filters = [200, 180]
    num_comps = [6, 9]
    times = [3, 5]

    for filter in filters:
        for comps in num_comps:

            save_tag = f'BMix_LP{filter}_5x_PCA{comps}'
            anomalies.append(np.array(open_anomalies(save_tag)))

    anomalies_sum = np.sum(anomalies, axis=0)
    anomalies_sum = (anomalies_sum - np.min(anomalies_sum)) / (np.max(anomalies_sum) - np.min(anomalies_sum))

    plot_title = f'Beam Mixture - LP: 200 & 180 5x - Sum'
    save_tag = f'BMix_LPSUM_5x_PCA6'
    threshold = .6
    create_plot(anomalies_sum, plot_title, save_tag, threshold)