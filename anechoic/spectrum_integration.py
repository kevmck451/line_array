
# from process import average_spectrum
from acoustic.audio import Audio



from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.signal

def average_spectrum(audio_object, fft_size=65536, overlap=0.5, **kwargs):
    frequency_range = kwargs.get('frequency_range', (0, audio_object.sample_rate // 2))

    data = audio_object.data
    hop_size = int(fft_size * (1 - overlap))  # Overlapping windows

    # Compute STFT
    freq_bins, _, Zxx = scipy.signal.stft(data, fs=audio_object.sample_rate, nperseg=fft_size, noverlap=hop_size)

    # Compute magnitude and average over time
    magnitude_spectrum = np.mean(np.abs(Zxx), axis=1)

    # Select only positive frequencies in the given range
    positive_freq_mask = (freq_bins >= frequency_range[0]) & (freq_bins <= frequency_range[1])

    av_spec, freq_bins = magnitude_spectrum[positive_freq_mask], freq_bins[positive_freq_mask]

    # spectrogram_min, spectrogram_max = av_spec.min(), av_spec.max()
    # av_spec = (av_spec - spectrogram_min) / (spectrogram_max - spectrogram_min)

    return av_spec, freq_bins


if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Anechoic Chamber/test files'

    target_times = {
        'White Noise 0': (107, 10, 10),  # (center, (plus, minus))
        'White Noise 90': (36, 10, 10),
    }

    audio_raw = Audio(filepath=f'{base_path}/raw.wav', num_channels=1)
    start = (target_times.get('White Noise 0')[0]-target_times.get('White Noise 0')[2])*audio_raw.sample_rate
    finish = (target_times.get('White Noise 0')[0]+target_times.get('White Noise 0')[1])*audio_raw.sample_rate

    audio_raw.data = audio_raw.data[start:finish]
    av_spectrum_audio_raw_wn_0, frequency_bins_audio_raw_wn_0 = average_spectrum(audio_raw, display=True)

    audio_raw = Audio(filepath=f'{base_path}/raw.wav', num_channels=1)
    start = (target_times.get('White Noise 90')[0] - target_times.get('White Noise 0')[2]) * audio_raw.sample_rate
    finish = (target_times.get('White Noise 90')[0] + target_times.get('White Noise 0')[1]) * audio_raw.sample_rate
    audio_raw.data = audio_raw.data[start:finish]
    av_spectrum_audio_raw_wn_90, frequency_bins_audio_raw_wn_90 = average_spectrum(audio_raw, display=True)

    auc_wn_0 = np.sum(av_spectrum_audio_raw_wn_0)
    auc_wn_90 = np.sum(av_spectrum_audio_raw_wn_90)

    audio_beamed = Audio(filepath=f'{base_path}/beamed.wav', num_channels=7)
    start = (target_times.get('White Noise 0')[0] - target_times.get('White Noise 0')[2]) * audio_beamed.sample_rate
    finish = (target_times.get('White Noise 0')[0] + target_times.get('White Noise 0')[1]) * audio_beamed.sample_rate
    start_90 = (target_times['White Noise 90'][0] - target_times['White Noise 90'][2]) * audio_beamed.sample_rate
    finish_90 = (target_times['White Noise 90'][0] + target_times['White Noise 90'][1]) * audio_beamed.sample_rate

    sample_names = ['Raw 0', 'Raw 90', 'B12 0', 'B12 90', 'B6 0', 'B6 90', 'B4 0', 'B4 90', 'B3 0', 'B3 90', 'B3-2 0', 'B3-2 90', 'B2 0', 'B2 90', 'BM 0', 'BM 90']
    samples = [auc_wn_0, auc_wn_90]
    for i in range(7):
        audio_raw.data = audio_beamed.data[i][start:finish]
        av_spectrum_audio_beamed, _ = average_spectrum(audio_raw, display=False)
        samples.append(np.sum(av_spectrum_audio_beamed))  # Beamed 0° spectrum AUC

        audio_raw.data = audio_beamed.data[i][start_90:finish_90]
        av_spectrum_audio_beamed_90, _ = average_spectrum(audio_raw, display=False)
        samples.append(np.sum(av_spectrum_audio_beamed_90))

    # plt.figure(figsize=(5, 5))
    # plt.bar(sample_names, samples)  # Labels and heights
    # plt.ylabel('Total Spectral Energy (AUC)')
    # plt.title('Spectral Energy Comparison')
    # plt.show()

    # ----------------------------------------------------------------------

    # Compute absolute differences
    differences = [abs(samples[i] - samples[i + 1]) for i in range(0, len(samples), 2)]

    # Generate labels for differences
    difference_labels = [name.split()[0] for name in sample_names[::2]]  # Extract unique names

    # Define the new desired order
    new_order = ['Raw', 'B2', 'B3', 'B3-2', 'B4', 'B6', 'B12', 'BM']

    # Extract indices based on the new order
    reordered_samples = []
    reordered_labels = []

    for label in new_order:
        index = sample_names.index(f'{label} 0')  # Find the index of the first occurrence
        reordered_samples.append(abs(samples[index] - samples[index + 1]))  # Compute abs difference
        reordered_labels.append(label)  # Store label

    # Plot the reordered differences
    plt.figure(figsize=(10, 5))
    plt.bar(reordered_labels, reordered_samples, color='dodgerblue')
    plt.ylabel('Absolute Difference in Spectral Energy (AUC)')
    plt.title('Spectral Energy Differences Between 0° and 90°')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    plt.savefig(f"WN Absolute Difference.png", dpi=500)

    # ----------------------------------------------------------------------

    # Compute Relative Difference (Percent Change)
    relative_diff = [(abs(samples[i] - samples[i + 1]) / max(samples[i], samples[i + 1])) * 100 for i in range(0, len(samples), 2)]

    # Compute Energy Ratio (AUC_0 / AUC_90)
    energy_ratio = [samples[i] / samples[i + 1] for i in range(0, len(samples), 2)]

    # Reorder data according to the preferred order
    reordered_relative_diff = []
    reordered_energy_ratio = []
    reordered_labels = []

    for label in new_order:
        index = sample_names.index(f'{label} 0')  # Find the index of the first occurrence
        reordered_relative_diff.append(relative_diff[index // 2])
        reordered_energy_ratio.append(energy_ratio[index // 2])
        reordered_labels.append(label)

    # Plot Relative Difference (Percent Change)
    plt.figure(figsize=(10, 5))
    plt.bar(reordered_labels, reordered_relative_diff, color='dodgerblue')
    plt.ylabel('Relative Difference (%)')
    plt.title('Relative Spectral Energy Change (0° vs. 90°)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    plt.savefig(f"WN Relative Difference.png", dpi=500)

    # Plot Energy Ratio (AUC_0 / AUC_90)
    plt.figure(figsize=(10, 5))
    plt.bar(reordered_labels, reordered_energy_ratio, color='dodgerblue')
    plt.ylabel('Energy Ratio (AUC_0 / AUC_90)')
    plt.title('Spectral Energy Ratio (0° vs. 90°)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=1)  # Reference line at 1
    # plt.show()
    plt.savefig(f"WN Energy Ratio.png", dpi=500)

    # fig, ax = plt.subplots(figsize=(16, 4))
    # ax.set_title(f'Spectral Plot: White Noise 0 vs 90')
    # # fig.tight_layout(pad=1)
    # ax.plot(frequency_bins_audio_raw_wn_0, av_spectrum_audio_raw_wn_0, color='darkblue', label='0 Degrees')
    # ax.plot(frequency_bins_audio_raw_wn_90, av_spectrum_audio_raw_wn_90, color='green', label='90 Degrees')
    # ax.set_xscale('symlog')
    # ax.set_xlim([10, 10000])
    # ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    # ax.set_ylabel('Magnitude', fontweight='bold')
    #
    # ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    # ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    # ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
    # ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
    # ax.grid(True, which='both')
    # plt.tight_layout(pad=.1)
    # plt.legend(loc='best')
    # # plt.ylim(0, 1)
    # plt.show()



