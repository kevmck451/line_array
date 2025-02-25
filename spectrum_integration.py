
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

    # base_path_calibration = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29/Samples/calibration'
    #
    # calibration_dict = {}
    #
    # for filepath in Path(base_path_calibration).iterdir():
    #     if filepath.suffix == '.wav':
    #         audio = Audio(filepath=filepath, num_channels=1)
    #         av_spectrum, frequency_bins = average_spectrum(audio, display=False)
    #         # print(av_spectrum)
    #         calibration_dict[filepath.stem] = (av_spectrum, frequency_bins)
    #
    #
    # base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29/Samples/samples'
    # # sample names: 30m_[tank, truck][1-5]_[raw, beam12, beammix]
    #
    # for filepath in Path(base_path).iterdir():
    #     if filepath.suffix == '.wav':
    #         audio = Audio(filepath=filepath, num_channels=1)
    #         av_spectrum, frequency_bins = average_spectrum(audio, display=False)


    base_path_calibration = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29/Samples/calibration/beammix_cal.wav'
    audio_cal = Audio(filepath=base_path_calibration, num_channels=1)
    av_spectrum_cal, frequency_bins_cal = average_spectrum(audio_cal, display=False)

    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29/Samples/samples/30m_tank5_beammix.wav'
    audio = Audio(filepath=base_path, num_channels=1)
    av_spectrum, frequency_bins = average_spectrum(audio, display=False)

    # print(f'Cal: {av_spectrum_cal.shape}\t|\t Bins: {frequency_bins_cal.shape}')
    # print(f'Test: {av_spectrum.shape}\t|\t Bins: {frequency_bins.shape}')

    # fig, ax = plt.subplots(figsize=(16, 4))
    # ax.set_title(f'Spectral Plot: {audio.name}')
    # # fig.tight_layout(pad=1)
    # ax.plot(frequency_bins, av_spectrum, color='blue', label='Tank Very High')
    # ax.plot(frequency_bins, av_spectrum_cal, color='green', label='no target')
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


    new_spec = np.maximum(av_spectrum - av_spectrum_cal, 0)

    # fig, ax = plt.subplots(figsize=(16, 4))
    # ax.set_title(f'Spectral Plot: {audio.name}')
    # # fig.tight_layout(pad=1)
    # ax.plot(frequency_bins, av_spectrum, color='blue', label='Tank Very High')
    # ax.plot(frequency_bins, av_spectrum_cal, color='green', label='no target')
    # ax.plot(frequency_bins, new_spec, color='red', label='Tank minus Cal')
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



    og_sig = np.sum(av_spectrum)
    og_cal = np.sum(av_spectrum_cal)
    target_signal = np.sum(new_spec)
    print(f'OG Sig: {og_sig}\t|\tOG Cal: {og_cal}\t|\tTar Sig: {target_signal}')


