# Functions to Process Audio

from audio import Audio


from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import matplotlib.pyplot as plt
from copy import deepcopy
from copy import copy
from scipy import signal
import numpy as np
import librosa
from math import ceil
import seaborn as sns
import matplotlib.colors as colors
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker
import threading



#-----------------------------------
# FEATURES -------------------------
#-----------------------------------
# Function to calculate the 1D average spectrum of audio (Features are 1D)
def average_spectrum(audio_object, **kwargs):
    frequency_range = kwargs.get('frequency_range', (0, audio_object.sample_rate//2))
    data = audio_object.data
    spectrum = np.fft.fft(data)  # Apply FFT to the audio data
    magnitude = np.abs(spectrum)
    frequency_bins = np.fft.fftfreq(len(data), d=1 / audio_object.sample_rate)
    positive_freq_mask = (frequency_bins >= frequency_range[0]) & (frequency_bins <= frequency_range[1])
    # channel_spectrums = [magnitude[positive_freq_mask][:len(frequency_bins)]]
    # average_spectrum = np.mean(channel_spectrums, axis=0)

    selected_magnitude = magnitude[positive_freq_mask]
    frequency_bins = frequency_bins[positive_freq_mask]
    average_spectrum = selected_magnitude

    # Apply Min-Max normalization to the
    norm = kwargs.get('norm', True)
    if norm:
        spectrogram_min, spectrogram_max = average_spectrum.min(), average_spectrum.max()
        average_spectrum = (average_spectrum - spectrogram_min) / (spectrogram_max - spectrogram_min)

    # average_spectrum = [0 if v <= 0.1 else v for v in average_spectrum]

    # frequency_list = [np.round(f, 2) for f in frequency_bins]
    # frequency_list = frequency_list[:len(average_spectrum)]
    # frequency_bins = np.array(frequency_list)
    # frequency_bins = np.squeeze(frequency_bins)

    display = kwargs.get('display', False)
    if display:
        # plt.plot(frequency_bins, average_spectrum)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f'Spectral Plot: {audio_object.name}')
        # fig.tight_layout(pad=1)
        ax.plot(frequency_bins, average_spectrum)
        ax.set_xscale('symlog')
        ax.set_xlim([10, 10000])
        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Magnitude', fontweight='bold')

        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
        ax.grid(True, which='both')
        if norm:
            plt.ylim(0, 1)

        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/{audio_object.name}')
            plt.close()
        else:
            plt.show()

    return average_spectrum, frequency_bins

# Function to calculate spectrogram of audio (Features are 2D)
def spectrogram(audio_object, **kwargs):
    '''
    :param audio_object: audio object from Audio Abstract
    :param kwargs:
        - feature_params (dictionary with bandwidth, window_size, or hop_length)
        - normalize (True or False) default is True
        - display (True or False) default is False
        - details (True or False) default is False
    :return: numpy array either 1 channel or multi channel of spectrogram values in dB from 0 - 24,000 Hz
            if details True then returns spec, freqs, and time arrays
    '''


    feature_params = kwargs.get('feature_params', 'None')
    window_sizes = [65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 254]

    if feature_params == 'None':
        bandwidth = (0, audio_object.sample_rate//2)
        window_size = window_sizes[4]
        hop_length = window_size // 4
    else:
        if 'bandwidth' in feature_params:
            bandwidth = feature_params.get('bandwidth')
        else: bandwidth = (0, audio_object.sample_rate//2)
        if 'window_size' in feature_params:
            window_size = feature_params.get('window_size')
        else: window_size = window_sizes[0]
        if 'hop_length' in feature_params:
            hop_length = feature_params.get('hop_length')
        else: hop_length = window_size // 4

    data = audio_object.data

    # Initialize an empty list to store the spectrograms for each channel
    spectrograms = []
    frequencies = []
    times = []

    # Check if audio_object is multi-channel
    if len(data.shape) == 1:
        # Mono audio data, convert to a list with a single item for consistency
        data = [data]
    for channel_data in data:
        # Calculate the spectrogram using Short-Time Fourier Transform (STFT)
        frequency_list, times_list, Zxx = signal.stft(channel_data, nfft=window_size, fs=audio_object.sample_rate)
        frequency_list = [int(np.round(f)) for f in frequency_list]
        # print(len(frequency_list), Zxx.shape)
        frequencies.append(frequency_list)
        times.append(times_list)
        # print(f'frequencies: {frequency_list}')
        # print(f'times: {times_list}')
        # print(f'Zxx: {Zxx}')
        # print(f'Zxx Shape: {Zxx.shape}')

        # Calculate the magnitude of the STFT (spectrogram)
        spectrogram = np.abs(Zxx)
        # mins = np.min(spectrogram)
        # maxs = np.max(spectrogram)
        # means = np.mean(spectrogram)
        # print(f'Spec Values:\nMin: {mins}\nMax: {maxs}\nMean: {means}')

        # Convert to decibels
        # spectrogram = 20 * np.log10(spectrogram + 1e-10)
        # minsd = np.min(spectrogram_db)
        # maxsd = np.max(spectrogram_db)
        # meansd = np.mean(spectrogram_db)
        # print(f'Spec_dB Values:\nMin d: {minsd}\nMaxd: {maxsd}\nMeand: {meansd}')
        # print(f'Spec Shape: {spectrogram_db.shape}')

        # Apply Min-Max normalization to the
        norm = kwargs.get('norm', True)
        if norm:
            spectrogram_min, spectrogram_max = spectrogram.min(), spectrogram.max()
            spectrogram = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min)
            # print(spectrogram_db_min)
            # print(spectrogram_db_max)
            # print(spectrogram_db)

        # Connect freq index to freq band
        nyquist_frequency = audio_object.sample_rate / 2
        frequency_resolution = nyquist_frequency / (window_size / 2)

        def find_closest_index(lst, target): return min(range(len(lst)), key=lambda i: abs(lst[i] - target))
        bottom_index = find_closest_index(frequency_list, bandwidth[0])
        top_index = find_closest_index(frequency_list, bandwidth[1])

        # Cut the spectrogram to the desired frequency bandwidth and append to the
        spectrograms.append(spectrogram[bottom_index:top_index+1])

        stats = kwargs.get('stats', False)
        if stats:
            # print(f'Spectro_dB: {spectrogram_db}')
            print(f'Spectro_dB Shape: {spectrogram.shape}')
            print(f'Freq Range: ({bandwidth[0]},{bandwidth[1]}) Hz')
            print(f'Freq Resolution: {frequency_resolution} Hz')
            # print(f'Freq List: {freq_list}')

        display = kwargs.get('display', False)
        if display:
            plt.imshow(spectrogram, aspect='auto', origin='lower',
                       extent=[times_list[0], times_list[-1], frequency_list[0], frequency_list[-1]],
                       vmin=0, vmax=1)
            plt.colorbar()
            plt.xlabel('Time [sec]')
            plt.ylabel('Frequency [Hz]')
            plt.title('Spectrogram')
            plt.show()

    spectrograms = np.array(spectrograms)
    frequencies = np.array(frequencies)
    times = np.array(times)

    spectrograms = np.squeeze(spectrograms) # removes all singular axis
    frequencies = np.squeeze(frequencies)  # removes all singular axis
    times = np.squeeze(times)  # removes all singular axis

    details = kwargs.get('details', False)
    if details: return spectrograms, frequencies, times
    else: return spectrograms

# Function to calculate spectrogram of audio (Features are 2D)
def spectrogram_2(audio_object, **kwargs):
    feature_params = kwargs.get('feature_params', None)
    bandwidth = feature_params.get('bandwidth', (0, 24000))
    nperseg = feature_params.get('nperseg', 32768) #32768 16384

    # print(audio_object.data.shape)

    if audio_object.num_channels > 1: data = audio_object.data[0]
    else: data = audio_object.data

    # Define a small constant to prevent log of zero
    epsilon = 1e-10

    # Compute the spectrogram
    f, t, Sxx = signal.spectrogram(data, fs=audio_object.sample_rate, nperseg=nperseg)
    spec = 10 * np.log10(Sxx + epsilon)

    # Normalize spec between 0 and 1
    spec_min = spec.min()
    spec_max = spec.max()
    if spec_max - spec_min != 0:
        spec = (spec - spec_min) / (spec_max - spec_min)
    else:
        spec = np.zeros(spec.shape)  # Handle the case where spec_max == spec_min

    # Find indices of the frequency range
    freq_indices = np.where((f >= bandwidth[0]) & (f <= bandwidth[1]))[0]
    f_subset = f[freq_indices]
    spec = spec[freq_indices, :]

    # Plot the normalized spectrogram for the specified frequency range
    display = kwargs.get('display', False)
    if display:
        plt.figure(figsize=(16,4))
        plt.pcolormesh(t, f_subset, spec, shading='gouraud', vmin=0, vmax=1)
        plt.ylabel(f'Frequency {bandwidth[0]}-{bandwidth[1]}Hz')
        plt.xlabel('Time [sec]')
        plt.title(f'{audio_object.name}')
        plt.colorbar(label='Intensity', extend='both')
        plt.yscale('log')
        plt.tight_layout(pad=1)

        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/{audio_object.name}')
            plt.close()
        else:
            plt.show()

    else:
        spectrograms = np.array(spec)
        frequencies = np.array(f_subset)
        times = np.array(t)

        spectrograms = np.squeeze(spectrograms)  # removes all singular axis
        frequencies = np.squeeze(frequencies)  # removes all singular axis
        times = np.squeeze(times)  # removes all singular axis

        details = kwargs.get('details', False)
        if details:
            return spectrograms, frequencies, times
        else:
            return spectrograms

# Function to calculate MFCC of audio (Features are 2D)
def mfcc(audio_object, **kwargs):
    stats = kwargs.get('stats', False)
    feature_params = kwargs.get('feature_params', 'None')

    if feature_params == 'None':
        n_mfcc = 15
    else:
        n_mfcc = feature_params.get('n_coeffs')

    # data = audio_object.data
    # Normalize audio data
    Audio_Object = normalize(audio_object)
    data = Audio_Object.data

    # Initialize an empty list to store the MFCCs for each channel
    mfccs_all_channels = []

    # Check if audio_object is multi-channel
    if len(data.shape) == 1:
        # Mono audio data, convert to a list with a single item for consistency
        data = [data]

    for channel_data in data:
        # Calculate MFCCs for this channel
        if n_mfcc == 'None':
            n_mfcc = 15
        mfccs = librosa.feature.mfcc(y=channel_data, sr=audio_object.sample_rate, n_mfcc=n_mfcc, n_fft=2048, n_mels=128)

        # Normalize the MFCCs
        mfccs = StandardScaler().fit_transform(mfccs)

        if stats:
            print(f'MFCC: {mfccs}')

        # Append to the list
        mfccs_all_channels.append(mfccs)

    # Convert the list of MFCCs to a numpy array and return
    mfccs_all_channels = np.array(mfccs_all_channels)
    mfccs_all_channels = np.squeeze(mfccs_all_channels)  # removes all singular axis

    display = kwargs.get('display', False)
    if display:

        # Number of MFCCs to plot individually
        num_individual_mfccs = n_mfcc

        # Setup the matplotlib figure
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14,8))
        fig.subplots_adjust(hspace=0.5)

        # Heatmap of All Coefficients
        sns.heatmap(mfccs_all_channels, cmap='coolwarm', ax=axes[0], norm=colors.Normalize(vmin=-3.5, vmax=3.5))
        axes[0].set_title('MFCC Heatmap')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('MFCC Coefficients')

        # Line Plots for Select Coefficients
        for i in range(num_individual_mfccs):
            axes[1].plot(mfccs_all_channels[i], label=f'MFCC {i}')
        axes[1].set_title('Line Plots of Individual MFCCs')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Coefficient Value')
        axes[1].set_ylim(-3.5, 3.5)  # Set y-axis limits
        # axes[1].legend(loc='right')

        # Histograms for Distribution Analysis
        for i in range(num_individual_mfccs):
            sns.histplot(mfccs_all_channels[i], kde=True, ax=axes[2], label=f'MFCC {i}')
        axes[2].set_title('Histograms of Individual MFCCs')
        axes[2].set_xlabel('Coefficient Value')
        axes[2].set_ylabel('Frequency')
        axes[2].set_xlim(-3.5, 3.5)  # Set x-axis limits
        axes[2].set_ylim(0, 150)  # Set x-axis limits
        # axes[2].legend(loc='right')

        # Box Plots for Statistical Overview
        # axes[3].boxplot(mfccs_all_channels[:num_individual_mfccs].T, notch=True, patch_artist=True)
        # axes[3].set_title('Box Plots of Individual MFCCs')
        # axes[3].set_xlabel('MFCC Coefficient')
        # axes[3].set_ylabel('Coefficient Value')
        # axes[3].set_ylim(-3.5, 3.5)  # Set y-axis limits
        # axes[3].set_xticklabels([(i) for i in range(num_individual_mfccs)])

        # Show the plots
        plt.suptitle(audio_object.name)
        plt.tight_layout(pad=1)

        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/{audio_object.name}')
            plt.close()
        else:
            plt.show()

    return mfccs_all_channels

# Function to calculate Zero Crossing Rate of audio (Features are 1D)
def zcr(audio_object, **kwargs):
    # Extract audio data
    data = audio_object.data

    # Check for mono or stereo and handle accordingly
    if len(data.shape) == 1:
        # Mono audio
        data = [data]
    else:
        # Stereo or multi-channel audio, transpose to iterate over channels
        data = data.T

    zcr_values = []
    for channel_data in data:
        # Calculate ZCR for each channel
        zcr_channel = librosa.feature.zero_crossing_rate(y=channel_data)
        zcr_values.append(zcr_channel)

    # Visualization
    display = kwargs.get('display', False)
    if display:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 6))

        # Line Plot
        for idx, zcr_channel in enumerate(zcr_values):
            axes[0].plot(zcr_channel.flatten(), label=f'Channel {idx}')
        axes[0].set_title(f'ZCR: {audio_object.name}')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('ZCR')
        axes[0].set_ylim(0, 0.1)  # Set y-axis limits
        # axes[0].set_xlim(-3.5, 3.5)  # Set y-axis limits
        axes[0].legend()

        # Histogram
        for zcr_channel in zcr_values:
            sns.histplot(zcr_channel.flatten(), kde=True, ax=axes[1])
        axes[1].set_title('Histogram of ZCR Values')
        axes[1].set_xlabel('ZCR')
        axes[1].set_ylabel('Frequency')
        axes[1].set_ylim(0, 250)  # Set y-axis limits
        axes[1].set_xlim(0, 0.1)  # Set y-axis limits

        plt.tight_layout()

        # Saving the plot if required
        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')

        if save:
            plt.savefig(f'{save_path}/{audio_object.name}.png')
            plt.close()
        else:
            plt.show()

    # Statistical information
    stats = kwargs.get('stats', False)
    if stats:
        for idx, zcr_channel in enumerate(zcr_values):
            print(f'Channel {idx} ZCR Stats:')
            print(f'Mean: {np.mean(zcr_channel)}')
            print(f'Standard Deviation: {np.std(zcr_channel)}')
            print('---')

    zcr_values = np.array(zcr_values)
    zcr_values = np.squeeze(zcr_values)

    return zcr_values

# Function to calculate Energy of signal
def energy(audio_object, frame_length=1024, hop_length=512, **kwargs):
    # Extract audio data
    data = audio_object.data

    # Check for mono or stereo and handle accordingly
    if len(data.shape) == 1:
        # Mono audio
        data = [data]
    else:
        # Stereo or multi-channel audio, transpose to iterate over channels
        data = data.T

    energy_values = []
    for channel_data in data:
        # Calculate the energy for each frame for this channel
        energy = np.array([
            sum(abs(channel_data[i:i+frame_length]**2))
            for i in range(0, len(channel_data), hop_length)
        ])
        energy_values.append(energy)

    # Visualization
    display = kwargs.get('display', False)
    if display:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 6))

        # Line Plot for Energy
        for idx, energy in enumerate(energy_values):
            axes[0].plot(energy, label=f'Channel {idx}')
        axes[0].set_title(f'Energy: {audio_object.name}')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Energy')
        axes[0].legend()
        axes[0].set_ylim(0, 150)  # Set y-axis limits

        # Histogram for Energy Distribution
        for energy in energy_values:
            sns.histplot(energy, kde=True, ax=axes[1])
        axes[1].set_title('Histogram of Energy Values')
        axes[1].set_xlabel('Energy')
        axes[1].set_ylabel('Frequency')
        axes[1].set_ylim(0, 250)  # Set y-axis limits
        axes[1].set_xlim(0, 150)  # Set y-axis limits

        plt.tight_layout()

        # Saving the plot if required
        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/{audio_object.name}_Energy.png')
            plt.close()
        else:
            plt.show()

    # Statistical information
    stats = kwargs.get('stats', False)
    if stats:
        for idx, energy in enumerate(energy_values):
            print(f'Channel {idx} Energy Stats:')
            print(f'Mean: {np.mean(energy)}')
            print(f'Standard Deviation: {np.std(energy)}')
            print('---')

    energy_values = np.array(energy_values)
    energy_values = np.squeeze(energy_values)

    return energy_values

# Function to calculate Spectral Centroid of signal
def spectral_centroid(audio_object, sr=22050, frame_length=1024, hop_length=512, **kwargs):
    # Extract audio data
    data = audio_object.data

    # Check for mono or stereo and handle accordingly
    if len(data.shape) == 1:
        # Mono audio
        data = [data]
    else:
        # Stereo or multi-channel audio, transpose to iterate over channels
        data = data.T

    spectral_centroid_values = []
    for channel_data in data:
        # Calculate the spectral centroid for each channel
        centroid = \
        librosa.feature.spectral_centroid(y=channel_data, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
        spectral_centroid_values.append(centroid)

    # Visualization
    display = kwargs.get('display', False)
    if display:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 6))

        # Line Plot for Spectral Centroid
        for idx, centroid in enumerate(spectral_centroid_values):
            axes[0].plot(centroid, label=f'Channel {idx}')
        axes[0].set_title(f'Spectral Centroid: {audio_object.name}')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].legend()
        axes[0].set_ylim(0, 2000)  # Set y-axis limits

        # Histogram for Spectral Centroid Distribution
        for centroid in spectral_centroid_values:
            sns.histplot(centroid, kde=True, ax=axes[1])
        axes[1].set_title('Histogram of Spectral Centroid Values')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_ylim(0, 160)  # Set y-axis limits
        axes[1].set_xlim(0, 2000)  # Set y-axis limits

        plt.tight_layout()

        # Saving the plot if required
        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/{audio_object.name}_SpectralCentroid.png')
            plt.close()
        else:
            plt.show()

    # Statistical information
    stats = kwargs.get('stats', False)
    if stats:
        for idx, centroid in enumerate(spectral_centroid_values):
            print(f'Channel {idx} Spectral Centroid Stats:')
            print(f'Mean: {np.mean(centroid)}')
            print(f'Standard Deviation: {np.std(centroid)}')
            print('---')

    spectral_centroid_values = np.array(spectral_centroid_values)
    spectral_centroid_values = np.squeeze(spectral_centroid_values)

    return spectral_centroid_values

# Function to create custom array of features
def feature_combo_1(audio_object, **kwargs):
    num_mfccs = 15

    # # Shared dictionary to store results
    # results = {}
    #
    # def thread_function(func, name, *args, **kwargs):
    #     # Call the function and store the result in the shared dictionary
    #     results[name] = func(*args, **kwargs)
    #
    # # Create and start the threads
    # threads = []
    # for func, name, args, kwargs in [(mfcc, 'mfcc', (audio_object,), {'feature_params': {'n_coeffs': num_mfccs}}),
    #                                  (average_spectrum, 'average_spectrum', (audio_object,),
    #                                   {'frequency_range': (350, 2800)}),
    #                                  (zcr, 'zcr', (audio_object,), {}),
    #                                  (spectral_centroid, 'spectral_centroid', (audio_object,), {}),
    #                                  (energy, 'energy', (audio_object,), {})]:
    #     thread = threading.Thread(target=thread_function, args=(func, name) + args, kwargs=kwargs)
    #     threads.append(thread)
    #     thread.start()
    #
    # # Wait for all threads to complete
    # for thread in threads:
    #     thread.join()
    #
    # # Now, results dictionary will have the output of each function
    # mfccs = results['mfcc']
    # av_spec, _ = results['average_spectrum']
    # zcr_values = results['zcr']
    # spec_centroid = results['spectral_centroid']
    # energy_values = results['energy']

    mfccs = mfcc(audio_object, feature_params={'n_coeffs': num_mfccs})
    av_spec, _ = average_spectrum(audio_object, frequency_range=(350, 2800))
    zcr_values = zcr(audio_object)
    spec_centroid = spectral_centroid(audio_object)
    energy_values = energy(audio_object)

    # Create Feature Combo Size
    size = mfccs.shape[1]
    num_features = mfccs.shape[0] + 4
    feature_array = np.zeros((num_features, size))

    # MFCCs
    for i, f in enumerate(mfccs):
        feature_array[i, :] = f
    # del mfccs

    # Average Spectrum
    if av_spec.shape[0] != size:
        feat = interpolate_values(av_spec, size)
        feature_array[(num_features-4), :] = feat
    else:
        feature_array[(num_features-4), :] = av_spec
    # del av_spec

    # Zero Crossing Rate
    if zcr_values.shape[0] != size:
        feat = interpolate_values(zcr_values, size)
        feature_array[(num_features-3), :] = feat
    else:
        feature_array[(num_features-3), :] = zcr_values
    # del zcr_values

    # Spectral Centroid
    if spec_centroid.shape[0] != size:
        feat = interpolate_values(spec_centroid, size)
        feature_array[(num_features-2), :] = feat
    else:
        feature_array[(num_features-2), :] = spec_centroid
    # del spec_centroid

    # Energy
    if energy_values.shape[0] != size:
        feat = interpolate_values(energy_values, size)
        feature_array[(num_features-1), :] = feat
    else:
        feature_array[(num_features-1), :] = energy_values
    # del energy_values

    stats = kwargs.get('stats', False)
    if stats:
        for feature in feature_array:
            print(f'Mean: {np.mean(feature)}')

    for i, feature in enumerate(feature_array):
        feat = np.interp(feature, (feature.min(), feature.max()), (0, 1))
        feature_array[i, :] = feat

    # Visualization
    display = kwargs.get('display', False)
    if display:
        plt.figure(figsize = (14, 6))
        plt.title(f'Feature Combo: {audio_object.name}\nMFCC(0-{num_features-5}), Av Spec({num_features-4}), '
                  f'ZCR({num_features-3}), Spec Cen({num_features-2}), Energy{num_features-1})')
        plt.imshow(feature_array, aspect='auto', origin='lower')
        plt.xlabel('Time Sample')
        plt.ylabel('Features')
        plt.yticks(range(num_features))
        plt.tight_layout(pad=1)

        # Saving the plot if required
        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/{audio_object.name}.png')
            plt.close()
        else:
            plt.show()

    # feature_array = np.array(feature_array)
    # feature_array = np.squeeze(feature_array)

    return feature_array

#-----------------------------------
# OTHER ----------------------------
#-----------------------------------
# Function to interpolate values of an array
def interpolate_values(array, num_values):
    interpolator = interp1d(np.arange(len(array)), array)
    new_indices = np.linspace(0, len(array) - 1, num_values)
    return interpolator(new_indices)

# Function to Normalize Data
def takeoff_trim(audio_object, takeoff_time):
    audio_copy = deepcopy(audio_object)
    samples_to_remove = int(np.round((takeoff_time * audio_object.sample_rate) + 1))
    audio_copy.data = audio_object.data[:, samples_to_remove:]

    return audio_copy

# Function to window over a sample of a specific length
def generate_windowed_chunks(audio_object, window_size):
    # Ensure window_samples is an integer, round up to ensure the window is not smaller than intended
    window_samples = int(ceil(audio_object.sample_rate * window_size))
    increment_samples = audio_object.sample_rate  # Increment by 1 second worth of samples
    total_samples = len(audio_object.data)

    audio_ob_list = []
    labels = []

    for i, window_start in enumerate(range(0, total_samples - window_samples + 1, increment_samples)):
        audio_copy = deepcopy(audio_object)
        audio_copy.sample_length = 4.0
        audio_copy.num_samples = window_samples
        start = window_start
        end = start + window_samples

        # Ensure the window doesn't exceed the audio length
        if end <= total_samples:
            try:
                label = int(audio_object.path.parent.stem)
            except ValueError:
                label = audio_object.path.parent.stem
            labels.append(label)  # Add Label (folder name)

            audio_copy.data = audio_object.data[start:end]
            audio_copy.chunk_time = (start, end)
            audio_copy.chunk_index = i
            audio_ob_list.append(audio_copy)


    if len(audio_ob_list) != len(labels):
        print(f'Error: {audio_object.path.stem}')
        raise Exception('Audio Object List and Label List Length don\'t Match')

    return audio_ob_list, labels

# Function to convert audio sample to a specific length
def generate_chunks(audio_object, length):

    num_samples = audio_object.sample_rate * length
    start = 0
    end = num_samples
    total_samples = len(audio_object.data)

    audio_ob_list = []
    labels = []

    # If the audio file is too short, pad it with zeroes
    if total_samples < num_samples:
        audio_object.data = np.pad(audio_object.data, (0, num_samples - len(audio_object.data)))
        audio_object.sample_length = length
        audio_object.num_samples = length * audio_object.sample_rate
        audio_ob_list.append(audio_object)
        label = int(audio_object.path.parent.stem)
        labels.append(label)  # Add Label (folder name)
    # If the audio file is too long, shorten it

    else:
        while end <= total_samples:
            audio_copy = copy(audio_object)
            audio_copy.data = audio_object.data[start:end]
            audio_copy.sample_length = length
            audio_copy.num_samples = length * audio_copy.sample_rate
            audio_ob_list.append(audio_copy)
            start, end = (start + num_samples), (end + num_samples)
            try:
                label = int(audio_object.path.parent.stem)
            except:
                try:
                    label = audio_object.path.parent.stem
                except:
                    label = ''
            labels.append(label)  # Add Label (folder name)


    if len(audio_ob_list) != len(labels):
        print(f'Error: {audio_object.path.stem}')
    return audio_ob_list, labels

# Function to convert audio sample to a specific length
def generate_chunks_4ch(audio_object, length, training=False):
    num_samples = audio_object.sample_rate * length
    start = 0
    end = num_samples
    total_samples = audio_object.data.shape[1]

    audio_ob_list = []
    labels = []

    # If the audio file is too short, pad it with zeroes
    if total_samples < num_samples:
        audio_object.data = np.pad(audio_object.data, (0, num_samples - len(audio_object.data)))
        audio_ob_list.append(audio_object)
        if training:
            label = int(audio_object.path.parent.stem)
            labels.append(label)  # Add Label (folder name)
    # If the audio file is too long, shorten it

    elif total_samples > num_samples:
        while end <= total_samples:
            audio_copy = deepcopy(audio_object)
            audio_copy.data = audio_object.data[:, start:end]
            audio_ob_list.append(audio_copy)
            start, end = (start + num_samples), (end + num_samples)
            if training:
                label = int(audio_object.path.parent.stem)
                labels.append(label)  # Add Label (folder name)

    if training:
        if len(audio_ob_list) != len(labels):
            print(f'Error: {audio_object.path.stem}')
        return audio_ob_list, labels
    else: return audio_ob_list

# Function to convert 4 channel wav to list of 4 objects
def channel_to_objects(audio_object):

    if audio_object.num_channels == 4:
        audio_a = deepcopy(audio_object)
        audio_a.data = audio_object.data[0]
        audio_a.which_channel = 1
        audio_a.num_channels = 1
        audio_b = deepcopy(audio_object)
        audio_b.data = audio_object.data[1]
        audio_b.which_channel = 2
        audio_b.num_channels = 1
        audio_c = deepcopy(audio_object)
        audio_c.data = audio_object.data[2]
        audio_c.which_channel = 3
        audio_c.num_channels = 1
        audio_d = deepcopy(audio_object)
        audio_d.data = audio_object.data[3]
        audio_d.which_channel = 4
        audio_d.num_channels = 1

        return [audio_a, audio_b, audio_c, audio_d]

    elif audio_object.num_channels == 3:
        audio_a = deepcopy(audio_object)
        audio_a.data = audio_object.data[0]
        audio_a.which_channel = 1
        audio_a.num_channels = 1
        audio_b = deepcopy(audio_object)
        audio_b.data = audio_object.data[1]
        audio_b.which_channel = 2
        audio_b.num_channels = 1
        audio_c = deepcopy(audio_object)
        audio_c.data = audio_object.data[2]
        audio_c.which_channel = 3
        audio_c.num_channels = 1

        return [audio_a, audio_b, audio_c]

    else:
        audio_a = deepcopy(audio_object)
        audio_a.data = audio_object.data[0]
        audio_a.which_channel = 1
        audio_a.num_channels = 1
        audio_b = deepcopy(audio_object)
        audio_b.data = audio_object.data[1]
        audio_b.which_channel = 2
        audio_b.num_channels = 1

        return [audio_a, audio_b]

# Function to calculate the Signal to Noise Ratio with PSD
def signal_noise_ratio_psd(signal, noise, ):
    frequencies, psd_signal = power_spectral_density(signal)
    frequencies , psd_noise = power_spectral_density(noise)

    snr = 10 * np.log10(psd_signal / psd_noise)
    return frequencies, snr

# Function to calculate the Signal to Noise Ratio with RMS
def signal_noise_ratio_rms(signal, noise):
    rms_sig = root_mean_square(signal)
    rms_noise = root_mean_square(noise)
    snr = 10 * np.log10(rms_sig / rms_noise)
    return snr

# Function to calculate the Power Spectral Density
def power_spectral_density(audio_object):
    # freq is list of frequencies and psd is array with number of channels
    # that contains the power at those frequencies power (watts) per Hz

    frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate/4, nfft=32768) # 2048, 4096, 8192, 16384
    # frequencies, psd = signal.welch(audio_object.data, fs=audio_object.sample_rate, average='mean') # 2048, 4096, 8192, 16384


    from matplotlib import pyplot as plt
    import pandas as pd

    avg_freq, avg_data = signal.welch(x=audio_object.data[0], fs=audio_object.sample_rate/4, average='mean')
    avg_data_pd = pd.Series(avg_data).rolling(13, center=True).mean().to_numpy()
    plt.figure(1, figsize=(14,8)).clf()
    plt.semilogy(avg_freq, avg_data, label='Raw PSD', lw=1, alpha=0.75)
    plt.semilogy(avg_freq, avg_data_pd, label='Rolled', lw=1, alpha=0.75)
    plt.title('\nPower Spectral Density Estimate')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Power Spectral Density $\left(\frac{V^{2}}{Hz}\right)$')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


    return frequencies, psd

# Function to calculate the RMS Power
def root_mean_square(audio_object):
    rms = np.sqrt(np.mean(np.square(audio_object.data)))

    return rms

# Function to Increase or Decrease Sample Gain
def amplify(audio_object, gain_db):
    Audio_Object_amp = deepcopy(audio_object)

    # convert gain from decibels to linear scale
    gain_linear = 10 ** (gain_db / 20)

    # multiply the audio data by the gain factor
    Audio_Object_amp.data *= gain_linear

    return Audio_Object_amp

# Function to mix down multiple channels to mono
def mix_to_mono(audio_objects):
    # Check if audio objects have the same sample rate
    sample_rate = audio_objects[0].sample_rate
    for audio in audio_objects:
        if audio.sample_rate != sample_rate:
            # audio.data = librosa.resample(y=audio.data, orig_sr=audio.sample_rate, target_sr=sample_rate)
            audio.data = resample(audio.data, sample_rate)

    # Initialize the combined data as zeros, using the maximum number of samples from the audio objects
    max_samples = max([audio.num_samples for audio in audio_objects])
    combined_data = np.zeros(max_samples)

    # Iterate through each audio object and sum up the data
    for audio in audio_objects:
        # If the audio has multiple channels, sum them to mono first
        if audio.num_channels > 1:
            if len(audio.data.shape) == 1:  # if audio.data is 1D
                mono_data = audio.data
            else:  # if audio.data is multi-dimensional
                mono_data = np.mean(audio.data, axis=0)
            # Ensure that the mono_data length matches the combined_data
            if len(mono_data) < len(combined_data):
                padding = np.zeros(len(combined_data) - len(mono_data))
                mono_data = np.concatenate((mono_data, padding))
            combined_data += mono_data
        else:
            # Ensure that the audio.data length matches the combined_data
            if len(audio.data) < len(combined_data):
                padding = np.zeros(len(combined_data) - len(audio.data))
                combined_data += np.concatenate((audio.data, padding))
            else:
                combined_data += audio.data

    # Average the combined data
    combined_data /= len(audio_objects)

    # Create a new Audio_Abstract object with the combined data
    mixed_audio = Audio_Abstract(data=combined_data, sample_rate=sample_rate, num_channels=1,
                                 num_samples=len(combined_data))

    return mixed_audio

#-----------------------------------
# PREPROCESSING --------------------
#-----------------------------------
# Function to Normalize Data
def normalize(audio_object, percentage=95):
    max_value = np.max(audio_object.data).round(5)
    if max_value == 0:
        print(audio_object.path)
        raise Exception('Max Value is Zero')

    # Assuming your audio data is in a variable called 'audio_data'
    audio_object.data = np.nan_to_num(audio_object.data)

    # make a deep copy of the audio object to preserve the original
    audio_normalized = deepcopy(audio_object)



    max_value = np.max(np.abs(audio_normalized.data))
    normalized_data = audio_normalized.data / max_value * (percentage / 100.0)

    audio_normalized.data = normalized_data

    return audio_normalized

# Function to compress audio
def compression(audio_object, threshold=-20, ratio=3.0, gain=1, attack=5, release=40):
    # Extracting the audio data
    audio_data = audio_object.data

    # Ensure audio_data is mono for simplicity
    if audio_object.num_channels > 1:
        audio_data = np.mean(audio_data, axis=0)

    # Convert dB threshold to amplitude
    threshold_amplitude = librosa.db_to_amplitude(threshold)

    # Apply compression
    compressed_data = np.zeros_like(audio_data)
    for i, sample in enumerate(audio_data):
        if abs(sample) > threshold_amplitude:
            gain_dB = librosa.amplitude_to_db(np.array([abs(sample)]))[0]
            gain = (gain_dB - threshold) / ratio
        else:
            gain = 1.0

        # Attack/release dynamics (basic form)
        target_gain = min(gain, 1.0)
        step = (target_gain - gain) / (attack if target_gain < gain else release)
        gain += step

        compressed_data[i] = sample * gain

    # Return a new Audio_Abstract object with the compressed data
    compressed_audio = Audio_Abstract(data=compressed_data, sample_rate=audio_object.sample_rate, num_channels=1,
                                      num_samples=len(compressed_data))

    return compressed_audio

# Function to subtract Hex from sample
def spectra_subtraction_hex(audio_object, **kwargs):
    pass

# Function for a High Pass Filter
def high_pass_shelf_filter(audio_object, threshold, **kwargs):

    spectrum, f_bins = average_spectrum(audio_object, norm=False, display=False)

    print(spectrum.shape)
    print(f_bins.shape)

    # check for duplicates
    if len(f_bins) == len(set(f_bins)):
        print("All elements are unique.")
    else:
        print("There are duplicates in the list.")

    print(f_bins)

    high_pass_array = np.ones(f_bins.shape)
    print(high_pass_array.shape)

    high_pass_array[f_bins <= threshold] = 0

    spectrum = spectrum * high_pass_array

    display = kwargs.get('display', False)
    if display:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(f'Spectral Plot: {audio_object.name}')
        # fig.tight_layout(pad=1)
        ax.plot(f_bins, spectrum)
        ax.set_xscale('symlog')
        ax.set_xlim([10, 10000])
        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Magnitude', fontweight='bold')

        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
        ax.grid(True, which='both')
        save = kwargs.get('save', False)
        save_path = kwargs.get('save_path', '')
        if save:
            plt.savefig(f'{save_path}/{audio_object.name}')
            plt.close()
        else:
            plt.show()

    audio_filtered = deepcopy(audio_object)
    audio_filtered.data = np.real(np.fft.ifft(spectrum))
    cut_off = 0.01 * len(audio_filtered.data)
    audio_filtered.data = audio_filtered.data[cut_off:(len(audio_filtered.data)-cut_off)]
    audio_filtered.waveform(display=True)


class Process:
    def __init__(self, source_directory, dest_directory):

        utils.copy_directory_structure(source_directory, dest_directory)




if __name__ == '__main__':

    # filepath = '/Users/KevMcK/Dropbox/2 Work/1 Optics Lab/1 Acoustic/Data/Investigations/Static Tests/Static Test 1/Samples/Engine_1/3_10m-D-DEIdle.wav'
    # audio = Audio_Abstract(filepath=filepath)
    # audio.data = audio.data[2]
    # audio = normalize(audio)
    # print(audio)

    # feature = zcr(audio, stats=False)
    # print(feature.shape)
    #
    # feature = spectrogram(audio)
    # print(feature.shape)
    #
    # feature = mfcc(audio)
    # print(feature.shape)

    # spectra_subtraction_hex(audio)

    filepath = af.hex_hover_combo_thin
    # filepath = af.hex_hover_combo_thick
    # filepath = af.hex_hover_10m_1k_static1
    # filepath = af.hex_hover_10m_static2
    # filepath = af.angel_ff_1
    # filepath = af.amb_orlando_1
    # filepath = af.diesel_bulldozer_1_1

    # filepath = af.hex_diesel_99
    # filepath = af.hex_diesel_59
    # filepath = af.hex_diesel_1

    audio = Audio_Abstract(filepath=filepath, num_channels=1)
    high_pass_shelf_filter(audio, threshold=90, display=True)

    # audio.mfccs = mfcc(audio, feature_params={'n_coeffs':12}, display=True)
    # audio.mfccs = mfcc(audio, feature_params={'n_coeffs': 12}, display=False)
    # print(audio.mfccs.shape)
    # plt.imshow(audio.mfccs)
    # plt.tight_layout(pad=1)
    # plt.show()

    # audio.av_spec, audio.av_spec_fb = average_spectrum(audio, display=True)
    # print(audio.av_spec.shape)

    # audio.spectrogram = spectrogram(audio, stats=False, feature_params={'bandwidth': (0, 20000)}, display=True)
    # audio.spectrogram, audio.spec_freqs, audio.spec_times = spectrogram(audio, stats=False, feature_params={'bandwidth':(0, 24000)}, display=False, details=True, norm=True)
    # print(f'Max: {np.max(audio.spectrogram)}\nMin: {np.min(audio.spectrogram)}\nMean: {np.mean(audio.spectrogram)}')
    #
    # fig, ax = plt.subplots()
    # for i in range(0, len(audio.spec_times)):
    #     if i%20 == 0:
    #         ax.plot(audio.spec_freqs, audio.spectrogram[:, i])
    #         ax.set_xlabel('Frequency (Hz)', fontweight='bold')
    #         ax.set_ylabel('Magnitude', fontweight='bold')
    #         ax.set_title(f'Spectral Plot: {audio.name}')
    #         ax.grid(True)
    #         fig.tight_layout(pad=1)
    # plt.show()

    # audio_list, _ = generate_windowed_chunks(audio, window_size=0.1)
    #
    # for audio in audio_list:
    #     audio.av_spec = average_spectrum(audio, display=False)
    # plt.show()