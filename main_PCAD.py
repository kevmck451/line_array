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







def process_audio(audio):
    std_threshold = 0.5
    top_cutoff_freq = 170
    bottom_cutoff_freq = 100
    normalize_percentage = 100
    new_sample_rate = 2000

    # print('Reducing Noise')
    # audio.data = noise_reduction_filter(audio, std_threshold)
    # print('Passing High Freq')
    # audio.data = high_pass_filter(audio, bottom_cutoff_freq)
    # print('Pass Low Freq')
    # audio.data = low_pass_filter(audio, top_cutoff_freq, order=8)
    # print('Reducing Noise')
    # audio.data = noise_reduction_filter(audio)
    print('Down Sampling')
    audio.data = downsample(audio, new_sample_rate)
    audio.sample_rate = new_sample_rate
    print('Normalizing')
    audio.data = normalize(audio, normalize_percentage)


    return audio




if __name__ == '__main__':
    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29'

    # filename_calibration = 'calibration bf-12m'
    filename_calibration = 'calibration bf-12m pro1'
    filepath_calibration = f'{base_path}/Calibration/{filename_calibration}.wav'
    audio_calibration = Audio(filepath=filepath_calibration, num_channels=1)
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

    # audio_calibration = process_audio(audio_calibration)

    pca_detector = PCA_Calculator(nperseg=2**12, num_components=3)
    detector = Detector()

    chunk_size = 1 * audio_calibration.sample_rate
    overlap = chunk_size #// 4  # 50% overlap
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

    # filename = 'Test1_B12'
    filename = 'Test1_B12 Pro_1'
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

    # audio = process_audio(audio)

    anomalies = []

    overlap = chunk_size
    start = 0

    detector.anomaly_threshold = 1

    while start < len(audio.data):
        end = start + chunk_size
        components = pca_detector.process_chunk(audio.data[start:end])
        anomalies.append(detector.detect_anomalies(components))
        start += overlap

    targets = [
        75, 118, 161, 203, 246, 290, 333, 377, 421, 464,
        637, 681, 724, 767, 810, 853, 896, 939, 982, 1025,
        1156, 1200, 1244, 1287, 1330, 1374, 1417, 1460, 1503, 1546]

    colors = ['purple', 'blue', 'green']

    flight_time = (50, 2000)
    threshold = 7

    plt.figure(figsize=(24,4))
    for i, target in enumerate(targets):
        color = colors[i // 10]
        plt.axvline(x=target-flight_time[0], color=color, linestyle=':', alpha=0.5)

    plt.axhline(y=threshold, color='red', linestyle=':', alpha=0.8)
    plt.plot(anomalies[flight_time[0]:flight_time[1]])
    plt.tight_layout(pad=0.1)
    plt.show()
















