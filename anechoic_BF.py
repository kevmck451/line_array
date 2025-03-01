

from acoustic.audio import Audio
from array_config import array_config_list
from mic_coordinates import generate_mic_coordinates
from Beamform.map_ch_positions import map_channels_to_positions
from Beamform.beamform import generate_beamformed_audio_iterative
from acoustic.down_sample import downsample
from acoustic.normalize import normalize
from acoustic.save_to_wav import save_to_wav
from acoustic.noise_reduction import noise_reduction_filter


from pathlib import Path
import numpy as np




if __name__ == '__main__':

    base_path = '/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Anechoic Chamber'
    directory = 'raw'
    # filename = '1_0'
    filename = '0_0'
    filepath = f'{base_path}/{directory}/{filename}.wav'
    filepath_save = f'{base_path}/beamed'
    tag_index = '-Ensemble'

    thetas = [0]
    temp_F = 75

    print('opening audio')
    audio = Audio(filepath=filepath, num_channels=12)

    print('generating mic coordinates')
    mic_coords_list = [generate_mic_coordinates(config) for config in array_config_list]

    print('mapping audio channels')
    mapped_audio_data_list = [map_channels_to_positions(audio.data, config) for config in array_config_list]

    print('beamforming')
    beamed_data = np.zeros((len(mapped_audio_data_list), mapped_audio_data_list[0].shape[2]+200))

    for idx, (audio_data, mic_coords) in enumerate(zip(mapped_audio_data_list, mic_coords_list)):
        beamed_data[idx, :] = generate_beamformed_audio_iterative(audio_data, thetas, temp_F, mic_coords)


    print('packing audio')
    original_path = Path(filepath)
    # export_tag = f'_BF{tag_index}_{theta}-{phi}'
    export_tag = f'_BF{tag_index}'
    new_filename = original_path.stem + export_tag + original_path.suffix
    new_filepath = f'{filepath_save}/{new_filename}'

    # Save the filtered audio to the new file
    beamformed_audio_object = Audio(data=beamed_data, num_channels=beamed_data.shape[0], sample_rate=48000)
    beamformed_audio_object.path = Path(new_filepath)


    # print('downsampling audio')
    # new_sample_rate = 24000
    # beamformed_audio_object.data = downsample(beamformed_audio_object, new_sample_rate)
    # beamformed_audio_object.sample_rate = new_sample_rate

    # print('Normalizing')
    # percentage = 100
    # beamformed_audio_object.data = normalize(beamformed_audio_object, percentage)

    # print('Reducing Noise')
    # std_threshold = 0.5
    # beamformed_audio_object.data = noise_reduction_filter(beamformed_audio_object, std_threshold)
    #
    # print('Normalizing')
    # percentage = 100
    # beamformed_audio_object.data = normalize(beamformed_audio_object, percentage)

    save_to_wav(beamformed_audio_object.data, beamformed_audio_object.sample_rate, beamformed_audio_object.num_channels, new_filepath)



