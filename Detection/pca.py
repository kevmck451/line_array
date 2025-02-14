
from sklearn.decomposition import PCA
from scipy.signal import stft
import numpy as np





class PCA_Calculator:
    def __init__(self,
                 nperseg=4096,
                 num_components=3):
        self.nperseg = nperseg
        self.num_components = num_components

    def process_chunk(self, chunk):

        for i in range(16, 8, -1):

            num_seg = 2**i


            try:
                frequencies, times, stft_matrix = stft(chunk.T, nperseg=num_seg)
                # frequencies, times, stft_matrix = stft(chunk.T, nperseg=self.nperseg)
                feature_matrix = np.abs(stft_matrix).T

                pca = PCA(n_components=self.num_components, svd_solver='full')
                principal_components = pca.fit_transform(feature_matrix)

                data_array = np.array(principal_components.T)
                # print(f'NperSeg = {2**i}')
                return data_array

            except:
                print()