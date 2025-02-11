



import numpy as np



class Detector:
    def __init__(self):

        self.baseline_calculated = False # set outside this class
        self.anomaly_threshold = 8

        self.num_pca_components = None
        self.num_samples = None

        self.baseline_means = None
        self.baseline_stds = None

    def calculate_baseline(self, pca_data):
        '''
        pca_data is a numpy array where the first axis is direction from most negative to positive
        and the other is a numpy array of pca components from beamformed data
        pca_data.shape = (num_channels, num_pca_components, num_samples)
        '''
        # print(f'PCA_DATA: {type(pca_data)}\t|\t{pca_data.shape}')
        # PCA_DATA: <class 'numpy.ndarray'>	|	(3, 2049)

        # If this is the first time, initialize the baseline means and stds
        if self.baseline_means is None or self.baseline_stds is None:
            self.baseline_means = np.mean(pca_data, axis=1)
            self.baseline_stds = np.std(pca_data, axis=1)
        else:
            # Update the baseline with a moving average
            current_mean = np.mean(pca_data, axis=1)
            current_std = np.std(pca_data, axis=1)

            # Moving average update for the mean and std (adjust weight if needed)
            self.baseline_means = 0.9 * self.baseline_means + 0.1 * current_mean
            self.baseline_stds = 0.9 * self.baseline_stds + 0.1 * current_std


    def detect_anomalies(self, pca_data):
        self.num_pca_components, self.num_samples = pca_data.shape
        # print(pca_data.shape)

        if not self.baseline_calculated:
            self.calculate_baseline(pca_data)

        else:

            # Ensure the baseline mean and std are correctly shaped
            baseline_mean = self.baseline_means.reshape(self.num_pca_components, 1)
            baseline_std = self.baseline_stds.reshape(self.num_pca_components, 1)

            # Compute the deviation from the baseline
            deviations = np.abs(pca_data - baseline_mean) > (baseline_std * self.anomaly_threshold)
            # print(deviations)

            # Count the number of anomalies (deviations exceeding threshold)
            num_anomalies = np.sum(deviations)

            return num_anomalies



if __name__ == '__main__':
    pass