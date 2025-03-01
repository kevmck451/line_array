from acoustic.audio import Audio
from process import average_spectrum
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.interpolate import interp1d
import matplotlib.ticker as ticker


def parse_filename(filename):
    """Extracts distance, vehicle type, number, and processing type from filename."""
    match = re.match(r'(\d+m)_(tank|truck)(\d)_(raw|beam12|beammix)', filename)
    return match.groups() if match else None


if __name__ == '__main__':
    base_path = Path('/Users/KevMcK/Dropbox/1 EE Degree/7996 Thesis/3 Data/Line Array/Field Test/Test 1 1-29/Samples/samples')

    spectra_data = {
        'truck': {'raw': {}, 'beam12': {}, 'beammix': {}},
        'tank': {'raw': {}, 'beam12': {}, 'beammix': {}}
    }
    freq_bins_ref = None  # Reference frequency bins

    for filepath in base_path.iterdir():
        if filepath.suffix == '.wav':
            parsed = parse_filename(filepath.stem)
            if parsed:
                distance, vehicle, number, process_type = parsed
                number = int(number)  # Convert to integer for correct sorting

                # Load and process audio
                audio = Audio(filepath=filepath, num_channels=1)
                av_spectrum, frequency_bins = average_spectrum(audio, display=False)

                if freq_bins_ref is None:
                    freq_bins_ref = frequency_bins  # First valid file sets reference bins

                # Ensure uniform frequency resolution
                if len(frequency_bins) != len(freq_bins_ref) or not np.allclose(frequency_bins, freq_bins_ref):
                    interp_func = interp1d(frequency_bins, av_spectrum, kind='linear', bounds_error=False, fill_value=0)
                    av_spectrum = interp_func(freq_bins_ref)

                # Store all data
                category = f"{distance}"
                if category not in spectra_data[vehicle][process_type]:
                    spectra_data[vehicle][process_type][category] = []
                spectra_data[vehicle][process_type][category].append((number, av_spectrum))

    # Create 6 subplots (3 for trucks, 3 for tanks)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    process_types = ['raw', 'beam12', 'beammix']
    titles = ['Raw Truck', 'Beam12 Truck', 'Beammix Truck', 'Raw Tank', 'Beam12 Tank', 'Beammix Tank']

    for col, vehicle in enumerate(['truck', 'tank']):
        for row, process_type in enumerate(process_types):
            ax = axes[row, col]
            ax.set_title(titles[row + (3 * col)], fontweight='bold')

            for category, spectra in spectra_data[vehicle][process_type].items():
                # Sort by sample number (ensuring order 1, 2, 3, 4, 5)
                spectra.sort()

                # Plot each sample separately instead of averaging
                for num, spectrum in spectra:
                    ax.plot(freq_bins_ref, spectrum, label=f"{category} Sample {num}")

            ax.set_xscale('symlog')
            ax.set_xlim([10, 10000])
            ax.set_xlabel('Frequency (Hz)', fontweight='bold')
            ax.set_ylabel('Magnitude', fontweight='bold')

            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='auto'))
            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=12))
            ax.grid(True, which='both')
            ax.legend()

    plt.tight_layout(pad=1)
    plt.show()
