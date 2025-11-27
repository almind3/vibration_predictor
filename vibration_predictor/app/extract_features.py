import numpy as np


def extract_features(data, sampling_rate):
    data_length = len(data)
    data = data - np.mean(data)
    data = data * np.hanning(len(data))

    # time domain features
    rms = np.sqrt(np.mean(data ** 2))
    p2p = np.max(data) - np.min(data)
    crest = np.max(data) / rms

    # zero crossings
    crossings = 0
    for i in range(len(data) - 1):
        if data[i] * data[i + 1] < 0:
            crossings += 1

    # frequency domain
    fft_vals = np.fft.rfft(data)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(data_length, 1 / sampling_rate)

    # frequency band energy
    band1_idx = np.argwhere(freqs <= 1e6)
    band2_idx = np.where((1e6 < freqs) & (freqs <= 5e6))
    band3_idx = np.where((5e6 < freqs) & (freqs <= 10e6))
    band4_idx = np.where((10e6 < freqs) & (freqs <= 20e6))

    band1_mag = np.mean(fft_mag[band1_idx])
    band2_mag = np.mean(fft_mag[band2_idx])
    band4_mag = np.mean(fft_mag[band4_idx])
    band3_mag = np.mean(fft_mag[band3_idx])
    centroid = np.sum(freqs * fft_mag) / np.sum(fft_mag)

    # frequency with the highest amplitude
    max_freq_idx = np.argmax(fft_mag)
    max_freq = freqs[max_freq_idx]

    features = [rms, p2p, crest, crossings, band1_mag, band2_mag, band3_mag, band4_mag, centroid, max_freq]
    return features
