import matplotlib.pyplot as plt
import numpy as np

from synthgen import SyntheticDatasetGenerator

n_samples = int(1e4)

ds = SyntheticDatasetGenerator(length=n_samples)
ds.add_sin_wave(amplitude=0.3, frequency=200)
ds.add_sin_wave(amplitude=0.3, frequency=500)
ds.add_noise(mean=0, std=0.3)
ds.add_exponential_growth(1 + (3 / n_samples))
ds.add_random_signal_with_precursor(
    precursor_amplitude=2,
    signal_amplitude=4,
    max_precursor_length=200,
    min_delay=150,
    max_delay=200,
    num_signals=int(n_samples / 500),
)


original_signal = ds.data[:200]

original_signal = (original_signal - original_signal.mean()) / original_signal.std()


def low_pass_filter(signal, cutoff_frequency, fs):
    x = signal
    x_mean = np.mean(x)
    x -= x_mean
    x_var = np.var(x) + 1e-5
    x /= np.sqrt(x_var)
    low_specx = np.fft.rfft(x)
    low_specx[cutoff_frequency:] = 0
    low_x = np.fft.irfft(low_specx, n=len(x))
    x = (low_x) * np.sqrt(x_var) + x_mean
    return x


fs = 500  # Sampling frequency
cutoffs = [120, 60, 40, 20]  # Cutoff frequencies
filtered_signals = [low_pass_filter(original_signal, cutoff, fs) for cutoff in cutoffs]
t = np.linspace(0, len(original_signal) / fs, len(original_signal))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))

# Original signal
axes[0, 0].plot(t * fs, original_signal, "b", linestyle="--")
axes[0, 0].set_title("No Filter")
axes[0, 0].set_ylim(-4, 4)
axes[1, 0].magnitude_spectrum(original_signal, Fs=fs, color="b")
axes[1, 0].set_title("Original")

# Filtered signals
titles = ["COF: 120", "COF: 60", "COF: 40", "COF: 20"]
harmonics = [6, 3, 2, 1]
for i in range(4):
    axes[0, i + 1].plot(t * fs, original_signal, "b", linewidth=0.5, linestyle="--")
    axes[0, i + 1].plot(t * fs, filtered_signals[i], "orange", linewidth=2)
    axes[0, i + 1].set_title(titles[i])
    axes[0, i + 1].set_ylim(-4, 4)

    axes[1, i + 1].magnitude_spectrum(filtered_signals[i], Fs=fs, color="orange")
    axes[1, i + 1].axvline(x=harmonics[i] * 20, color="gray", linestyle="--")
    axes[1, i + 1].set_title(f"COF at {harmonics[i]} harmonic")

plt.tight_layout()
plt.show()
