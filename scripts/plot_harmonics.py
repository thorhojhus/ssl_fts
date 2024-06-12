import matplotlib.pyplot as plt
import numpy as np

#from synthgen import SyntheticDatasetGenerator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

class SyntheticDatasetGenerator:
    def __init__(self, length):
        self.length = length
        self.time_series = np.arange(length)
        self.data = np.zeros(length)
        self.shift_counter = 0
        self.signal_counter = 0
    
    def add_linear_trend(self, slope, intercept):
        self.data += slope * self.time_series + intercept
    
    def add_sin_wave(self, amplitude, frequency):
        self.data += amplitude * np.sin(2 * np.pi * frequency * self.time_series / self.length)
    
    def add_noise(self, mean=0, std=1):
        self.data += np.random.normal(mean, std, self.length)
    
    def add_exponential_growth(self, base):
        self.data += base ** self.time_series
    
    def add_mean_shift(self, shift_magnitude, num_shifts, gaussian=False):
        for i in range(num_shifts):
            start = np.random.randint(0, self.length)
            sign = np.random.choice([-1, 1])
            if gaussian:
                magnitude = np.random.normal(shift_magnitude, abs(shift_magnitude) * 0.1)
            else:
                magnitude = shift_magnitude
            self.data[start:] += sign * magnitude
    
    def add_balanced_mean_shifts(self, shift_magnitude, num_shifts, gaussian=False):
        num_positive_shifts = num_shifts // 2
        num_negative_shifts = num_shifts - num_positive_shifts
        
        for _ in range(num_positive_shifts):
            self.add_single_mean_shift(shift_magnitude, gaussian, sign=1)
        
        for _ in range(num_negative_shifts):
            self.add_single_mean_shift(shift_magnitude, gaussian, sign=-1)
    
    def add_single_mean_shift(self, shift_magnitude, gaussian, sign):
        start = np.random.randint(0, self.length)
        if gaussian:
            magnitude = np.random.normal(shift_magnitude, abs(shift_magnitude) * 0.1)
        else:
            magnitude = shift_magnitude
        self.data[start:] += sign * magnitude
    
    def add_random_signal_with_precursor(self, precursor_amplitude, signal_amplitude, max_precursor_length, min_delay, max_delay, num_signals):
        for _ in range(num_signals):
            start = np.random.randint(0, self.length)
            precursor_length = np.random.randint(1, max_precursor_length)
            precursor_end = min(start + precursor_length, self.length)
            
            delay = np.random.randint(min_delay, max_delay)
            signal_start = precursor_end + delay
            signal_end = min(signal_start + precursor_length, self.length)
            
            sign = (-1) ** self.signal_counter
            if precursor_end < self.length:
                self.data[start:precursor_end] += sign * precursor_amplitude
            if signal_start < self.length:
                self.data[signal_start:signal_end] += sign * signal_amplitude
            self.signal_counter += 1
    
    def save_to_csv(self, filename):
        df = pd.DataFrame({"time": self.time_series, "value": self.data})
        df.to_csv(filename, index=False)

    def plot_data(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.time_series, self.data, label="Synthetic Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Synthetic Dataset")
        plt.legend()
        plt.show()


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

fs = 400
cutoffs = [120, 60, 40, 20]


original_signal = ds.data[:fs]

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

filtered_signals = [low_pass_filter(original_signal, cutoff, fs) for cutoff in cutoffs]
t = np.linspace(0, len(original_signal) / fs, len(original_signal))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))

# Original signal
axes[0, 0].plot(t * fs, original_signal, "b", linestyle="--")
axes[0, 0].set_title("No Filter")
axes[0, 0].set_ylim(-4, 4)
axes[1, 0].magnitude_spectrum(original_signal.flatten(), Fs=fs, color="b")
axes[1, 0].set_title("Original")

# Filtered signals
titles = ["COF: 120", "COF: 60", "COF: 40", "COF: 20"]
harmonics = [6, 3, 2, 1]
for i in range(4):
    axes[0, i + 1].plot(t * fs, original_signal, "b", linewidth=0.5, linestyle="--")
    axes[0, i + 1].plot(t * fs, filtered_signals[i], "orange", linewidth=2)
    axes[0, i + 1].set_title(titles[i])
    axes[0, i + 1].set_ylim(-4, 4)
    # remove ticks
    axes[0, i + 1].set_yticks([])

    axes[1, i + 1].magnitude_spectrum(filtered_signals[i].flatten(), Fs=fs, color="orange")
    axes[1, i + 1].axvline(x=harmonics[i] * 20, color="gray", linestyle="--")
    axes[1, i + 1].set_title(f"COF at {harmonics[i]} harmonic")

    # remove ticks
    axes[1, i + 1].set_yticks([])

    # remove y label
    axes[1, i + 1].set_ylabel("")
    

plt.tight_layout()
plt.show()
