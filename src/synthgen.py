import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


class SyntheticDatasetGenerator:
    def __init__(self, length: int, num_channels: int = 0):
        self.length = length
        self.time_series = np.arange(length)
        self.channels = num_channels
        self.data = np.zeros((length, num_channels))
        self.shift_counter = 0
        self.signal_counter = 0

    def add_relative_amplitude_lag(
        self,
        lag: int,
        amplitude_proportion: float,
        lag_channel: int,
        reference_channel: int,
    ):
        for i in range(lag, self.length):
            self.data[i, lag_channel] = (
                amplitude_proportion * self.data[i - lag, reference_channel]
            )

    def add_linear_trend(self, slope, intercept, channel: int = 0):
        self.data[:, channel] += slope * self.time_series + intercept

    def add_sin_wave(self, amplitude, frequency, channel: int = 0):
        self.data[:, channel] += amplitude * np.sin(
            2 * np.pi * frequency * self.time_series / self.length
        )

    def add_noise(self, mean=0, std=1, channel: int = 0):
        self.data[:, channel] += np.random.normal(mean, std, self.length)

    def add_exponential_growth(self, base, channel: int = 0):
        self.data[:, channel] += base**self.time_series

    def add_mean_shift(
        self, shift_magnitude, num_shifts, gaussian=False, channel: int = 0
    ):
        for i in range(num_shifts):
            start = np.random.randint(0, self.length)
            sign = np.random.choice([-1, 1])
            if gaussian:
                magnitude = np.random.normal(
                    shift_magnitude, abs(shift_magnitude) * 0.1
                )
            else:
                magnitude = shift_magnitude
            self.data[start:, channel] += sign * magnitude

    def add_balanced_mean_shifts(
        self, shift_magnitude, num_shifts, gaussian=False, channel: int = 0
    ):
        num_positive_shifts = num_shifts // 2
        num_negative_shifts = num_shifts - num_positive_shifts

        for _ in range(num_positive_shifts):
            self.add_single_mean_shift(shift_magnitude, gaussian, sign=1)

        for _ in range(num_negative_shifts):
            self.add_single_mean_shift(shift_magnitude, gaussian, sign=-1)

    def add_single_mean_shift(self, shift_magnitude, gaussian, sign, channel: int = 0):
        start = np.random.randint(0, self.length)
        if gaussian:
            magnitude = np.random.normal(shift_magnitude, abs(shift_magnitude) * 0.1)
        else:
            magnitude = shift_magnitude
        self.data[start:, channel] += sign * magnitude

    def add_random_signal_with_precursor(
        self,
        precursor_amplitude,
        signal_amplitude,
        max_precursor_length,
        min_delay,
        max_delay,
        num_signals,
        channel: int = 0,
    ):
        for _ in range(num_signals):
            start = np.random.randint(0, self.length)
            precursor_length = np.random.randint(1, max_precursor_length)
            precursor_end = min(start + precursor_length, self.length)

            delay = np.random.randint(min_delay, max_delay)
            signal_start = precursor_end + delay
            signal_end = min(signal_start + precursor_length, self.length)

            sign = (-1) ** self.signal_counter
            if precursor_end < self.length:
                self.data[start:precursor_end, channel] += sign * precursor_amplitude
            if signal_start < self.length:
                self.data[signal_start:signal_end, channel] += sign * signal_amplitude
            self.signal_counter += 1

    def save_to_csv(self, filename):
        df = pd.DataFrame({"time": self.time_series, "value": self.data})
        df.to_csv(filename, index=False)

    def plot_data(self):
        plt.figure(figsize=(10, 5))
        for channel in range(self.channels):
            plt.plot(
                self.time_series,
                self.data[:, channel],
                label=f"Synthetic Data [Channel {channel}]",
            )
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Synthetic Dataset")
        plt.legend()
        plt.show()


Defaultlength = 1000000


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset.")
    parser.add_argument(
        "--length", type=int, default=1000000, help="Length of the dataset."
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=1 / Defaultlength,
        help="Slope of the linear trend.",
    )
    parser.add_argument(
        "--intercept", type=float, default=10, help="Intercept of the linear trend."
    )
    parser.add_argument(
        "--amplitude", type=float, default=5, help="Amplitude of the sine wave."
    )
    parser.add_argument(
        "--frequency", type=float, default=5, help="Frequency of the sine wave."
    )
    parser.add_argument("--mean", type=float, default=0, help="Mean of the noise.")
    parser.add_argument(
        "--std", type=float, default=2, help="Standard deviation of the noise."
    )
    parser.add_argument(
        "--base",
        type=float,
        default=1 + (1 / Defaultlength),
        help="Base of the exponential growth.",
    )
    parser.add_argument(
        "--shift_magnitude", type=float, default=10, help="Magnitude of the mean shift."
    )
    parser.add_argument(
        "--num_shifts",
        type=int,
        default=round(Defaultlength / 1000),
        help="Number of mean shifts.",
    )
    parser.add_argument(
        "--gaussian_shifts",
        type=str,
        choices=["true", "false"],
        default="true",
        help="If true, mean shifts will be Gaussian distributed around the shift magnitude.",
    )
    parser.add_argument(
        "--precursor_amplitude",
        type=float,
        default=10,
        help="Amplitude of the precursor signal.",
    )
    parser.add_argument(
        "--signal_amplitude",
        type=float,
        default=20,
        help="Amplitude of the following signal.",
    )
    parser.add_argument(
        "--max_precursor_length",
        type=int,
        default=round(Defaultlength / 2000),
        help="Max length of the precursor signal.",
    )
    parser.add_argument(
        "--min_delay",
        type=int,
        default=round(Defaultlength / 2000),
        help="Min delay between precursor and following signal.",
    )
    parser.add_argument(
        "--max_delay",
        type=int,
        default=round(Defaultlength / 1000),
        help="Max delay between precursor and following signal.",
    )
    parser.add_argument(
        "--num_signals",
        type=int,
        default=round(Defaultlength / 10000),
        help="Number of random signals.",
    )
    args = parser.parse_args()

    gaussian_shifts = args.gaussian_shifts.lower() == "true"

    generator = SyntheticDatasetGenerator(args.length)
    generator.add_linear_trend(args.slope, args.intercept)
    generator.add_sin_wave(args.amplitude, args.frequency)
    generator.add_noise(args.mean, args.std)
    generator.add_exponential_growth(args.base)
    if args.shift_magnitude != 0:
        generator.add_balanced_mean_shifts(
            args.shift_magnitude, args.num_shifts, gaussian_shifts
        )
    if args.precursor_amplitude != 0 and args.signal_amplitude != 0:
        generator.add_random_signal_with_precursor(
            args.precursor_amplitude,
            args.signal_amplitude,
            args.max_precursor_length,
            args.min_delay,
            args.max_delay,
            args.num_signals,
        )
    generator.plot_data()


if __name__ == "__main__":
    main()
