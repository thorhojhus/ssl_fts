import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

class SyntheticDatasetGenerator:
    def __init__(self, length):
        self.length = length
        self.time_series = np.arange(length)
        self.data = np.zeros(length)
    
    def add_linear_trend(self, slope=1, intercept=0):
        self.data += slope * self.time_series + intercept
    
    def add_sin_wave(self, amplitude=1, frequency=1):
        self.data += amplitude * np.sin(2 * np.pi * frequency * self.time_series / self.length)
    
    def add_noise(self, mean=0, std=1):
        self.data += np.random.normal(mean, std, self.length)
    
    def add_exponential_growth(self, base=1.01):
        self.data += base ** self.time_series
    
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

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset.")
    parser.add_argument("--length", type=int, default=1000, help="Length of the dataset.")
    parser.add_argument("--slope", type=float, default=0, help="Slope of the linear trend.")
    parser.add_argument("--intercept", type=float, default=0, help="Intercept of the linear trend.")
    parser.add_argument("--amplitude", type=float, default=0, help="Amplitude of the sine wave.")
    parser.add_argument("--frequency", type=float, default=0, help="Frequency of the sine wave.")
    parser.add_argument("--mean", type=float, default=0, help="Mean of the noise.")
    parser.add_argument("--std", type=float, default=0, help="Standard deviation of the noise.")
    parser.add_argument("--base", type=float, default=0, help="Base of the exponential growth.")
    args = parser.parse_args()

    generator = SyntheticDatasetGenerator(args.length)
    generator.add_linear_trend(args.slope, args.intercept)
    generator.add_sin_wave(args.amplitude, args.frequency)
    generator.add_noise(args.mean, args.std)
    generator.add_exponential_growth(args.base)
    generator.plot_data()

if __name__ == "__main__":
    main()

#example use
#python synthgen.py --length 1500 --slope 1.0 --intercept 1.0 --amplitude 2 --frequency 0.2 --mean 0.1 --std 2.0 --base 1.001

#python synthgen.py --length 1500 --mean 0.1 --std 2.0 
