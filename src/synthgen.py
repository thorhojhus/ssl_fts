import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Function to create a synthetic time series
def create_time_series(start_date, end_date, freq, functions, noise_std=1.0, seed=None):
    if seed:
        np.random.seed(seed)

    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    time_series_data = pd.DataFrame(index=dates)

    # Initialize the series
    series = np.zeros(len(dates))

    # Apply each function to the series
    for func in functions:
        series += func(dates)

    # Add noise
    noise = np.random.normal(0, noise_std, len(dates))
    series += noise

    # Save to DataFrame
    time_series_data['value'] = series

    return time_series_data

# Example functions
def trend(dates):
    return np.linspace(0, 10, len(dates))

def seasonality(dates, period=365.25, amplitude=10):
    return amplitude * np.sin(2 * np.pi * dates.dayofyear / period)

def non_linear_effect(dates):
    return 5 * np.log1p(dates.dayofyear)

# Parameters
start_date = '2020-01-01'
end_date = '2023-01-01'
freq = 'D'  # Daily frequency

# List of functions to apply
functions = [trend, seasonality, non_linear_effect]

# Create the synthetic time series
time_series_data = create_time_series(start_date, end_date, freq, functions, noise_std=2.0, seed=42)

# Save to CSV
csv_file = 'synthetic_time_series.csv'
time_series_data.to_csv(csv_file)

# Plot the generated time series
plt.figure(figsize=(12, 6))
plt.plot(time_series_data.index, time_series_data['value'], label='Synthetic Time Series')
plt.title('Synthetic Time Series')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
