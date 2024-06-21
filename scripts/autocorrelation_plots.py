import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

directory = '../data'
out_dir = "../plots"
files = os.listdir(directory)
max_lag = 500


excluded_files = ['MRO.csv', 'gd.csv']
files = [file for file in files if file not in excluded_files and file.endswith('.csv')]

def autocorrelation(x, max_lag):
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode='full')[-n:]
    result = r / (variance * (np.arange(n, 0, -1)))
    return result[:max_lag]

for file in files:
    file_path = os.path.join(directory, file)
    data = pd.read_csv(file_path, parse_dates=['date'])
    data = data.set_index('date')
    data = data.resample('h').mean()
    
    plt.figure(figsize=(14,7))
    
    for column in data.columns:
        series = data[column].dropna()
        autocorr = autocorrelation(series, max_lag)
        lags = np.arange(len(autocorr))    
        plt.plot(lags, autocorr, linestyle='-', label=column)
    
    plt.title(f'Autocorrelation Plot of {file.replace(".csv", "")}')
    plt.xlabel('Time Lag (hours)')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.xlim(0, max_lag)
    
    plot_file_path = os.path.join(directory, f'autocorr_plot_{file.replace(".csv", ".png")}')
    plt.savefig(plot_file_path, dpi=300) 
    plt.close()
    print(f'Autocorrelation plot saved to {plot_file_path}')