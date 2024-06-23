import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf

directory = '../data'
out_dir = "../plots"
files = os.listdir(directory)
max_lag = 24*10

excluded_files = ['MRO.csv', 'gd.csv', 'national_illness.csv']
#files = ["national_illness.csv"]
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
    
    plt.figure(figsize=(10,6))
    
    n_colors = len(data.columns)
    colors = plt.cm.brg(np.linspace(0, 1, n_colors))

    all_acf_values = []
    for i, column in enumerate(data.columns):
        series = data[column].dropna()
        autocorr = acf(series, nlags=max_lag)
        lags = np.arange(len(autocorr))    
        plt.plot(lags, autocorr, linestyle='-', label=column)#, color=colors[i])
        all_acf_values.append(autocorr)
    
    all_acf_values = np.array(all_acf_values)
    mean_acf = np.mean(all_acf_values, axis=0)
    plt.plot(range(len(mean_acf)), mean_acf, color='black', linewidth=3)
    
    plt.title(f'Autocorrelation Plot of {file.replace(".csv", "")}')
    plt.xlabel('Time Lag (hours)')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.xlim(0, max_lag)
    
    xticks = np.arange(24, max_lag + 1, 24)
    plt.xticks(xticks)

    plot_file_path = os.path.join(out_dir, f'autocorr_plot_{file.replace(".csv", ".png")}')
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight') 
    plt.close()
    print(f'Autocorrelation plot saved to {plot_file_path}')