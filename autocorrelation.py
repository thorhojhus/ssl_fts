import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import acf

directory = 'dataset'
out_dir = "plots"
files = os.listdir(directory)
max_lag = 7589 # exchange_rate
files = ['exchange_rate.csv']

# max_lag = 15725 # stocks
# files = ['MRO.csv', 'GD.csv']

for file in files:
    file_path = os.path.join(directory, file)
    data = pd.read_csv(file_path, parse_dates=['date'])
    data = data.set_index('date')
    data = data.resample('h').mean()

    plt.figure(figsize=(12,6))
    
    n_colors = len(data.columns)
    colors = plt.cm.brg(np.linspace(0, 1, n_colors))

    all_acf_values = []
    for i, column in enumerate(data.columns):
        series = data[column].dropna()
        autocorr = acf(series, nlags=max_lag)
        lags = np.arange(len(autocorr))    
        plt.plot(lags, autocorr, linestyle='-', label=column)
        all_acf_values.append(autocorr)
    
    all_acf_values = np.array(all_acf_values)
    mean_acf = np.mean(all_acf_values, axis=0)
    plt.plot(range(len(mean_acf)), mean_acf, color='black', linewidth=3)
    
    plt.title(f'Autocorrelation Plot of {file.replace(".csv", "")}')
    plt.xlabel('Time Lag (years)')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.xlim(0, max_lag)
    
    # Adjust xticks
    xticks = np.arange(0, max_lag + 1, 365*2)  # Every 2 years
    plt.xticks(xticks, [f"{x//365}" for x in xticks])
    # plt.xticks(rotation=45)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plot_file_path = os.path.join(out_dir, f'autocorr_plot_{file.replace(".csv", ".png")}')
    plt.savefig(plot_file_path, dpi=300, bbox_inches='tight') 
    plt.close()
    print(f'Autocorrelation plot saved to {plot_file_path}')