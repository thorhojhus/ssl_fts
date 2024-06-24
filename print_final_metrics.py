import os
import glob
import pandas as pd

def parse_metrics(file_content):
    metrics = {}
    lines = file_content.split('\n')
    for line in lines:
        if 'MSE' in line and ('Repeat' in line or 'FITS' in line or 'DLinear' in line):
            parts = line.split()
            model_name = 'Repeat' if 'Repeat' in line else parts[0].strip()
            mse = float(parts[2])
            mae = float(parts[4])
            se = float(parts[6])
            rrmse = float(parts[8][:-1])  # Remove the trailing '%' character
            metrics[model_name] = {'MSE': mse, 'MAE': mae, 'SE': se, 'RRMSE': rrmse}
    return metrics

def get_metrics(dataset, seq_len, pred_len, model):
    if model == 'Repeat':
        # Look for Repeat data in any model's log file
        log_file_pattern = f'logs/LongForecasting/*_{dataset}_{seq_len}_{pred_len}.log'
    else:
        log_file_pattern = f'logs/LongForecasting/{model}*_{dataset}_{seq_len}_{pred_len}.log'
    
    log_files = glob.glob(log_file_pattern)
    if log_files:
        with open(log_files[0], 'r') as f:
            content = f.read()
            metrics = parse_metrics(content)
            if model in metrics:
                return metrics[model]
    return None

def print_metrics_table(datasets, seq_len, pred_lengths, models):
    print(f"{'Dataset':<15} {'Exchange Rate':<40} {'General Dynamics $GD':<40} {'Marathon Oil Corp $MRO':<40}")
    print(f"{'Model':<15} {'(4 col)':<40} {'(4 col)':<40} {'(4 col)':<40}")
    
    for model in models:
        print(f"\n{model}")
        print(f"{'Horizon':<7} {'MSE':>10} {'MAE':>10} {'SE':>10} {'RRMSE':>10} {'MSE':>10} {'MAE':>10} {'SE':>10} {'RRMSE':>10} {'MSE':>10} {'MAE':>10} {'SE':>10} {'RRMSE':>10}")
        
        for pred_len in pred_lengths:
            row = f"{pred_len:<7}"
            for dataset in datasets:
                metrics = get_metrics(dataset, seq_len, pred_len, model)
                if metrics:
                    row += f"{metrics['MSE']:10.3f} {metrics['MAE']:10.3f} {metrics['SE']:10.3f} {metrics['RRMSE']:9.2f}%"
                else:
                    row += f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            print(row)

# Example usage
datasets = ['exchange_rate', 'GD', 'MRO']
seq_len = 336
pred_lengths = [96, 192, 336, 720]
models = ['Repeat', 'DLinear', 'FITS', 'DLinear_FITS', 'FITS_DLinear']

print_metrics_table(datasets, seq_len, pred_lengths, models)