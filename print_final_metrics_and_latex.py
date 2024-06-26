import os
import glob
import pandas as pd
import numpy as np
import pyperclip

def parse_metrics(file_content):
    metrics = {}
    lines = file_content.split('\n')
    for line in lines:
        if 'MSE' in line and ('FITS' in line or 'DLinear' in line or 'Repeat' in line):
            parts = line.split()
            model_name = parts[0].strip()
            mse = float(parts[2])
            mae = float(parts[4])
            se = float(parts[6])
            rrmse = float(parts[8])  # RRMSE is already a decimal, no need to remove '%'
            metrics[model_name] = {'MSE': mse, 'MAE': mae, 'SE': se, 'RRMSE': rrmse * 100}  # Convert RRMSE to percentage
    return metrics

def get_metrics(log_folder, dataset, seq_len, pred_len, model):
    mode = 'MS' if 'MS_seq_len' in log_folder else 'M' if 'M_seq_len' in log_folder else 'S'
    channels = '*'  # Use wildcard for channels
    
    if model == 'Repeat':
        log_file_pattern = f'{log_folder}/*_{dataset}_{seq_len}_{pred_len}_{mode}_channels_{channels}.log'
    else:
        log_file_pattern = f'{log_folder}/*{model}*_{dataset}_{seq_len}_{pred_len}_{mode}_channels_{channels}.log'
    
    # print(f"Searching for pattern: {log_file_pattern}")  # Debug print
    
    log_files = glob.glob(log_file_pattern)
    # print(f"Found files: {log_files}")  # Debug print
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
            metrics = parse_metrics(content)
            if model in metrics:
                return metrics[model]
            elif 'Repeat' in metrics and model == 'Repeat':
                return metrics['Repeat']
    return None

def print_metrics_table(log_folder, datasets, seq_len, pred_lengths, models):
    print(f"{'Dataset':<15} {'Exchange Rate':<40} {'General Dynamics $GD':<40} {'Marathon Oil Corp $MRO':<40}")
    print(f"{'Model':<15} {'(4 col)':<40} {'(4 col)':<40} {'(4 col)':<40}")

    for model in models:
        print(f"\n{model}")
        print(f"{'Horizon':<7} {'MSE':>10} {'MAE':>10} {'SE':>10} {'RRMSE':>10} {'MSE':>10} {'MAE':>10} {'SE':>10} {'RRMSE':>10} {'MSE':>10} {'MAE':>10} {'SE':>10} {'RRMSE':>10}")
        
        for pred_len in pred_lengths:
            row = f"{pred_len:<7}"
            for dataset in datasets:
                metrics = get_metrics(log_folder, dataset, seq_len, pred_len, model)
                if metrics:
                    row += f"{metrics['MSE']:10.3f} {metrics['MAE']:10.3f} {metrics['SE']:10.3f} {metrics['RRMSE']:9.2f}%"
                else:
                    row += f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}"
            print(row)

def format_latex_table(log_folder, datasets, seq_len, pred_lengths, models):
    mode = 'MS' if 'MS_seq_len' in log_folder else 'M' if 'M_seq_len' in log_folder else 'S'

    data = {}
    for model in models:
        data[model] = {}
        for pred_len in pred_lengths:
            data[model][pred_len] = {}
            for dataset in datasets:
                metrics = get_metrics(log_folder, dataset, seq_len, pred_len, model)
                if metrics:
                    data[model][pred_len][dataset] = metrics
                else:
                    data[model][pred_len][dataset] = {'MSE': np.inf, 'MAE': np.inf, 'SE': np.inf, 'RRMSE': np.inf}

    latex_table = []
    latex_table.append(r"\begin{table*}[ht!]")
    latex_table.append(r"\centering")
    latex_table.append(r"\scalebox{0.75}{")
    latex_table.append(r"\begin{tabular}{@{}c|c|cccc|cccc|cccc|c@{}}")
    latex_table.append(r"\toprule")
    latex_table.append(r"Dataset & & \multicolumn{4}{c|}{Exchange Rate} & \multicolumn{4}{c|}{General Dynamics $\underline{GD}$} & \multicolumn{4}{c|}{Marathon Oil Corp $\underline{MRO}$} & Best \\ ")
    latex_table.append(r"\midrule")
    latex_table.append(r"Model & Horizon & MSE & MAE & SE & RRMSE & MSE & MAE & SE & RRMSE & MSE & MAE & SE & RRMSE & 48 \\ ")
    latex_table.append(r"\midrule")

    for model in models:
        latex_table.append(rf"\multirow{{4}}{{4em}}{{{model}}}")
        best_count = 0
        rows = []
        for pred_len in pred_lengths:
            row = f"& {pred_len} "
            for dataset in datasets:
                metrics = data[model][pred_len][dataset]
                for metric in ['MSE', 'MAE', 'SE', 'RRMSE']:
                    value = metrics[metric]
                    all_values = [data[m][pred_len][dataset][metric] for m in models]
                    sorted_values = sorted(set(all_values))
                    if metric == 'RRMSE':
                        if value == sorted_values[0]:
                            row += rf"& \textbf{{{value:.2f}\%}} "
                            best_count += 1
                        elif value == sorted_values[1]:
                            row += rf"& \underline{{{value:.2f}\%}} "
                        else:
                            row += f"& {value:.2f}\% "
                    else:
                        if value == sorted_values[0]:
                            row += rf"& \textbf{{{value:.3f}}} "
                            best_count += 1
                        elif value == sorted_values[1]:
                            row += rf"& \underline{{{value:.3f}}} "
                        else:
                            row += f"& {value:.3f} "
            rows.append(row)
        
        latex_table.append(rows[0] + rf"& \multirow{{4}}{{*}}{{{best_count}/48}} \\")
        for row in rows[1:]:
            latex_table.append(row + r"\\")
        latex_table.append(r"\midrule")

    latex_table.append(r"\end{tabular}")
    latex_table.append(r"}")
    if mode == 'S':
        latex_table.append(r"\caption{Univariate input on univariate columns}")
        latex_table.append(r"\label{tab:univariate_univariate}")
    elif mode == 'MS':
        latex_table.append(r"\caption{Multivariate input on univariate columns}")
        latex_table.append(r"\label{tab:multivariate_univariate}")
    elif mode == 'M':
        latex_table.append(r"\caption{Multivariate input on multivariate columns}")
        latex_table.append(r"\label{tab:multivariate_multivariate}")
    else:
        latex_table.append(r"\caption{}")
        latex_table.append(r"\label{tab:price_datasets}")

    latex_table.append(r"\end{table*}")
    
    return "\n".join(latex_table)

# Example usage
datasets = ['exchange_rate', 'GD', 'MRO']
seq_len = 336
pred_lengths = [96, 192, 336, 720]
models = ['Repeat', 'DLinear', 'FITS', 'FITS_100', 'FITS_10', 'DLinear_FITS', 'FITS_DLinear']
log_folder = 'logs/LongForecasting/S_seq_len_336'  # Change this as needed

print_metrics_table(log_folder, datasets, seq_len, pred_lengths, models)
print("\n\nLaTeX Table:\n")
latex_table = format_latex_table(log_folder, datasets, seq_len, pred_lengths, models)
print(latex_table)

# Copy the LaTeX table to clipboard
pyperclip.copy(latex_table)
print("\nLaTeX table has been copied to clipboard.")