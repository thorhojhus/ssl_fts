import re

def parse_log_output(log_content):
    data = {}
    current_model = None
    current_horizon = None
    current_dataset = None
    
    for line in log_content.split('\n'):
        if 'exchange_rate' in line:
            current_dataset = 'Exchange Rate'
        elif 'general_dynamics' in line:
            current_dataset = 'General Dynamics'
        elif 'pred_len=' in line:
            current_horizon = int(re.search(r'pred_len=(\d+)', line).group(1))
        elif line.startswith(('DLinear', 'FITS', 'DLinear_FITS', 'Repeat')):
            parts = line.split()
            current_model = parts[0]
            if current_model == 'DLinear_FITS':
                current_model = 'DLinear + FITS'
            if current_model not in data:
                data[current_model] = {}
            if current_dataset not in data[current_model]:
                data[current_model][current_dataset] = {}
            if current_horizon not in data[current_model][current_dataset]:
                data[current_model][current_dataset][current_horizon] = {}
            
            data[current_model][current_dataset][current_horizon]['MSE'] = float(parts[2])
            data[current_model][current_dataset][current_horizon]['MAE'] = float(parts[4])
            data[current_model][current_dataset][current_horizon]['SE'] = float(parts[6])
            data[current_model][current_dataset][current_horizon]['RRMSE'] = parts[8]
    
    return data

def merge_data(existing_data, new_data):
    for model in new_data:
        if model not in existing_data:
            existing_data[model] = {}
        for dataset in new_data[model]:
            if dataset not in existing_data[model]:
                existing_data[model][dataset] = {}
            for horizon in new_data[model][dataset]:
                existing_data[model][dataset][horizon] = new_data[model][dataset][horizon]
    return existing_data

def fill_latex_table(data):
    table_lines = []
    models = ['Repeat', 'DLinear', 'FITS', 'DLinear + FITS']
    horizons = [96, 192, 336, 720]
    datasets = ['Exchange Rate', 'General Dynamics']
    
    for model in models:
        for i, horizon in enumerate(horizons):
            line = ""
            if i == 0:
                line += f"\\multirow{{4}}{{*}}{{{model}}}\n"
            
            line += f"& {horizon}"
            
            for dataset in datasets:
                if model in data and dataset in data[model] and horizon in data[model][dataset]:
                    metrics = data[model][dataset][horizon]
                    line += f" & {metrics['MSE']:.3f} & {metrics['MAE']:.3f} & {metrics['SE']:.3f} & {metrics['RRMSE']}"
                else:
                    line += f" & & & &"
            
            line += " \\\\"
            table_lines.append(line)
        
        if model != models[-1]:
            table_lines.append("\\midrule")
    
    return "\n".join(table_lines)

# Existing data (from previous run)
existing_data = {
    'Repeat': {'Exchange Rate': {96: {'MSE': 0.081, 'MAE': 0.196, 'SE': 0.164, 'RRMSE': '19.53%'}, 192: {'MSE': 0.167, 'MAE': 0.289, 'SE': 0.342, 'RRMSE': '28.41%'}, 336: {'MSE': 0.306, 'MAE': 0.397, 'SE': 0.626, 'RRMSE': '38.85%'}, 720: {'MSE': 0.813, 'MAE': 0.677, 'SE': 1.781, 'RRMSE': '63.52%'}}},
    'DLinear': {'Exchange Rate': {96: {'MSE': 0.081, 'MAE': 0.203, 'SE': 0.184, 'RRMSE': '19.51%'}, 192: {'MSE': 0.157, 'MAE': 0.293, 'SE': 0.321, 'RRMSE': '27.51%'}, 336: {'MSE': 0.299, 'MAE': 0.417, 'SE': 0.492, 'RRMSE': '38.44%'}, 720: {'MSE': 1.040, 'MAE': 0.767, 'SE': 1.804, 'RRMSE': '71.86%'}}},
    'FITS': {'Exchange Rate': {96: {'MSE': 0.101, 'MAE': 0.228, 'SE': 0.211, 'RRMSE': '21.78%'}, 192: {'MSE': 0.199, 'MAE': 0.325, 'SE': 0.408, 'RRMSE': '30.99%'}, 336: {'MSE': 0.352, 'MAE': 0.436, 'SE': 0.750, 'RRMSE': '41.69%'}, 720: {'MSE': 0.993, 'MAE': 0.764, 'SE': 2.144, 'RRMSE': '70.22%'}}},
    'DLinear + FITS': {'Exchange Rate': {96: {'MSE': 0.086, 'MAE': 0.213, 'SE': 0.160, 'RRMSE': '20.20%'}, 192: {'MSE': 0.164, 'MAE': 0.298, 'SE': 0.298, 'RRMSE': '28.18%'}, 336: {'MSE': 0.356, 'MAE': 0.453, 'SE': 0.675, 'RRMSE': '41.93%'}, 720: {'MSE': 0.957, 'MAE': 0.734, 'SE': 2.113, 'RRMSE': '68.91%'}}}
}

# New log content (for General Dynamics data)
log_content2 = """
Running experiments for seq_len=336 pred_len=96:
logs/LongForecasting/DLinear3.0_GD_336_96.log
Final Metrics:
DLinear      MSE: 1.659      MAE: 0.948      SE: 2.356      RRMSE: 12.93%     | RRMSE: 0.129      RMAE: 0.095      (numpy)
DLinear      MSE: 1.659      MAE: 0.948      SE: 2.356      RRMSE: 12.93%     | RRMSE: 0.129      RMAE: 0.095      (torch)

Repeat       MSE: 1.196      MAE: 0.763      SE: 2.283      RRMSE: 10.98%     | RRMSE: 0.110      RMAE: 0.077      (numpy)
Repeat       MSE: 1.196      MAE: 0.763      SE: 2.283      RRMSE: 10.98%     | RRMSE: 0.110      RMAE: 0.077      (torch)


logs/LongForecasting/FITS3.0_GD_336_96.log
Final Metrics:
FITS         MSE: 1.620      MAE: 0.929      SE: 2.670      RRMSE: 12.78%     | RRMSE: 0.128      RMAE: 0.093      (numpy)
FITS         MSE: 1.620      MAE: 0.929      SE: 2.670      RRMSE: 12.78%     | RRMSE: 0.128      RMAE: 0.093      (torch)

Repeat       MSE: 1.196      MAE: 0.763      SE: 2.283      RRMSE: 10.98%     | RRMSE: 0.110      RMAE: 0.077      (numpy)
Repeat       MSE: 1.196      MAE: 0.763      SE: 2.283      RRMSE: 10.98%     | RRMSE: 0.110      RMAE: 0.077      (torch)


logs/LongForecasting/DLinear_FITS3.0_GD_336_96.log
Final Metrics:
DLinear_FITS MSE: 1.433      MAE: 0.855      SE: 3.108      RRMSE: 12.02%     | RRMSE: 0.120      RMAE: 0.086      (numpy)
DLinear_FITS MSE: 1.433      MAE: 0.855      SE: 3.108      RRMSE: 12.02%     | RRMSE: 0.120      RMAE: 0.086      (torch)

Repeat       MSE: 1.196      MAE: 0.763      SE: 2.283      RRMSE: 10.98%     | RRMSE: 0.110      RMAE: 0.077      (numpy)
Repeat       MSE: 1.196      MAE: 0.763      SE: 2.283      RRMSE: 10.98%     | RRMSE: 0.110      RMAE: 0.077      (torch)


Running experiments for seq_len=336 pred_len=192:
logs/LongForecasting/DLinear3.0_GD_336_192.log
Final Metrics:
DLinear      MSE: 2.430      MAE: 1.153      SE: 4.825      RRMSE: 15.66%     | RRMSE: 0.157      RMAE: 0.116      (numpy)
DLinear      MSE: 2.430      MAE: 1.153      SE: 4.825      RRMSE: 15.66%     | RRMSE: 0.157      RMAE: 0.116      (torch)

Repeat       MSE: 2.250      MAE: 1.077      SE: 4.923      RRMSE: 15.07%     | RRMSE: 0.151      RMAE: 0.108      (numpy)
Repeat       MSE: 2.250      MAE: 1.077      SE: 4.923      RRMSE: 15.07%     | RRMSE: 0.151      RMAE: 0.108      (torch)


logs/LongForecasting/FITS3.0_GD_336_192.log
Final Metrics:
FITS         MSE: 2.787      MAE: 1.236      SE: 5.564      RRMSE: 16.77%     | RRMSE: 0.168      RMAE: 0.124      (numpy)
FITS         MSE: 2.787      MAE: 1.236      SE: 5.564      RRMSE: 16.77%     | RRMSE: 0.168      RMAE: 0.124      (torch)

Repeat       MSE: 2.250      MAE: 1.077      SE: 4.923      RRMSE: 15.07%     | RRMSE: 0.151      RMAE: 0.108      (numpy)
Repeat       MSE: 2.250      MAE: 1.077      SE: 4.923      RRMSE: 15.07%     | RRMSE: 0.151      RMAE: 0.108      (torch)


logs/LongForecasting/DLinear_FITS3.0_GD_336_192.log
Final Metrics:
DLinear_FITS MSE: 2.387      MAE: 1.136      SE: 4.725      RRMSE: 15.52%     | RRMSE: 0.155      RMAE: 0.114      (numpy)
DLinear_FITS MSE: 2.387      MAE: 1.136      SE: 4.725      RRMSE: 15.52%     | RRMSE: 0.155      RMAE: 0.114      (torch)

Repeat       MSE: 2.250      MAE: 1.077      SE: 4.923      RRMSE: 15.07%     | RRMSE: 0.151      RMAE: 0.108      (numpy)
Repeat       MSE: 2.250      MAE: 1.077      SE: 4.923      RRMSE: 15.07%     | RRMSE: 0.151      RMAE: 0.108      (torch)


Running experiments for seq_len=336 pred_len=336:
logs/LongForecasting/DLinear3.0_GD_336_336.log
Final Metrics:
DLinear      MSE: 4.189      MAE: 1.551      SE: 8.089      RRMSE: 20.50%     | RRMSE: 0.205      RMAE: 0.155      (numpy)
DLinear      MSE: 4.189      MAE: 1.551      SE: 8.089      RRMSE: 20.50%     | RRMSE: 0.205      RMAE: 0.155      (torch)

Repeat       MSE: 3.899      MAE: 1.444      SE: 8.558      RRMSE: 19.78%     | RRMSE: 0.198      RMAE: 0.145      (numpy)
Repeat       MSE: 3.899      MAE: 1.444      SE: 8.558      RRMSE: 19.78%     | RRMSE: 0.198      RMAE: 0.145      (torch)


logs/LongForecasting/FITS3.0_GD_336_336.log
Final Metrics:
FITS         MSE: 4.900      MAE: 1.662      SE: 10.084     RRMSE: 22.17%     | RRMSE: 0.222      RMAE: 0.166      (numpy)
FITS         MSE: 4.900      MAE: 1.662      SE: 10.084     RRMSE: 22.17%     | RRMSE: 0.222      RMAE: 0.166      (torch)

Repeat       MSE: 3.899      MAE: 1.444      SE: 8.558      RRMSE: 19.78%     | RRMSE: 0.198      RMAE: 0.145      (numpy)
Repeat       MSE: 3.899      MAE: 1.444      SE: 8.558      RRMSE: 19.78%     | RRMSE: 0.198      RMAE: 0.145      (torch)


logs/LongForecasting/DLinear_FITS3.0_GD_336_336.log
Final Metrics:
DLinear_FITS MSE: 4.248      MAE: 1.551      SE: 7.639      RRMSE: 20.64%     | RRMSE: 0.206      RMAE: 0.155      (numpy)
DLinear_FITS MSE: 4.248      MAE: 1.551      SE: 7.639      RRMSE: 20.64%     | RRMSE: 0.206      RMAE: 0.155      (torch)

Repeat       MSE: 3.899      MAE: 1.444      SE: 8.558      RRMSE: 19.78%     | RRMSE: 0.198      RMAE: 0.145      (numpy)
Repeat       MSE: 3.899      MAE: 1.444      SE: 8.558      RRMSE: 19.78%     | RRMSE: 0.198      RMAE: 0.145      (torch)


Running experiments for seq_len=336 pred_len=720:
logs/LongForecasting/DLinear3.0_GD_336_720.log
Final Metrics:
DLinear      MSE: 8.923      MAE: 2.341      SE: 15.157     RRMSE: 29.70%     | RRMSE: 0.297      RMAE: 0.233      (numpy)
DLinear      MSE: 8.923      MAE: 2.341      SE: 15.157     RRMSE: 29.70%     | RRMSE: 0.297      RMAE: 0.233      (torch)

Repeat       MSE: 9.859      MAE: 2.366      SE: 21.451     RRMSE: 31.22%     | RRMSE: 0.312      RMAE: 0.235      (numpy)
Repeat       MSE: 9.859      MAE: 2.366      SE: 21.451     RRMSE: 31.22%     | RRMSE: 0.312      RMAE: 0.235      (torch)


logs/LongForecasting/FITS3.0_GD_336_720.log
Final Metrics:
FITS         MSE: 11.310     MAE: 2.630      SE: 21.824     RRMSE: 33.44%     | RRMSE: 0.334      RMAE: 0.262      (numpy)
FITS         MSE: 11.310     MAE: 2.630      SE: 21.824     RRMSE: 33.44%     | RRMSE: 0.334      RMAE: 0.262      (torch)

Repeat       MSE: 9.859      MAE: 2.366      SE: 21.451     RRMSE: 31.22%     | RRMSE: 0.312      RMAE: 0.235      (numpy)
Repeat       MSE: 9.859      MAE: 2.366      SE: 21.451     RRMSE: 31.22%     | RRMSE: 0.312      RMAE: 0.235      (torch)


logs/LongForecasting/DLinear_FITS3.0_GD_336_720.log
Final Metrics:
DLinear_FITS MSE: 9.833      MAE: 2.454      SE: 15.653     RRMSE: 31.18%     | RRMSE: 0.312      RMAE: 0.244      (numpy)
DLinear_FITS MSE: 9.833      MAE: 2.454      SE: 15.653     RRMSE: 31.18%     | RRMSE: 0.312      RMAE: 0.244      (torch)

Repeat       MSE: 9.859      MAE: 2.366      SE: 21.451     RRMSE: 31.22%     | RRMSE: 0.312      RMAE: 0.235      (numpy)
Repeat       MSE: 9.859      MAE: 2.366      SE: 21.451     RRMSE: 31.22%     | RRMSE: 0.312      RMAE: 0.235      (torch)


"""

new_data = parse_log_output(log_content2)
merged_data = merge_data(existing_data, new_data)

latex_table_content = fill_latex_table(merged_data)
print(latex_table_content)