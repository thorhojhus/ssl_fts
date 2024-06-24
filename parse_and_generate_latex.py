import re
import sys

def parse_output(file_content):
    data = {}
    current_dataset = None
    current_pred_len = None

    for line in file_content.split('\n'):
        if line.startswith('dataset='):
            current_dataset = line.split('=')[1].strip()
            if current_dataset not in data:
                data[current_dataset] = {}
        elif line.startswith('Running experiments for seq_len=336 pred_len='):
            current_pred_len = int(line.split('=')[-1].strip(':'))
        elif line.startswith(('DLinear ', 'FITS ', 'DLinear_FITS ', 'Repeat ')):
            model = line.split()[0]
            if model not in data[current_dataset]:
                data[current_dataset][model] = {}
            values = re.findall(r'\d+\.\d+', line)
            data[current_dataset][model][current_pred_len] = values[:4]

    return data

def generate_latex_table(data):
    models = ['Repeat', 'DLinear', 'FITS', 'DLinear_FITS']
    pred_lens = [96, 192, 336, 720]
    datasets = list(data.keys())

    latex_output = ""
    for model in models:
        latex_output += f"\\multirow{{4}}{{*}}{{{model.replace('_', ' + ')}}}\n"
        for pred_len in pred_lens:
            latex_output += f"& {pred_len}"
            for dataset in datasets:
                if model in data[dataset] and pred_len in data[dataset][model]:
                    values = data[dataset][model][pred_len]
                    latex_output += f" & {values[0]} & {values[1]} & {values[2]} & {values[3]}\\%"
                else:
                    latex_output += " & & & &"
            latex_output += " \\\\\n"
        latex_output += "\\midrule\n"

    return latex_output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the input file path as an argument.")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, 'r') as file:
        file_content = file.read()

    parsed_data = parse_output(file_content)
    latex_table = generate_latex_table(parsed_data)
    print(latex_table)