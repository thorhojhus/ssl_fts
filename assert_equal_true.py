import os
import numpy as np
from functools import reduce

def compare_npy_files(file1, file2):
    return np.array_equal(np.load(file1), np.load(file2))

# Define the directory path
dir_path = 'results/exchange_rate_336_720_M_channels_7'

# Function to get files with a specific ending
def get_files_with_ending(ending):
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(ending)]

# Get the full path of the files
true_files = get_files_with_ending('_true.npy')
naive_pred_files = get_files_with_ending('_naive_pred.npy')

print(f"True files found:\n{true_files}")
print(f"Naive pred files found:\n{naive_pred_files}")

def check_files(files, file_type):
    if len(files) < 2:
        print(f"Not enough {file_type} files to compare.")
    else:
        try:
            all_equal = reduce(lambda x, y: x and y, 
                               [compare_npy_files(files[0], f) for f in files[1:]])
            
            assert all_equal, f"Not all {file_type} files are the same!"
            print(f"All {file_type} files are identical.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Make sure you're running the script from the correct directory.")

print("\nChecking true files:")
check_files(true_files, "true")

print("\nChecking naive pred files:")
check_files(naive_pred_files, "naive pred")

print("\nCurrent working directory:", os.getcwd())