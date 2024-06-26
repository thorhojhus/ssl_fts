import os
import fileinput

def process_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith('FITS_10_') and not filename.startswith('FITS_100_'):
            filepath = os.path.join(directory, filename)
            with fileinput.input(filepath, inplace=True) as file:
                for line in file:
                    if 'MSE' in line and 'FITS' in line:
                        print(line.replace('FITS', 'FITS_10'), end='')
                    else:
                        print(line, end='')

# Directory path
dir_path = 'logs/LongForecasting/M_seq_len_336'

# Process the files
process_files(dir_path)

print("Processing complete.")