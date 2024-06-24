import os
import glob

def print_final_metrics():
    if os.path.exists('final_metrics.txt'):
        with open('final_metrics.txt', 'r') as f:
            print(f.read())
        os.remove('final_metrics.txt')  # Optional: remove the temporary file
    else:
        print("\nNo final metrics found.")

print_final_metrics()