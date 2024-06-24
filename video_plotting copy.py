import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import cv2

# Configuration
root_folder = 'test_results/MRO_336_720_DLinear'
input_len = 336
specified_model = 'FITS'  # Change this to the model you want to focus on
models = {
    'DLinear': {'color': 'blue', 'style': '-'},
    'DLinear_FITS': {'color': 'magenta', 'style': '-'},
    'FITS': {'color': 'red', 'style': '-'},
    'Repeat': {'color': 'gray', 'style': '--'},
    'FITS_DLinear': {'color': 'orange', 'style': '-'},
}

exclude_repeat = True  # Set this to False if you want to include 'Repeat' in the comparison

# Load data
def load_data(model_name):
    folder = root_folder if model_name == 'DLinear' else f'test_results/MRO_336_720_{model_name}'
    gt_path = os.path.join(folder, 'gt.npy')
    pd_path = os.path.join(folder, 'pd.npy')
    
    if os.path.exists(pd_path):
        gt = np.load(gt_path)
        pd = np.load(pd_path)
        return gt, pd
    return None, None

# Load naive_pred if it exists
naive_pred_path = os.path.join(root_folder, 'naive_pred.npy')
naive_pred = None
if os.path.exists(naive_pred_path):
    naive_pred = np.load(naive_pred_path)

# Load data for all models
data = {model: load_data(model) for model in models.keys() if model != 'Repeat'}
gt = data['DLinear'][0]  # Use DLinear's ground truth for all

# Video setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = os.path.join(root_folder, f'forecast_video_{specified_model}.mp4')
video = None

total_samples = gt.shape[0]

for sample in range(total_samples):
    plt.figure(figsize=(12, 6), dpi=100)
    
    gt_data = gt[sample]
    
    # Plot ground truth
    plt.plot(range(len(gt_data)), gt_data, label='Ground Truth', linewidth=2, color='black')
    
    results = {}
    for model, (_, pd) in data.items():
        if pd is not None:
            pd_data = pd[sample]
            mse = np.mean((gt_data[input_len:] - pd_data[input_len:])**2)
            se = (gt_data[-1] - pd_data[-1])**2
            results[model] = {'mse': mse, 'se': se, 'pd_data': pd_data}
            
            label = f'{model} [MSE: {mse:.3f}]'
            plt.plot(range(len(pd_data)), pd_data, label=label, linewidth=2, 
                     color=models[model]['color'], linestyle=models[model]['style'], alpha=0.6)
            plt.scatter(len(pd_data) - 1, pd_data[-1], color=models[model]['color'], s=100, zorder=5)
            plt.annotate(f'SE: {se:.3f}', (len(pd_data) - 1, pd_data[-1]), 
                         xytext=(5, 5), textcoords='offset points', color=models[model]['color'])

    # Plot naive prediction if available
    if naive_pred is not None:
        naive_pred_data = naive_pred[sample * 32, :, -1]
        mse_naive = np.mean((gt_data[input_len:] - naive_pred_data)**2)
        se_naive = (gt_data[-1] - naive_pred_data[-1])**2
        results['Repeat'] = {'mse': mse_naive, 'se': se_naive, 'pd_data': naive_pred_data}
        
        naive_label = f'Repeat [MSE: {mse_naive:.3f}]'
        plt.plot(range(input_len, input_len + len(naive_pred_data)), naive_pred_data, 
                 label=naive_label, linestyle='--', linewidth=2, color='gray')
        plt.scatter(input_len + len(naive_pred_data) - 1, naive_pred_data[-1], color='gray', s=100, zorder=5)
        plt.annotate(f'SE: {se_naive:.3f}', (input_len + len(naive_pred_data) - 1, naive_pred_data[-1]), 
                     xytext=(5, -15), textcoords='offset points', color='gray')

    plt.axvline(x=input_len, color='silver', linestyle='--', label='Forecast Start')

    plt.legend(loc='upper left')

    # Find the best and second-best models
    if exclude_repeat:
        sorted_models = sorted([(k, v) for k, v in results.items() if k != 'Repeat'], key=lambda x: x[1]['mse'])
    else:
        sorted_models = sorted(results.items(), key=lambda x: x[1]['mse'])
    
    if len(sorted_models) >= 2:
        best_model, second_best_model = sorted_models[0][0], sorted_models[1][0]

        # Compare specified model to the best or second-best model
        if specified_model == best_model:
            compared_model = second_best_model
        else:
            compared_model = best_model

        mse_diff = results[compared_model]['mse'] - results[specified_model]['mse']
        
        if mse_diff > 0:
            title = f'{specified_model} better than {compared_model} by +{mse_diff:.3f} MSE. Sample {sample+1}/{total_samples}'
        else:
            title = f'{specified_model} worse than {compared_model} by {-mse_diff:.3f} MSE. Sample {sample+1}/{total_samples}'
    else:
        # In case there's only one model (or only 'Repeat' if it's not excluded)
        title = f'{specified_model} MSE: {results[specified_model]["mse"]:.3f}. Sample {sample+1}/{total_samples}'

    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.tight_layout()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # Convert BytesIO to numpy array and decode to an image
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    
    # Initialize video writer with the first frame's shape
    if video is None:
        h, w = frame.shape[:2]
        video = cv2.VideoWriter(video_path, fourcc, 2, (w, h))
    
    # Add to video
    video.write(frame)

    print(f"Processed sample {sample+1}/{total_samples}")

# After the loop, release the video writer
if video is not None:
    video.release()

print(f"Video saved to {video_path}")