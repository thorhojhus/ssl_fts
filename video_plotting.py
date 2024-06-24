import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import cv2

def configure(dataset, seq_len, pred_len, model):
    root_folder = f'test_results/{dataset}_{seq_len}_{pred_len}_{model}'
    models = {
        'DLinear': {'color': 'blue', 'style': '-'},
        'DLinear_FITS': {'color': 'magenta', 'style': '-'},
        'FITS': {'color': 'red', 'style': '-'},
        'Repeat': {'color': 'gray', 'style': '--'},
        'FITS_DLinear': {'color': 'orange', 'style': '-'},
    }
    exclude_repeat = True  # Set this to False if you want to include 'Repeat' in the comparison
    return root_folder, seq_len, models, exclude_repeat

def load_data(dataset, seq_len, pred_len, model_name):
    folder = f'test_results/{dataset}_{seq_len}_{pred_len}_{model_name}'
    gt_path = os.path.join(folder, 'gt.npy')
    pd_path = os.path.join(folder, 'pd.npy')
    
    if os.path.exists(pd_path):
        return np.load(gt_path), np.load(pd_path)
    return None, None

def load_naive_pred(root_folder):
    naive_pred_path = os.path.join(root_folder, 'naive_pred.npy')
    return np.load(naive_pred_path) if os.path.exists(naive_pred_path) else None

def setup_video(root_folder, specified_model):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(root_folder, f'forecast_video_{specified_model}.mp4')
    return fourcc, video_path, None

def plot_and_save_frame(gt_data, results, models, input_len, sample, total_samples, specified_model, exclude_repeat):
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(range(len(gt_data)), gt_data, label='Ground Truth', linewidth=2, color='black')

    for model, data in results.items():
        if model != 'Repeat' or not exclude_repeat:
            label = f'{model} [MSE: {data["mse"]:.3f}]'
            plt.plot(range(len(data["pd_data"])), data["pd_data"], label=label, linewidth=2, 
                     color=models[model]['color'], linestyle=models[model]['style'], alpha=0.6)
            plt.scatter(len(data["pd_data"]) - 1, data["pd_data"][-1], color=models[model]['color'], s=100, zorder=5)
            plt.annotate(f'SE: {data["se"]:.3f}', (len(data["pd_data"]) - 1, data["pd_data"][-1]), 
                         xytext=(5, 5), textcoords='offset points', color=models[model]['color'])

    plt.axvline(x=input_len, color='silver', linestyle='--', label='Forecast Start')
    plt.legend(loc='upper left')

    sorted_models = sorted([(k, v) for k, v in results.items() if k != 'Repeat' or not exclude_repeat], key=lambda x: x[1]['mse'])
    
    if len(sorted_models) >= 2:
        best_model, second_best_model = sorted_models[0][0], sorted_models[1][0]
        compared_model = second_best_model if specified_model == best_model else best_model
        mse_diff = results[compared_model]['mse'] - results[specified_model]['mse']
        title = f'{specified_model} {"better" if mse_diff > 0 else "worse"} than {compared_model} by {abs(mse_diff):.3f} MSE. Sample {sample+1}/{total_samples}'
    else:
        title = f'{specified_model} MSE: {results[specified_model]["mse"]:.3f}. Sample {sample+1}/{total_samples}'

    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

def process_data(dataset, seq_len, pred_len, specified_model):
    root_folder, input_len, models, exclude_repeat = configure(dataset, seq_len, pred_len, specified_model)
    naive_pred = load_naive_pred(root_folder)
    
    data = {model: load_data(dataset, seq_len, pred_len, model) for model in models.keys() if model != 'Repeat'}
    gt = data[specified_model][0]  # Use specified model's ground truth

    fourcc, video_path, video = setup_video(root_folder, specified_model)
    total_samples = gt.shape[0]
    print(total_samples)

    # for sample in range(total_samples):
    #     gt_data = gt[sample]
    #     results = {}
        
    #     for model, (_, pd) in data.items():
    #         if pd is not None:
    #             pd_data = pd[sample]
    #             mse = np.mean((gt_data[input_len:] - pd_data[input_len:])**2)
    #             se = (gt_data[-1] - pd_data[-1])**2
    #             results[model] = {'mse': mse, 'se': se, 'pd_data': pd_data}

    #     if naive_pred is not None:
    #         naive_pred_data = naive_pred[sample * 32, :, -1]
    #         mse_naive = np.mean((gt_data[input_len:] - naive_pred_data)**2)
    #         se_naive = (gt_data[-1] - naive_pred_data[-1])**2
    #         results['Repeat'] = {'mse': mse_naive, 'se': se_naive, 'pd_data': naive_pred_data}

    #     frame = plot_and_save_frame(gt_data, results, models, input_len, sample, total_samples, specified_model, exclude_repeat)

    #     if video is None:
    #         h, w = frame.shape[:2]
    #         video = cv2.VideoWriter(video_path, fourcc, 2, (w, h))
        
    #     video.write(frame)
    #     print(f"Processed sample {sample+1}/{total_samples}")

    # if video is not None:
    #     video.release()
    # print(f"Video saved to {video_path}")

# Usage
dataset = 'MRO'
seq_len = 336
pred_len = 720
specified_model = 'FITS'

process_data(dataset, seq_len, pred_len, specified_model)