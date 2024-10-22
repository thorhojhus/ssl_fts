import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from io import BytesIO
import cv2
from utils.metrics import MSE, MAE, SE, RMSE


def configure(dataset, seq_len, pred_len, root_folder, model_list):
    # Get a blue colormap
    cmap = plt.get_cmap('gnuplot')
    
    # Generate colors from the colormap
    colors = [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0.3, 0.9, len(model_list))]
    
    # Create the models dictionary
    models = {'Repeat': {'color': 'gray', 'style': '--'}}
    models.update({model: {'color': color, 'style': '-'} for model, color in zip(model_list, colors)})
    
    exclude_repeat = False
    return root_folder, seq_len, models, exclude_repeat

def load_data(root_folder, model_name):
    pd_path = os.path.join(root_folder, f'{model_name}_pred.npy')
    
    # List of possible metric file names
    metric_file_names = [
        f'{model_name}_metrics.npy',
        f'{"_".join(model_name.split("_")[::-1])}_metrics.npy',  # Reverse order of underscores
        f'{"_".join(model_name.split("_"))}_metrics.npy'  # Original order
    ]
    
    metrics_path = None
    for file_name in metric_file_names:
        path = os.path.join(root_folder, file_name)
        if os.path.exists(path):
            metrics_path = path
            break
    
    if os.path.exists(pd_path):
        pd = np.load(pd_path)
        metrics = np.load(metrics_path, allow_pickle=True).item() if metrics_path else None
        # print(f"Loaded data for {model_name}. Shape: pd {pd.shape} from {pd_path}")
        # if metrics_path:
        #     print(f"Loaded metrics from {metrics_path}")
        # else:
        #     print(f"No metrics file found for {model_name}")
        return pd, metrics
    else:
        print(f"Data not found for {model_name} in {root_folder}")
        return None, None


def load_naive_pred(root_folder):
    gt_path = os.path.join(root_folder, 'gt.npy')
    naive_pred_path = os.path.join(root_folder, 'naive_pred.npy')
    
    if os.path.exists(gt_path) and os.path.exists(naive_pred_path):
        gt = np.load(gt_path)
        naive_pred = np.load(naive_pred_path)
        print(f"Loaded data for Ground Truth. Shape: {gt.shape} from {gt_path}")
        print(f"Loaded data for Repeat. Shape: {naive_pred.shape} from {naive_pred_path}")
        return gt, naive_pred
    else:
        print(f"Ground truth or naive prediction file not found in {root_folder}")
        return None, None


def calculate_metrics(pred, true):
    # Only compare the last pred_len time steps
    pred_len = pred.shape[1]
    true = true[:, -pred_len:, :]
    
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    se = np.mean((pred[:, -1, :] - true[:, -1, :]) ** 2)
    rmse = RMSE(pred, true)
    rrmse = rmse / np.mean(np.abs(true))
    rmae = mae / np.mean(np.abs(true))
    return {
        'mse': mse,
        'mae': mae,
        'se': se,
        # 'RMSE': rmse,
        'relative_rmse': rrmse,
        # 'RMAE': rmae
    }


def assert_metrics(calculated_metrics, loaded_metrics, model_name):
    model_metrics = loaded_metrics.get(model_name, {})
    for key in calculated_metrics:
        loaded_key = key.lower()
        if loaded_key in model_metrics:
            np.testing.assert_almost_equal(calculated_metrics[key], model_metrics[loaded_key], decimal=4, 
                                           err_msg=f"Mismatch in {key} for {model_name}")
        else:
            print(f"Warning: {key} not found in loaded metrics for {model_name}")

    print(f"All available metrics for {model_name} verified successfully.")


def verify_metrics(dataset, seq_len, pred_len):
    root_folder, _, models, _ = configure(dataset, seq_len, pred_len)
    gt, _ = load_naive_pred(root_folder)

    for model in models.keys():
        if model != 'Repeat':
            pd, loaded_metrics = load_data(root_folder, model)
            if gt is not None and pd is not None:
                print("")
                # print(f"\nVerifying metrics for {model}")
                # print(f"Ground truth shape: {gt.shape}")
                # print(f"Prediction shape: {pd.shape}")
                
                if loaded_metrics:
                    model_metrics = loaded_metrics.get(model, {})
                    filtered_metrics = {k: v for k, v in model_metrics.items() if not (k.endswith('_torch') or k.startswith('relative_mae'))}
                    print(f"Loaded metrics:     {model}: {filtered_metrics}")
                
                calculated_metrics = calculate_metrics(pd, gt)
                print(f"Calculated metrics: {model}: {calculated_metrics}")
                
                if loaded_metrics:
                    try:
                        assert_metrics(calculated_metrics, loaded_metrics, model)
                    except AssertionError as e:
                        print(f"Metric verification failed for {model}: {str(e)}")
                else:
                    print(f"No loaded metrics to verify for {model}")
            else:
                print(f"Skipping metric verification for {model} due to missing data.")


def setup_video(root_folder, specified_model, total_samples, dim_to_plot):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(root_folder, f'forecast_video_{specified_model}_{total_samples}_samples_dim_{dim_to_plot+1}.mp4')
    return fourcc, video_path, None


def plot_and_save_frame(gt_data, results, models, input_len, sample, total_samples, specified_model, exclude_repeat, dim_to_plot, dataset):
    plt.figure(figsize=(12, 6), dpi=100)
    
    # Plot ground truth with gradient fill
    x_gt = range(len(gt_data))
    y_gt = gt_data[:, dim_to_plot]
    
    # Create gradient fill
    cmap = plt.get_cmap('YlGnBu_r')
    y_min = min(np.min(gt_data[:, dim_to_plot]), min(np.min(data['pd_data'][:, dim_to_plot]) for data in results.values()))
    y_max = max(np.max(gt_data[:, dim_to_plot]), max(np.max(data['pd_data'][:, dim_to_plot]) for data in results.values()))
    y_range = y_max - y_min
    y_min -= 0.1 * y_range  # Extend 10% below
    y_max += 0.1 * y_range  # Extend 10% above
    
    # Create gradient image
    xv, yv = np.meshgrid(np.linspace(0, len(x_gt)-1, 100), np.linspace(y_min, y_max, 100))
    zv = yv
    plt.imshow(zv, cmap=cmap, origin='upper', aspect='auto',
               extent=[0, len(x_gt)-1, y_min, y_max], alpha=0.1)

    # Plot ground truth line
    plt.plot(x_gt, y_gt, label=f'Ground Truth', linewidth=1.8, color='black', zorder=5)
    
    # Fill above the ground truth line with semi-transparent white
    plt.fill_between(x_gt, y_gt, y_max, color='white', alpha=0.7, zorder=5)

    # Plot ground truth line with higher zorder
    plt.plot(x_gt, y_gt, label=f'Ground Truth', linewidth=1.8, color='black', zorder=6)

    # Plot model predictions
    for model, data in results.items():
        if model != 'Repeat' or not exclude_repeat:
            label = f'{model} [MSE: {data["mse"]:.3f}]'
            
            forecast_data = data["pd_data"][:, dim_to_plot]
            x_forecast = range(input_len, input_len + len(forecast_data))
            
            # Plot the main line
            plt.plot(x_forecast, forecast_data, label=label, linewidth=1.2, 
                     color=models[model]['color'], linestyle=models[model]['style'], zorder=7)
            
            # Add end point marker and annotation
            plt.scatter(x_forecast[-1], forecast_data[-1], color=models[model]['color'], s=100, zorder=8)
            plt.annotate(f'SE: {data["se"]:.3f}', (x_forecast[-1], forecast_data[-1]), 
                         xytext=(5, 5), textcoords='offset points', color=models[model]['color'], zorder=9)

    # Plot forecast start line
    plt.axvline(x=input_len, color='silver', linestyle='--', label='Forecast Start', zorder=4)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    sorted_models = sorted([(k, v) for k, v in results.items() if k != 'Repeat' or not exclude_repeat], key=lambda x: x[1]['mse'])
    
    if len(sorted_models) >= 2:
        best_model, second_best_model = sorted_models[0][0], sorted_models[1][0]
        compared_model = second_best_model if specified_model == best_model else best_model
        mse_diff = results[compared_model]['mse'] - results[specified_model]['mse']
        title = f'{specified_model} {"better" if mse_diff > 0 else "worse"} than {compared_model} by {"-" if mse_diff < 0 else "+"}{abs(mse_diff):.3f} MSE. Sample {sample+1}/{total_samples} on {dataset}. (Dim: {dim_to_plot+1})'
    else:
        title = f'{specified_model} MSE: {results[specified_model]["mse"]:.3f}. Sample {sample+1}/{total_samples}, Dim {dim_to_plot+1}'

    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.ylim(y_min, y_max)  # Set y-axis limits
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


def plot_single_frame(dataset, seq_len, pred_len, specified_model, sample_index, dim_to_plot, root_folder, model_list):
    dim_to_plot = dim_to_plot - 1
    sample_index -= 1
    root_folder, input_len, models, exclude_repeat = configure(dataset, seq_len, pred_len, root_folder, model_list)
    
    gt, naive_pred = load_naive_pred(root_folder)
    
    data = {}
    for model in models.keys():
        if model != 'Repeat':
            pd, metrics = load_data(root_folder, model)
            if gt is not None and pd is not None:
                data[model] = (pd, metrics)
    
    if not data:
        print(f"No data found for any model in {root_folder}")
        return

    total_samples = gt.shape[0]
    if sample_index >= total_samples:
        print(f"Sample index {sample_index+1} is out of range. Total samples: {total_samples}")
        return
    else:
        print(f"Plotting index: {sample_index+1}/{total_samples}")

    gt_data = gt[sample_index]
    results = {}
    
    for model, model_data in data.items():
        pd, _ = model_data
        if pd is not None:
            pd_data = pd[sample_index]
            gt_slice = gt_data[-pred_len:, dim_to_plot]
            pd_slice = pd_data[:, dim_to_plot]
            mse = np.mean((gt_slice - pd_slice)**2)
            se = np.mean((gt_data[-1, dim_to_plot] - pd_data[-1, dim_to_plot])**2)
            results[model] = {'mse': mse, 'se': se, 'pd_data': pd_data}

    if naive_pred is not None:
        naive_pred_data = naive_pred[sample_index]
        gt_slice = gt_data[-pred_len:, dim_to_plot]
        naive_slice = naive_pred_data[:, dim_to_plot]
        mse_naive = np.mean((gt_slice - naive_slice)**2)
        se_naive = np.mean((gt_data[-1, dim_to_plot] - naive_pred_data[-1, dim_to_plot])**2)
        results['Repeat'] = {'mse': mse_naive, 'se': se_naive, 'pd_data': naive_pred_data}

    plt.figure(figsize=(12, 6), dpi=100)
    
    # Plot ground truth with gradient fill
    x_gt = range(len(gt_data))
    y_gt = gt_data[:, dim_to_plot]
    
    # Create gradient fill
    cmap = plt.get_cmap('YlGnBu_r')
    y_min = min(np.min(gt_data[:, dim_to_plot]), min(np.min(data['pd_data'][:, dim_to_plot]) for data in results.values()))
    y_max = max(np.max(gt_data[:, dim_to_plot]), max(np.max(data['pd_data'][:, dim_to_plot]) for data in results.values()))
    y_range = y_max - y_min
    y_min -= 0.1 * y_range  # Extend 10% below
    y_max += 0.1 * y_range  # Extend 10% above
    
    # Create gradient image
    xv, yv = np.meshgrid(np.linspace(0, len(x_gt)-1, 100), np.linspace(y_min, y_max, 100))
    zv = yv
    plt.imshow(zv, cmap=cmap, origin='upper', aspect='auto',
               extent=[0, len(x_gt)-1, y_min, y_max], alpha=0.1)

    # Fill above the ground truth line with semi-transparent white
    plt.fill_between(x_gt, y_gt, y_max, color='white', alpha=0.7, zorder=5)

    # Plot ground truth line with higher zorder
    plt.plot(x_gt, y_gt, label=f'Ground Truth', linewidth=1.8, color='black', zorder=6)

    # Plot model predictions
    for model, data in results.items():
        if model != 'Repeat' or not exclude_repeat:
            label = f'{model} [MSE: {data["mse"]:.3f}]'
            
            forecast_data = data["pd_data"][:, dim_to_plot]
            x_forecast = range(input_len, input_len + len(forecast_data))
            
            # Plot the main line
            plt.plot(x_forecast, forecast_data, label=label, linewidth=1.5, 
                     color=models[model]['color'], linestyle=models[model]['style'], zorder=7, alpha=0.5 if 'DLinear' or 'FITS' in model else 1.0)
            
            # Add end point marker and annotation
            plt.scatter(x_forecast[-1], forecast_data[-1], color=models[model]['color'], s=100, zorder=8)
            plt.annotate(f'SE: {data["se"]:.3f}', (x_forecast[-1], forecast_data[-1]), 
                         xytext=(5, 5), textcoords='offset points', color=models[model]['color'], zorder=9)

    # Plot forecast start line
    plt.axvline(x=input_len, color='silver', linestyle='--', label='Forecast Start', zorder=4)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    sorted_models = sorted([(k, v) for k, v in results.items() if k != 'Repeat' or not exclude_repeat], key=lambda x: x[1]['mse'])
    
    if len(sorted_models) >= 2:
        best_model, second_best_model = sorted_models[0][0], sorted_models[1][0]
        compared_model = second_best_model if specified_model == best_model else best_model
        mse_diff = results[compared_model]['mse'] - results[specified_model]['mse']
        title = f'{specified_model} {"better" if mse_diff > 0 else "worse"} than {compared_model} by {"-" if mse_diff < 0 else "+"}{abs(mse_diff):.3f} MSE. Sample {sample_index+1}/{total_samples} on \${dataset}. (Dim: {dim_to_plot+1})'
    else:
        title = f'{specified_model} MSE: {results[specified_model]["mse"]:.3f}. Sample {sample_index+1}/{total_samples}, Dim {dim_to_plot+1}'

    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.ylim(y_min, y_max)  # Set y-axis limits
    plt.tight_layout()

    plt.show()

def process_data(dataset, seq_len, pred_len, specified_model, dim_to_plot, sample_interval, root_folder, model_list):
    dim_to_plot = dim_to_plot - 1
    root_folder, input_len, models, exclude_repeat = configure(dataset, seq_len, pred_len, root_folder, model_list)
    
    gt, naive_pred = load_naive_pred(root_folder)
    
    data = {}
    for model in models.keys():
        if model != 'Repeat':
            pd, metrics = load_data(root_folder, model)
            if gt is not None and pd is not None:
                data[model] = (pd, metrics)
    
    if not data:
        print(f"No data found for any model in {root_folder}")
        return

    total_samples = gt.shape[0]
    print(f"Total samples: {total_samples}")
    fourcc, video_path, video = setup_video(root_folder, specified_model, total_samples, dim_to_plot)

    for sample in range(0, total_samples, sample_interval):
        gt_data = gt[sample]
        results = {}
        
        for model, model_data in data.items():
            pd, _ = model_data
            if pd is not None:
                pd_data = pd[sample]
                gt_slice = gt_data[-pred_len:, dim_to_plot]
                pd_slice = pd_data[:, dim_to_plot]
                mse = np.mean((gt_slice - pd_slice)**2)
                se = np.mean((gt_data[-1, dim_to_plot] - pd_data[-1, dim_to_plot])**2)
                results[model] = {'mse': mse, 'se': se, 'pd_data': pd_data}

        if naive_pred is not None:
            naive_pred_data = naive_pred[sample]
            gt_slice = gt_data[-pred_len:, dim_to_plot]
            naive_slice = naive_pred_data[:, dim_to_plot]
            mse_naive = np.mean((gt_slice - naive_slice)**2)
            se_naive = np.mean((gt_data[-1, dim_to_plot] - naive_pred_data[-1, dim_to_plot])**2)
            results['Repeat'] = {'mse': mse_naive, 'se': se_naive, 'pd_data': naive_pred_data}

        frame = plot_and_save_frame(gt_data, results, models, input_len, sample, total_samples, specified_model, exclude_repeat, dim_to_plot, dataset)

        if video is None:
            h, w = frame.shape[:2]
            video = cv2.VideoWriter(video_path, fourcc, 2, (w, h))
        
        video.write(frame)
        print(f"Processed sample {sample+1}/{total_samples+1}")

    if video is not None:
        video.release()
    print(f"Video saved to {video_path}")

# Usage
dataset = 'AAPL'
seq_len = 336
pred_len = 192
specified_model = 'Straight_Line_FITS'
dim_to_plot = 6                     # 1-based indexing
sample_interval = 50                # Process every 50th sample
root_folder = f'results/{dataset}_{seq_len}_{pred_len}_M_channels_6'  # Specify the root folder here
model_list = [                      # Specify the models to plot here
    # 'DLinear',
    # 'DLinear_FITS',
    # 'FITS_DLinear',
    'Straight_Line',
    'Intercept',
    'Straight_Line_FITS'
]

sample_index = 550
plot_single_frame(dataset, seq_len, pred_len, specified_model, sample_index, dim_to_plot, root_folder, model_list)

# Uncomment the line below to process data and create a video
# process_data(dataset, seq_len, pred_len, specified_model, dim_to_plot, sample_interval, root_folder, model_list)
# verify_metrics(dataset, seq_len, pred_len)1