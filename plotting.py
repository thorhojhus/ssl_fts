import os
import numpy as np
import matplotlib.pyplot as plt

# Load data
root_folder = 'test_results/exchange_rate_336_720_DLinear'
gt = np.load(os.path.join(root_folder, 'gt.npy'))
pd = np.load(os.path.join(root_folder, 'pd.npy'))
naive_pred = np.load(os.path.join(root_folder, 'naive_pred.npy'))

# Try to load DLinear_FITS data
DLinear_FITS_folder = 'test_results/exchange_rate_336_720_DLinear_FITS'
pd_fits = None
if os.path.exists(os.path.join(DLinear_FITS_folder, 'pd.npy')):
    pd_fits = np.load(os.path.join(DLinear_FITS_folder, 'pd.npy'))

print(f"Shapes: gt {gt.shape}, pd {pd.shape}, naive_pred {naive_pred.shape}")
if pd_fits is not None:
    print(f"Shape: pd_fits {pd_fits.shape}")

total_samples = gt.shape[0]
input_len = 336  # Assuming this is still the input length

for sample in range(6, total_samples+6):
    gt_data = gt[sample]
    pd_data = pd[sample]
    naive_pred_data = naive_pred[sample * 32, :, -1]  # Take the last feature for the first instance of each sample

    # Calculate MSE and SE
    mse = np.mean((gt_data[input_len:] - pd_data[input_len:])**2)
    mse_naive = np.mean((gt_data[input_len:] - naive_pred_data)**2)
    se = (gt_data[-1] - pd_data[-1])**2
    se_naive = (gt_data[-1] - naive_pred_data[-1])**2

    if pd_fits is not None:
        pd_fits_data = pd_fits[sample]
        mse_fits = np.mean((gt_data[input_len:] - pd_fits_data[input_len:])**2)
        se_fits = (gt_data[-1] - pd_fits_data[-1])**2

    best_model = 'DLinear' if mse < mse_naive else 'Repeat'
    if pd_fits is not None:
        best_model = min(('DLinear', mse), ('Repeat', mse_naive), ('DLinear_FITS', mse_fits), key=lambda x: x[1])[0]
    improvement = mse_naive - mse

    # Plotting code
    plt.figure(figsize=(12, 6))

    # Plot ground truth
    plt.plot(range(len(gt_data)), gt_data, label='GroundTruth', linewidth=2, color='black')

    DLinear_color = 'blue'
    # Plot DLinear prediction
    main_label = f'DLinear [MSE: {mse:.3f}]{"*" if best_model == "DLinear" else ""}'
    plt.plot(range(len(pd_data)), pd_data, label=main_label, linewidth=2, color=DLinear_color, alpha=0.5)
    plt.scatter(len(pd_data) - 1, pd_data[-1], color=DLinear_color, s=100, zorder=5)
    plt.annotate(f'SE: {se:.3f}', (len(pd_data) - 1, pd_data[-1]), xytext=(5, 5), textcoords='offset points', color=DLinear_color)

    # Plot naive prediction
    naive_label = f'Repeat [MSE: {mse_naive:.3f}]{"*" if best_model == "Repeat" else ""}'
    plt.plot(range(input_len, input_len + len(naive_pred_data)), naive_pred_data, label=naive_label, linestyle='--', linewidth=2, color='gray')
    plt.scatter(input_len + len(naive_pred_data) - 1, naive_pred_data[-1], color='gray', s=100, zorder=5)
    plt.annotate(f'SE: {se_naive:.3f}', (input_len + len(naive_pred_data) - 1, naive_pred_data[-1]), xytext=(5, -15), textcoords='offset points', color='gray')

    # Plot DLinear_FITS prediction if available
    DLinear_FITS_color = 'red'
    if pd_fits is not None:
        fits_label = f'DLinear_FITS [MSE: {mse_fits:.3f}]{"*" if best_model == "DLinear_FITS" else ""}'
        plt.plot(range(len(pd_fits_data)), pd_fits_data, label=fits_label, linewidth=2, color=DLinear_FITS_color, alpha=0.5)
        plt.scatter(len(pd_fits_data) - 1, pd_fits_data[-1], color=DLinear_FITS_color, s=100, zorder=5)
        plt.annotate(f'SE: {se_fits:.3f}', (len(pd_fits_data) - 1, pd_fits_data[-1]), xytext=(5, 25), textcoords='offset points', color=DLinear_FITS_color)

    plt.axvline(x=input_len, color='silver', linestyle='--', label='Forecast Start')

    plt.legend(loc='upper left')

    if pd_fits is not None:
        improvement = max(mse_naive - mse, mse_naive - mse_fits)
    if improvement >= 0:
        title = f'Best model (+{improvement:.3f} better than Repeat) Sample {sample+1}/{total_samples}'
    else:
        title = f'Repeat ({-improvement:.3f} better than others) Sample {sample+1}/{total_samples}'
    
    plt.title(title)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()

    # Break after the first plot to allow for quick iteration
    break