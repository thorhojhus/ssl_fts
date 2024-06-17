from multiprocessing import Pool, cpu_count
from torch import nn
from statsforecast.models import AutoARIMA as auto_arima

import numpy as np
import torch

def forecast_worker(args):
    seq, pred_len, bt, i = args
    model = auto_arima()
    forecast = model.forecast(seq.numpy(), pred_len)["mean"]
    return forecast, bt, i

class ARIMA(nn.Module):
    """ARIMA model (auto-tuning)"""

    def __init__(self, configs):
        super(ARIMA, self).__init__()
        self.pred_len = configs.pred_len
        self.device = torch.device("cpu")

    def forward(self, x: torch.Tensor):
        """Forward pass of ARIMA model.
        
        Will be EXTREMELY slow, but auto ARIMA is not particularly fast.
        """
        result = np.zeros([x.shape[0], self.pred_len, x.shape[2]])
        tasks = []

        for bt, seqs in enumerate(x):
            for i in range(seqs.shape[-1]):
                seq = seqs[:, i]
                tasks.append((seq, self.pred_len, bt, i))

        with Pool(cpu_count()) as pool:
            results = pool.map(forecast_worker, tasks)
        
        for forecast, bt, i in results:
            result[bt, :, i] = forecast

        return torch.from_numpy(result).to(self.device)


