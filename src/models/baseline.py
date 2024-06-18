from torch import nn
import numpy as np
import torch


class NaiveForecast(nn.Module):
    """Naïve baseline forecasting (RWF)

    ```python
    Y_[T+h|T] = Y_T, h = 1..H
    ```
    where H = prediction length (pred_len)
    """

    def __init__(self, configs):
        super(NaiveForecast, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x):
        """
        Naïve forecast by returning x, but with the last observed value reapted 'pred_len' times.
        """
        device = x.device
        output = torch.zeros([x.shape[0], x.shape[1] + self.pred_len, x.shape[2]]).to(
            device
        )
        output[:, : x.shape[1], :] = x
        output[:, x.shape[1] :, :] = x[:, -1].unsqueeze(1).repeat(1, self.pred_len, 1)
        return output


class AverageForecast(nn.Module):
    """Average of Time Series as Forecasting value."""

    def __init__(self, configs):
        super(AverageForecast, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x):
        """
        Forecast by returning the average of the time series.
        """
        device = x.device
        output = torch.zeros([x.shape[0], x.shape[1] + self.pred_len, x.shape[2]]).to(
            device
        )
        output[:, : x.shape[1], :] = x
        output[:, x.shape[1] :, :] = (
            x.mean(dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        return output


class LinearRegressionForecast(nn.Module):
    """Perform Univariate Linear Regression to forecast"""

    def __init__(self, configs):
        super(LinearRegressionForecast, self).__init__()
        self.pred_len = configs.pred_len

    def forward(self, x: torch.Tensor):
        """
        Forecast by performing linear regression on the time series.
        """
        device = x.device
        output = torch.zeros([x.shape[0], x.shape[1] + self.pred_len, x.shape[2]]).to(
            device
        )

        for i in range(x.shape[0]):
            for j in range(x.shape[2]):
                x_ = torch.arange(0, x.shape[1]).to(device)
                y_ = x[i, :, j]
                model = np.polyfit(x_, y_, 1)
                y_pred = np.polyval(model, np.arange(0, x.shape[1] + self.pred_len))
                output[i, :, j] = torch.from_numpy(y_pred).to(device)
        return output
