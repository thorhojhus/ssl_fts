from torch import nn
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
        output = torch.zeros([x.shape[0], x.shape[1] + self.pred_len, x.shape[2]]).to(device)
        output[:, : x.shape[1], :] = x
        output[:, x.shape[1] :, :] = x[:, -1].unsqueeze(1).repeat(1, self.pred_len, 1)
        return output
