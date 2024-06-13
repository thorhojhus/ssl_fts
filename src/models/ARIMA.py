from threading import Thread
from torch import nn
from pmdarima.arima import auto_arima
import numpy as np
import torch


class ARIMAThread(Thread):
    """Thread Wrapper for ARIMA."""
    def _forecast(self):
        if self.device == "cuda":
            self.seq = self.seq.cpu()
        else:
            print(self.device)
        model = auto_arima(self.seq)
        forecast = model.predict(n_periods=self.pred_len)
        return forecast, self.bt, self.i

    def __init__(self, args: tuple):
        super(ARIMAThread, self).__init__()
        self.seq = args[0]
        self.pred_len = args[1]
        self.bt = args[2]
        self.i = args[3]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        self.result = self._forecast()

    def join_result(self):
        "Wait for and return result."
        Thread.join(self)
        return self.result


class ARIMA(nn.Module):
    """ARIMA model (auto-tuning)"""

    def __init__(self, configs):
        super(ARIMA, self).__init__()
        self.pred_len = configs.pred_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor):
        """Forward pass of ARIMA model.
        
        Will be EXTREMELY slow, but auto ARIMA is not particularly fast.
        """
        result = np.zeros([x.shape[0],self.pred_len,x.shape[2]])
        arima_threads: list[ARIMAThread] = []
        for bt,seqs in enumerate(x):
            for i in range(seqs.shape[-1]):
                seq = seqs[:,i]
                arima_thread = ARIMAThread(args=(seq,self.pred_len,bt,i))
                arima_threads.append(arima_thread)
                arima_thread.start()
        for every_thread in arima_threads:
            forecast,bt,i = every_thread.join_result()
            result[bt,:,i] = forecast
        return torch.from_numpy(result).to(self.device)
