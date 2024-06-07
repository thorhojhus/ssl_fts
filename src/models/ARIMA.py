import pmdarima as pm
from numpy.typing import NDArray
import numpy as np
from torch.utils.data import DataLoader
import torch
from src.dataset import load_and_process_data

import matplotlib.pyplot as plt


class TimeSeriesARIMA:
    """An ARIMA model for time series anaylsis."""

    def __init__(self, p, q, d, args, auto: bool = True):
        self.p, self.q, self.d = p, q, d
        self.args = args

    def fit(self, data_x, auto: bool = False):
        if auto:
            self.model = pm.auto_arima(data_x, trace=True)
        else:
            self.model = pm.ARIMA(order=(self.p, self.d, self.q))
            self.model.fit(data_x)

    def predict(self, np_y: NDArray):
        prediction = self.model.predict_in_sample()
        forecast, conf = self.model.predict(
            n_periods=len(np_y), dynamic=True, return_conf_int=True
        )
        # print(f"Prediction: {prediction}")
        # print(f"Forecast: {forecast}")
        return prediction, forecast

    def train(self, train_data: NDArray, test_data: NDArray):

        def RMAE(output, target):
            return np.sqrt(np.mean(np.abs(output - target)))

        train_loss_rmae = 0
        np_xy: NDArray = np.concatenate([train_data, test_data], axis=0)

        # Fit to batch
        self.fit(train_data, auto=True)

        # Reconstruct and forecast
        model_pred, model_forecast = self.predict(test_data)
        model_output = np.concatenate([model_pred, model_forecast], axis=0)

        # print(np_xy.shape)
        # loss_mse = torch.nn.MSELoss()(model_output, np_xy)
        train_loss_rmae = RMAE(model_output, np_xy).item()

        # Plot the model output versus the actual data

        plt.plot(np.arange(len(np_xy)), np_xy, label="Actual")
        plt.plot(np.arange(len(np_xy)), model_output, label="Model")
        plt.legend()
        plt.show()

        print(f"RMAE: {train_loss_rmae:.4f}")


def generate_test_data():

    x = np.linspace(0, 100, 1000)
    y = np.sin(x) + np.random.normal(0, 0.1, 1000)

    # split
    train_x = np.array(y[:800])
    test_x = np.array(y[800:])

    return train_x, test_x
