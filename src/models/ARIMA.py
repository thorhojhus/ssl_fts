from statsmodels.tsa.arima_model import ARIMA
from argparse import Namespace


class TimeSeriesARIMA:
    """An ARIMA model for time series anaylsis."""

    def __init__(self, timeseries_data, p, q, d, args: Namespace):
        self.data = timeseries_data
        self.order = (p, q, d)
        print(timeseries_data)

    def fit(self):
        self.model = ARIMA(self.data, order=self.order)
        self.model_fit = self.model.fit(disp=0)
        return self.model_fit

    def predict(self, start, end):
        return self.model_fit.predict(start=start, end=end)
