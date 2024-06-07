import pmdarima as pm


class TimeSeriesARIMA:
    """An ARIMA model for time series anaylsis."""

    def __init__(self, p, q, d):
        self.p, self.q, self.d = p, q, d

    def fit(self, data_x):
        self.model = pm.ARIMA(order=(self.p, self.d, self.q))
        self.model.fit(data_x)

    def auto_fit(self, data_x):
        self.model = pm.auto_arima(
            data_x, seasonal=False, suppress_warnings=True, trace=True
        )
