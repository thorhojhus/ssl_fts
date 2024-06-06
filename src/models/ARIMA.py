import pmdarima as pm
from argparse import Namespace
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset import load_and_process_data


class TimeSeriesARIMA:
    """An ARIMA model for time series anaylsis."""

    def __init__(self, p, q, d, args: Namespace):
        self.p, self.q, self.d = p, q, d
        self.x, self.y = load_and_process_data(
            root_path=args.root_path,
            dataset=args.dataset,
            target_columns=args.target_columns,
            all_cols=False,  # args.all_cols,
            normalize=args.normalize,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            mode_flag="train",
            augment_data=args.augment_data,
            aug_method=args.aug_method,
        )

        self.pred_len = args.pred_len
        self.seq_len = args.seq_len
        self.label_len = args.label_len

    def fit(self, data_x):
        self.model = pm.ARIMA(order=(self.p, self.d, self.q))
        self.model.fit(data_x)

    def auto_fit(self, data_x):
        self.model = pm.auto_arima(
            data_x, seasonal=False, suppress_warnings=True, trace=True
        )


# arima = TimeSeriesARIMA(p=2, q=25, d=1, args=args)
# arima.plot_correlations()
# arima.fit()
