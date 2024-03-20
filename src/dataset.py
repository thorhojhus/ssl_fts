import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch
from .augmentations import augmentation
from sklearn.model_selection import train_test_split


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        x_data: ndarray,
        y_data: ndarray,
    ):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def load_and_process_data(
    filepath: str,
    target_columns: List[str],
    seq_len: int = 720,
    label_len: int = 48,
    pred_len: int = 96,
    mode_flag: str = "train",
    all_cols: bool = True,
    normalize: bool = True,
    test_time_train: bool = False,
    data_size: int = 1,
    aug_rate: tuple = (0.5,),
    augment_data: bool = False,
    aug_method: str = "f_mix",
):
    df = pd.read_csv(filepath)

    mode = {"train": 0, "val": 1, "test": 2}[mode_flag]

    # if test_time_train:
    #     border1s = [0, 12960 - seq_len, 14400]
    #     border2s = [12960, 14400, 14400]
    # else:
    #     border1s = [0, 8640 - seq_len, 11520 - seq_len]
    #     border2s = [8640, 11520, 14400]

    # border1, border2 = border1s[mode], border2s[mode]

    if all_cols:
        target_columns = df.columns[1:]

    df = df[target_columns]

    if normalize:
        scaler = StandardScaler()
        # data = df[border1s[0] : border2s[0]]
        data = scaler.fit_transform(df.values)
    else:
        data = df.values

        # train_data, test_data = (
        #     data[: -int(len(data) * test_size)],
        #     data[-int(len(data) * test_size) :],
        # )

    # data_x = data[border1:border2]
    # data_y = data[border1:border2]
    data_x = data
    data_y = data
    x_data, y_data = split_data(
        data_x, data_y, seq_len, pred_len, label_len, mode, data_size=1
    )

    if augment_data:
        x_data, y_data = augment(x_data, y_data, aug_method, aug_rate)

    return x_data, y_data


def split_data(
    data_x: ndarray,
    data_y: ndarray,
    seq_len: int,
    pred_len: int,
    label_len: int,
    mode: int,
    data_size: int = 1,
):
    x_data = []
    y_data = []
    data_len = len(data_x) - seq_len - pred_len + 1
    mask_data_len = int((1 - data_size) * data_len) if data_size < 1 else 0
    for i in range(len(data_x) - seq_len - pred_len + 1):
        if (mode == 0 and i >= mask_data_len) or mode != 0:
            s_begin = i
            s_end = s_begin + seq_len
            r_begin = s_end - label_len
            r_end = r_begin + label_len + pred_len
            x_data.append(data_x[s_begin:s_end])
            y_data.append(data_y[r_begin:r_end])
    return x_data, y_data


def augment(x_data: list, y_data: list, aug_method: str, aug_rate: tuple):
    aug_size = [1 for i in len(x_data)]
    for i in range(len(x_data)):
        for _ in range(aug_size[i]):
            aug = augmentation("dataset")
            if aug_method == "f_mask":
                x, y = aug.freq_dropout(x_data[i], y_data[i], dropout_rate=aug_rate)
            elif aug_method == "f_mix":
                rand = float(np.random.random(1))
                i2 = int(rand * len(x_data))
                x, y = aug.freq_mix(
                    x_data[i],
                    y_data[i],
                    x_data[i2],
                    y_data[i2],
                    dropout_rate=aug_rate,
                )
            x_data.append(x)
            y_data.append(y)
    return x_data, y_data


def create_dataloaders(
    x_data_train: list,
    y_data_train: list,
    x_data_test: list,
    y_data_test: list,
    batch_size: int,
):
    train_dataset = TimeSeriesDataset(x_data_train, y_data_train)
    test_dataset = TimeSeriesDataset(x_data_test, y_data_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, test_dataloader


def data_setup(args):
    x_data, y_data = load_and_process_data(
        filepath=args.filepath,
        target_columns=args.target_columns,
        all_cols=args.all_cols,
        normalize=args.normalize,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        mode_flag="train",
    )

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data)

    return create_dataloaders(X_train, y_train, X_test, y_test, args.batch_size)
