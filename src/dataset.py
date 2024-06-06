import pandas as pd
import numpy as np
import os
from numpy import ndarray
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch
from .augmentations import augmentation
from .augmentations import DatasetAugmentation


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        x_data: NDArray,
        y_data: NDArray,
    ):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


def load_and_process_data(
    root_path: str = "data",
    dataset: str = "ETTh1",
    target_columns: List[str] = "OT",
    seq_len: int = 720,
    label_len: int = 48,
    pred_len: int = 96,
    mode_flag: str = "train",
    all_cols: bool = True,
    normalize: bool = True,
    test_time_train: bool = False,
    aug_rate: tuple = (0.5,),
    augment_data: bool = True,
    aug_method: str = "f_mix",
):
    df = pd.read_csv(os.path.join(root_path, str(dataset + ".csv")))
    mode = {"train": 0, "val": 1, "test": 2}[mode_flag]

    if all_cols:
        target_columns = df.columns[1:]
    df = df[target_columns]

    
    data = df.values

    x_data, y_data = split_data(
        data,
        seq_len,
        pred_len,
        label_len,
        mode,
        data_size=1,
        test_time_train=test_time_train,
        dataset=dataset,
        normalize=normalize
    )
    if augment_data and mode_flag == "train":
        x_data, y_data = augment(x_data, y_data, aug_method, aug_rate)
    return x_data, y_data


def split_data(
    data: NDArray,
    seq_len: int,
    pred_len: int,
    label_len: int,
    mode: int,
    data_size: int = 1,
    test_time_train: bool = False,
    dataset: str = "ETTh1",
    normalize: bool = True,
):
    if dataset == "ETTh1" or dataset == "ETTh2":
        border1s = [0, 8640 - seq_len, 11520 - seq_len]
        border2s = [8640, 11520, 14400]
        if test_time_train:
            border1s = [0, 12960 - seq_len, 14400]
            border2s = [12960, 14400, 14400]
    if dataset == "ETTm1" or dataset == "ETTm2":
        border1s = [0, 34560 - seq_len, 46080 - seq_len]
        border2s = [34560, 46080, 57600]
    else:
        n_train = int(len(data) * 0.7)
        n_test = int(len(data) * 0.2)
        n_val = len(data) - n_train - n_test
        border1s = [0, n_train - seq_len, len(data) - n_test - seq_len]
        border2s = [n_train, n_train + n_val, len(data)]
        if test_time_train:
            n_train = int(len(data) * 0.9)
            border1s = [0, n_train - seq_len, len(data)]
            border2s = [n_train, len(data), len(data)]
    
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    border1, border2 = border1s[mode], border2s[mode]
    data = data[border1:border2]
    data_len = len(data) - seq_len - pred_len + 1
    mask_data_len = int((1 - data_size) * data_len) if data_size < 1 else 0
    x_data, y_data = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        if (mode == 0 and i >= mask_data_len) or mode != 0:
            s_begin = i
            s_end = s_begin + seq_len
            r_begin = s_end - label_len
            r_end = r_begin + label_len + pred_len
            x_data.append(data[s_begin:s_end])
            y_data.append(data[r_begin:r_end])
    return x_data, y_data


# def augment(x_data: list, y_data: list, aug_method: str, aug_rate: tuple):
#     aug_size = [1 for i in range(len(x_data))]
#     for i in range(len(x_data)):
#         for _ in range(aug_size[i]):
#             aug = augmentation("dataset")
#             if aug_method == "f_mask":
#                 x, y = aug.freq_dropout(x_data[i], y_data[i], dropout_rate=aug_rate[0])
#             elif aug_method == "f_mix":
#                 rand = float(np.random.random(1))
#                 i2 = int(rand * len(x_data))
#                 x, y = aug.freq_mix(
#                     x_data[i],
#                     y_data[i],
#                     x_data[i2],
#                     y_data[i2],
#                     dropout_rate=aug_rate[0],  # Pass the first element of aug_rate
#                 )
#             x_data.append(x)
#             y_data.append(y)
#     return x_data, y_data



def augment(x_data: list, y_data: list, aug_method: str, aug_rate: tuple):
    aug_size = [1 for i in range(len(x_data))]

    aug = DatasetAugmentation()

    for i in range(len(x_data)):
        for _ in range(aug_size[i]):
            if aug_method == "f_mask":
                x, y = aug.freq_dropout(x_data[i], y_data[i], dropout_rate=aug_rate[0])
            elif aug_method == "f_mix":
                rand = float(np.random.random(1))
                i2 = int(rand * len(x_data))
                x, y = aug.freq_mix(
                    x_data[i],
                    y_data[i],
                    x_data[i2],
                    y_data[i2],
                    dropout_rate=aug_rate[0]
                )

            x_data.append(x.numpy())
            y_data.append(y.numpy())

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
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_dataloader, test_dataloader


def data_setup(args):
    X_train, y_train = load_and_process_data(
        root_path=args.root_path,
        dataset=args.dataset,
        target_columns=args.target_columns,
        all_cols=args.all_cols,
        normalize=args.normalize,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        mode_flag="train",
        augment_data=args.augment_data,
        aug_method=args.aug_method,
    )

    X_test, y_test = load_and_process_data(
        root_path=args.root_path,
        dataset=args.dataset,
        target_columns=args.target_columns,
        all_cols=args.all_cols,
        normalize=args.normalize,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        augment_data=False,
        mode_flag="test",
    )

    return create_dataloaders(X_train, y_train, X_test, y_test, args.batch_size)
