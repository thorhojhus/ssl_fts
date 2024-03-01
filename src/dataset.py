import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from typing import List
import torch


class TimeSeriesDataset(Dataset):
    def __init__(self, data: ndarray, input_length: int = 384, output_length: int = 96):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.data) - self.input_length - self.output_length + 1

    def __getitem__(self, idx):
        start = idx
        end = idx + self.input_length
        x = self.data[start:end]
        y = self.data[end : end + self.output_length]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def load_and_process_data(
    filepath: str,
    target_columns: List[str],
    all_cols: bool = True,
    test_size: float = 0.2,
    normalize: bool = True,
):
    df = pd.read_csv(filepath)

    if all_cols:
        target_columns = df.columns[1:]

    data = df[target_columns].values

    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    train_data, test_data = (
        data[: -int(len(data) * test_size)],
        data[-int(len(data) * test_size) :],
    )

    return train_data, test_data


def create_dataloaders(
    train_data: ndarray,
    test_data: ndarray,
    input_length: int = 384,
    output_length: int = 96,
    batch_size: int = 64,
):
    train_dataset = TimeSeriesDataset(train_data, input_length, output_length)
    test_dataset = TimeSeriesDataset(test_data, input_length, output_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def data_setup(args):
    train_data, test_data = load_and_process_data(
        args.filepath, args.target_columns, args.all_cols, args.test_size, args.normalize
    )
    return create_dataloaders(
        train_data,
        test_data,
        args.input_length,
        args.output_length,
        args.batch_size,
    )
