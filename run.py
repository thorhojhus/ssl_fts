import argparse
from src.dataset import data_setup

parser = argparse.ArgumentParser(description="Time Series Forecasting")

parser.add_argument(
    "--filepath",
    type=str,
    default="data/ETTh1.csv",
    help="Path to the dataset",
)

parser.add_argument(
    "--target_columns",
    type=str,
    nargs="+",
    default=["OT"],
    help="Columns to use as target",
)

parser.add_argument(
    "--all_cols",
    type=bool,
    default=True,
    help="Use all columns as target",
)

parser.add_argument(
    "--test_size",
    type=float,
    default=0.2,
    help="Size of the test set",
)

parser.add_argument(
    "--normalize",
    type=bool,
    default=True,
    help="Normalize the data",
)

parser.add_argument(
    "--input_length",
    type=int,
    default=384,
    help="Length of the input sequence",
)

parser.add_argument(
    "--output_length",
    type=int,
    default=96,
    help="Length of the output sequence",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size",
)



args = parser.parse_args()

train_loader, test_loader = data_setup(args)

print("Data setup complete!")
print(f"Train data shape: {train_loader.dataset.data.shape}")
print(f"Test data shape: {test_loader.dataset.data.shape}")
print(f"Number of train batches: {len(train_loader)}")
print(f"Number of test batches: {len(test_loader)}")
print(f"x batch shape: {next(iter(train_loader))[0].shape}")
print(f"y batch shape: {next(iter(train_loader))[1].shape}")