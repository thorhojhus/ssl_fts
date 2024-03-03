import argparse
from src.fits import FITS
from src.train import train
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
    help="Use all columns as target. If False, specify the target columns with --target_columns",
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

parser.add_argument(
    "--individual",
    type=bool,
    default=True,
    help="Individual frequency upsampling",
)

parser.add_argument(
    "--channels",
    type=int,
    default=1,
    help="Number of channels",
)

parser.add_argument(
    "--dominance_freq",
    type=int,
    default=20,
    help="Dominance frequency",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="Number of epochs",
)


args = parser.parse_args()

train_loader, test_loader = data_setup(args)
model = FITS(args)
for param in model.parameters():
    param.data.fill_(0)


model = train(model, train_loader, args.epochs)