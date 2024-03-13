import argparse
from src.fits import FITS
from src.train import train
from src.dataset import data_setup
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Time Series Forecasting")

# Data params

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

# Training params
parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    help="Number of epochs",
)

parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size",
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

# Model params
parser.add_argument(
    "--seq_len",
    type=int,
    default=720,
    help="Length of the input sequence",
)

parser.add_argument(
    "--label_len",
    type=int,
    default=96,
    help="Length of the input sequence",
)

parser.add_argument(
    "--pred_len",
    type=int,
    default=96,
    help="Length of the output sequence",
)

parser.add_argument(
    "--individual",
    type=bool,
    default=False,
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
    default=20,  # int(seq_len // 24 + 1) * 6 + 10 (int(args.seq_len // args.base_T + 1) * args.H_order + 10)
    help="Dominance frequency",
)

args = parser.parse_args()

train_loader, test_loader = data_setup(args)
model = FITS(args)

for param in model.parameters():
    param.data.fill_(0)

model = train(model, train_loader, test_loader, args.epochs, args.device, args.pred_len)
