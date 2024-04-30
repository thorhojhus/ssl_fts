import argparse
from src.models.FITS import FITS
from src.train import train
from src.dataset import data_setup
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Time Series Forecasting")


# Data params
# fmt: off
parser.add_argument(
    "--root_path",
    type=str,
    default="data",
    help="Root path",
)

parser.add_argument(
    "--dataset",
    type=str,
    default="ETTh1",
    help="Dataset",
)

parser.add_argument(
    "--debug",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="debug mode",
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


parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size",
)

parser.add_argument(
    "--test_size",
    type=float,
    default=0.1,
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
    default=360,
    help="Length of the input sequence",
)

parser.add_argument(
    "--label_len",
    type=int,
    default=48,
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
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Individual frequency upsampling",
)

parser.add_argument(
    "--channels",
    type=int,
    default=7,
    help="Number of channels",
)

parser.add_argument(
    "--dominance_freq",
    type=int,
    default=106,
    help="Dominance frequency",
)

parser.add_argument(
    "--augment_data",
    type=bool,
    default=False,
    help="Whether to use data augmentation"
)

parser.add_argument(
    "--aug_method",
    type=str,
    default="f_mix",
    help="Augmentations",
)

parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="Features to use (M or MS)",
)
parser.add_argument(
    "--use_wandb",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use wandb",
)

parser.add_argument(
    "--ft",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Fine-tune",
)

parser.add_argument(
    "--train_and_finetune",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Train and fine-tune",
)

args = parser.parse_args()

train_loader, test_loader = data_setup(args)
model = FITS(args)
print(model)

for param in model.parameters():
    param.data.fill_(0)

if args.train_and_finetune:
    model = train(model=model, train_loader=train_loader, test_loader=test_loader, epochs=args.epochs, pred_len=args.pred_len, features=args.features, ft=0, args=args)
    model = train(model=model, train_loader=train_loader, test_loader=test_loader, epochs=args.epochs, pred_len=args.pred_len, features=args.features, ft=1, args=args)

else:
    model = train(model=model, train_loader=train_loader, test_loader=test_loader, epochs=args.epochs, pred_len=args.pred_len, features=args.features, ft=args.ft, args=args)

