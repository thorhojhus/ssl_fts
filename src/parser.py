import argparse

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
    "--num_layers",
    type=int,
    default=1,
    help="Number of layers",
)

parser.add_argument(
    "--num_hidden",
    type=int,
    default=64,
    help="Number of hidden units (only used for layers greater than 1)",
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
    "--upsample_rate",
    type=float,
    default=0,
    help="Upsample rate",
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

parser.add_argument(
    "--H_order",
    type=int,
    default=2,
    help="The harmonic order (multiple of the base period - base_T)",
)

parser.add_argument(
    '--base_T',
    type=int,
    default=24,
    help="The dominant periodicity corresponding to the base frequency of the signal"
)

parser.add_argument(
    "--use_real_FITS",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Real FITS",
)

parser.add_argument(
    '--use_original_datahandling',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Use original data handling')

parser.add_argument(
    '--test_only',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='only do testing (ARIMA & NaiveForecast only)')


parser.add_argument(
    '--model',
    type=str,
    default="FITS",
    help='Select between [FITS, ARIMA, NF (NaiveForecast)]')

parser.add_argument(
    '--save_state_dict',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Save state dict of model')

parser.add_argument(
    "--use_deep",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use deep FITS",
)

# type: ignore
