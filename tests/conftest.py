import argparse
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    torch.manual_seed(123)


@pytest.fixture(scope="function")
def FITS(argparser):
    """Sample FITS model with debug mode."""
    from src.models.FITS import FITS

    args = argparser.parse_args(["--device", "cpu", "--individual", "--debug"])

    model = FITS(args)
    for param in model.parameters():
        param.data.fill_(0)

    return model


@pytest.fixture(scope="function")
def ogFITS(argparser):
    """Sample Original FITS model with debug mode."""
    from src.models.FITS_original import Model

    args = argparser.parse_args(["--device", "cpu", "--individual", "--debug"])

    model = Model(args)
    for param in model.parameters():
        param.data.fill_(0)

    return model


@pytest.fixture(scope="function")
def argparser():
    parser = argparse.ArgumentParser(description="Time Series Forecasting")
    parser.add_argument(
        "--filepath",
        type=str,
        default="data/ETTh1.csv",
        help="Path to the dataset",
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
        default=100,
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
        default=360,
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
        action=argparse.BooleanOptionalAction,
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
        default=106,  # int(seq_len // 24 + 1) * 6 + 10 (int(args.seq_len // args.base_T + 1) * args.H_order + 10)
        help="Dominance frequency",
    )
    return parser
