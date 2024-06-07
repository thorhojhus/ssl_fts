from src.models.FITS import FITS
from src.models.ARIMA import TimeSeriesARIMA
from src.train import train
from src.dataset import data_setup
from src.dataset import load_and_process_arima_data
from src.parser import parser
import warnings
from rich import print

warnings.filterwarnings("ignore")

args = parser.parse_args()

if args.dominance_freq == 0:
    args.dominance_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10
    print(args.dominance_freq)

if args.use_original_datahandling:
    from src.FITS_datahandling.data_factory import data_provider

    train_data, train_loader = data_provider(args, "train")
    test_data, test_loader = data_provider(args, "test")
else:
    train_loader, test_loader = data_setup(args)

model = FITS(args)

arima_train = False
if args.features == "MS":
    args = parser.parse_args(["--batch_size", "1", "--all_cols", ""])
    arima_model = TimeSeriesARIMA(4, 1, 2, args)
    arima_train = True
    arima_train_data, arima_test_data = load_and_process_arima_data(
        args.root_path,
        args.dataset,
        args.normalize,
    )

for param in model.parameters():
    param.data.fill_(0)

if args.train_and_finetune:
    model, _ = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        pred_len=args.pred_len,
        features=args.features,
        ft=0,
        args=args,
    )
    model, test_mse = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        pred_len=args.pred_len,
        features=args.features,
        ft=1,
        args=args,
    )

else:
    if arima_train:
        arima_model.train(arima_train_data, arima_test_data)
    model, test_mse = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        pred_len=args.pred_len,
        features=args.features,
        ft=args.ft,
        args=args,
    )
