from src.models.FITS import FITS
from src.models.baseline import NaiveForecast
from src.train_test import train
from src.dataset import data_setup
from src.parser import parser
import warnings
from rich import print

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.dominance_freq == 0:
        args.dominance_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10
        print("Dominance frequency:", args.dominance_freq)

    if args.use_original_datahandling:
        # only works on linux for some reason
        from src.FITS_datahandling.data_factory import data_provider

        train_data, train_loader = data_provider(args, "train")
        test_data, test_loader = data_provider(args, "test")
    else:
        train_loader, test_loader = data_setup(args)

    model = FITS(args)
    # model = NaiveForecast(args)
    print(model)

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
