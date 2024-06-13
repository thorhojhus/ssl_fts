from src.models.FITS import FITS
from src.models.baseline import NaiveForecast
from src.models.ARIMA import ARIMA
from src.train_test import train
from src.train_test import test
from src.dataset import data_setup
from src.parser import parser
import warnings
from rich import print
import wandb
import datetime
from torchinfo import summary

warnings.filterwarnings("ignore")

model_dict = {
    "FITS": FITS,
    "ARIMA": ARIMA,
    "NF": NaiveForecast,
}

if __name__ == "__main__":
    args = parser.parse_args()

    wandb.init(
        project="ssl_fts",
        entity="ssl_fts",
        name=f'{args.dataset}_feat_{args.features}_pred_{args.pred_len}_label_{args.label_len}_seq_{args.seq_len}_model_{args.model}-time_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
        mode="online" if args.use_wandb else "disabled",
        config=args
    )


    if args.dominance_freq == 0:
        args.dominance_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10
        print("Dominance frequency:", args.dominance_freq)

    if args.use_original_datahandling:
        # only works on linux for some reason
        from src.FITS_datahandling.data_factory import data_provider

        train_data, train_loader = data_provider(args, "train")
        val_data, val_loader = data_provider(args, "val")
        test_data, test_loader = data_provider(args, "test")
    else:
        train_loader, test_loader = data_setup(args)
        val_loader = test_loader
    
    
    model = model_dict[args.model](args)
    if args.use_real_FITS:
        from src.models.real_deep_FITS import FITS
        model = FITS(args)
    
    summary(model)

    if args.train_and_finetune and (not args.test_only):
        model, _ = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
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
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            pred_len=args.pred_len,
            features=args.features,
            ft=1,
            args=args,
        )

    elif not args.test_only:
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
    else:
        model, test_mse = test(
            model=model,
            test_loader=test_loader,
            pred_len=args.pred_len,
            f_dim=-1 if args.features == "MS" else 0,
            ft=args.ft,
        )
