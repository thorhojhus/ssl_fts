from src.models.FITS import FITS
from src.train import train
from src.dataset import data_setup
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

model = FITS(args, extra_sum_channel=True)
print(model)

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
