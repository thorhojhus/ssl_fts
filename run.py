import argparse

parser = argparse.ArgumentParser(
    description="Autoformer & Transformer family for Time Series Forecasting"
)

# basic config
parser.add_argument("--is_training", type=int, default=1, help="status")
parser.add_argument("--model_id", type=str, default="test", help="model id")
parser.add_argument(
    "--model",
    type=str,
    default="Autoformer",
    help="model name, options: [Autoformer, Informer, Transformer]",
)

# data loader
parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
parser.add_argument(
    "--root_path", type=str, default="./data/", help="root path of the data file"
)
parser.add_argument("--data_path", type=str, default="ETTm1.csv", help="data file")
parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
)
parser.add_argument(
    "--target", type=str, default="OT", help="target feature in S or MS task"
)
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./checkpoints/",
    help="location of model checkpoints",
)

# forecasting task
parser.add_argument("--seq_len", type=int, default=720, help="input sequence length")
parser.add_argument("--label_len", type=int, default=48, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=96, help="prediction sequence length"
)


parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)

parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)

# parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
# parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
# parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
# parser.add_argument('--c_out', type=int, default=7, help='output size')
# parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
# parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
# parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
# parser.add_argument('--factor', type=int, default=1, help='attn factor')
# parser.add_argument('--distil', action='store_false',
#                     help='whether to use distilling in encoder, using this argument means not using distilling',
#                     default=True)
# parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
# parser.add_argument('--activation', type=str, default='gelu', help='activation')
# parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
# parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# #Film
# parser.add_argument('--ab', type=int, default=2, help='ablation version')
# # SCINet

# parser.add_argument('--hidden_size', default=1, type=float, help='hidden channel of module')
# parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
# parser.add_argument('--groups', type=int, default=1)
# parser.add_argument('--levels', type=int, default=3)
# parser.add_argument('--stacks', type=int, default=1, help='1 stack or 2 stacks')

# optimization
parser.add_argument(
    "--num_workers", type=int, default=10, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=2, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")

# parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
# parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
# parser.add_argument('--des', type=str, default='test', help='exp description')
# parser.add_argument('--loss', type=str, default='mse', help='loss function')
# parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
# parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# # GPU
# parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=0, help='gpu')
# parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
# parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Augmentation
parser.add_argument(
    "--aug_method",
    type=str,
    default="NA",
    help="f_mask: Frequency Masking, f_mix: Frequency Mixing",
)
parser.add_argument("--aug_rate", type=float, default=0.5, help="mask/mix rate")
parser.add_argument(
    "--in_batch_augmentation",
    action="store_true",
    help="Augmentation in Batch (save memory cost)",
    default=False,
)
parser.add_argument(
    "--in_dataset_augmentation",
    action="store_true",
    help="Augmentation in Dataset",
    default=False,
)
parser.add_argument(
    "--data_size",
    type=float,
    default=1,
    help="size of dataset, i.e, 0.01 represents uses 1 persent samples in the dataset",
)
parser.add_argument(
    "--aug_data_size",
    type=int,
    default=1,
    help="size of augmented data, i.e, 1 means double the size of dataset",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="size of augmented data, i.e, 1 means double the size of dataset",
)

# continue learning
parser.add_argument("--testset_div", type=int, default=2, help="Division of dataset")
parser.add_argument(
    "--test_time_train", type=bool, default=False, help="Affect data division"
)

args = parser.parse_args()


from src.dataloader import data_setup

# python3 run.py --is_training=1 --model_id=FITS --model=FITS --seq_len=720 --pred_len=96 --features=M --data=ETTm1 --data_path=ETTm1.csv

dataset, dataloader = data_setup(args, "test")

print(next(iter(dataloader)))
