import os
import sys
from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    args.data = args.dataset
    args.target = args.target_columns
    args.data_size = 1
    args.test_time_train = False
    args.in_batch_augmentation = False
    args.in_dataset_augmentation = False
    args.num_workers = 8
    if args.data not in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        Data = data_dict['custom']
    else:
        Data = data_dict[args.data]
    #timeenc = 0 if args.embed != 'timeF' else 1
    timeenc = 1

    if 'ipykernel' in sys.modules:
        # Running in a Jupyter notebook
        args.root_path = '../data'
    else:
        # Running as a Python script
        args.root_path = 'data'
    args.freq = 'h'

    if args.data == 'ETTh1':
        args.data_path = 'ETTh1.csv'
        args.freq = 'h'
    if args.data == 'ETTh2':
        args.data_path = 'ETTh2.csv'
        args.freq = 'h'
    if args.data == 'ETTm1':
        args.data_path = 'ETTm1.csv'
        args.freq = 'm'
    if args.data == 'ETTm2':
        args.data_path = 'ETTm2.csv'
        args.freq = 'm'
    if args.data == "weather":
        args.data_path = 'weather.csv'
    if args.data == "electricity":
        args.data_path = 'electricity.csv'
    if args.data == "traffic":
        args.data_path = "traffic.csv"
    if args.data == "exchange_rate":
        args.data_path = "exchange_rate.csv"
    if args.data == "national_illness":
        args.data_path = "national_illness.csv"
    if args.data == "motor":
        args.data_path = "motortemp.csv"
    if args.data == "GD":
        args.data_path = "GD.csv"
    if args.data == "MRO":
        args.data_path = "MRO.csv"

    if flag == 'test':
        shuffle_flag = False if args.model != "ARIMA" else True
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        config=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(f"\nLength of {flag} set: {len(data_set)}")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
