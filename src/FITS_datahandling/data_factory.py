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
    args.num_workers = 4
    Data = data_dict[args.data]
    #timeenc = 0 if args.embed != 'timeF' else 1
    timeenc = 1

    args.root_path = 'data'

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

    if flag == 'test':
        shuffle_flag = False
        drop_last = False # True
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
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
