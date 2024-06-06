from data_loader import Dataset_ETT_hour
from dataclasses import dataclass

@dataclass
class Args:
    data: str
    embed: str
    batch_size: int
    freq: str
    root_path: str
    data_path: str
    seq_len: int
    label_len: int
    pred_len: int
    features: int
    target: int
    num_workers: int

args = Args(
    

data_set = Dataset_ETT_hour(
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