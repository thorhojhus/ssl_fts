from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from augmentations import augmentation
import numpy as np
from dataclasses import dataclass


@dataclass
class config:
    aug_method = "NA"
    aug_rate = (0.5,)
    in_batch_augmentation = False
    in_dataset_augmentation = False
    data_size = 1
    aug_data_size = 1
    seed = 114
    testset_div = 2
    test_time_train = False
    train_mode = 2
    cut_freq = 196
    base_T = 24
    H_order = 6


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        config=config,
        root_path="data/",
        flag="train",
        size=[720, 48, 96],
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = config
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.collect_all_data()
        if self.args.in_dataset_augmentation and self.set_type == 0:
            self.data_augmentation()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]

        if self.args.test_time_train:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 20 * 30 * 24]
            border2s = [18 * 30 * 24, 20 * 30 * 24, 20 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def regenerate_augmentation_data(self):
        self.collect_all_data()
        self.data_augmentation()

    def reload_data(self, x_data, y_data, x_time, y_time):
        self.x_data = x_data
        self.y_data = y_data
        self.x_time = x_time
        self.y_time = y_time

    def collect_all_data(self):
        self.x_data = []
        self.y_data = []
        data_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        mask_data_len = (
            int((1 - self.args.data_size) * data_len) if self.args.data_size < 1 else 0
        )
        for i in range(len(self.data_x) - self.seq_len - self.pred_len + 1):
            if (self.set_type == 0 and i >= mask_data_len) or self.set_type != 0:
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                self.x_data.append(self.data_x[s_begin:s_end])
                self.y_data.append(self.data_y[r_begin:r_end])

    def data_augmentation(self):
        origin_len = len(self.x_data)
        if not self.args.closer_data_aug_more:
            aug_size = [self.args.aug_data_size for i in range(origin_len)]
        else:
            aug_size = [
                int(self.args.aug_data_size * i / origin_len) + 1
                for i in range(origin_len)
            ]

        for i in range(origin_len):
            for _ in range(aug_size[i]):
                aug = augmentation("dataset")
                if self.args.aug_method == "f_mask":
                    x, y = aug.freq_dropout(
                        self.x_data[i], self.y_data[i], dropout_rate=self.args.aug_rate
                    )
                elif self.args.aug_method == "f_mix":
                    rand = float(np.random.random(1))
                    i2 = int(rand * len(self.x_data))
                    x, y = aug.freq_mix(
                        self.x_data[i],
                        self.y_data[i],
                        self.x_data[i2],
                        self.y_data[i2],
                        dropout_rate=self.args.aug_rate,
                    )
                self.x_data.append(x)
                self.y_data.append(y)

    def __getitem__(self, index):
        seq_x = self.x_data[index]
        seq_y = self.y_data[index]
        return seq_x, seq_y

    def __len__(self):
        return len(self.x_data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
