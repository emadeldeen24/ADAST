import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

import os
import numpy as np


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        self.num_channels = min(X_train.shape)
        if X_train.shape.index(self.num_channels) != 1:  # data dim is #samples, seq_len, #channels
            X_train = X_train.permute(0, 2, 1)
        
        self.len = X_train.shape[0]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, configs):
    # loading path
    train_dataset = torch.load(os.path.join(data_path, "train_" + domain_id + ".pt"))
    valid_dataset = torch.load(os.path.join(data_path, "val_" + domain_id + ".pt"))
    test_dataset = torch.load(os.path.join(data_path, "test_" + domain_id + ".pt"))

    # Loading datasets
    train_dataset = Load_Dataset(train_dataset)
    valid_dataset = Load_Dataset(valid_dataset)
    test_dataset = Load_Dataset(test_dataset)

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=False, num_workers=0)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
                                               shuffle=False, drop_last=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)
    return train_loader, valid_loader, test_loader

