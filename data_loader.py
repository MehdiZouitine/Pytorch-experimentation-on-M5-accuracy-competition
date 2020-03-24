import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

SIZE = 1913

class DatasetLSTM(Dataset):
    def __init__(self, df,start_day, target_size):

        "Initialization"
        self.list_seq = df
        self.target_size = target_size
        self.start_day = start_day
        self.mean_df = df.mean(axis=1)
        self.std_df = df.std(axis=1)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_seq)

    def __getitem__(self, index):
        "Generates one sample of data"

        data = torch.tensor(
            self.list_seq.iloc[index].values[self.start_day : -self.target_size],
            dtype=torch.float32,
        )
        data = torch.unsqueeze(data, 0).view(-1, 1)
        target = torch.tensor(
            self.list_seq.iloc[index].values[-self.target_size :], dtype=torch.float32,
        )
        target = torch.unsqueeze(target, 0).view(-1, 1)
        return {"data": data, "target": target}
    # data and target of size [sequence_lenght,1]
