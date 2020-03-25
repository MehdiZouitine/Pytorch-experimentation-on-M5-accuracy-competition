from typing import Dict, NoReturn
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

SIZE = 1913


class DatasetLSTM(Dataset):
    """Short summary.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        Description of parameter `df`.
    start_day : int
        Description of parameter `start_day`.
    target_size : int
        Description of parameter `target_size`.

    """

    def __init__(
        self, df: pandas.core.frame.DataFrame, start_day: int, target_size: int
    ) -> NoReturn:

        """Short summary.

        Parameters
        ----------
        df : pandas.core.frame.DataFrame
            Description of parameter `df`.
        start_day : int
            Description of parameter `start_day`.
        target_size : int
            Description of parameter `target_size`.

        Returns
        -------
        NoReturn
            Description of returned object.

        """

        "Initialization"

        self.list_seq = df
        self.target_size = target_size
        self.start_day = start_day

    def __len__(self) -> int:
        """Short summary.

        Returns
        -------
        int
            Description of returned object.

        """
        "Denotes the total number of samples"
        return len(self.list_seq)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Short summary.

        Parameters
        ----------
        index : int
            Description of parameter `index`.

        Returns
        -------
        Dict[str,torch.Tensor]
            Description of returned object.

        """
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
