from typing import Dict, NoReturn
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class DatasetAutoencoderLSTM(Dataset):
    """Short summary.

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Description of parameter `df`.
    subsamble_coef : int
        Description of parameter `subsamble_coef`.

    """

    def __init__(
        self, df: pd.core.frame.DataFrame, subsamble_coef: int = 1
    ) -> NoReturn:

        """Short summary.

        Parameters
        ----------
        df : pd.core.frame.DataFrame
            Description of parameter `df`.
        subsamble_coef : int
            Description of parameter `subsamble_coef`.

        Returns
        -------
        NoReturn
            Description of returned object.

        """
        self.list_seq = df
        self.subsamble_coef = subsamble_coef

    def __len__(self) -> int:

        """Short summary.

        Returns
        -------
        int
            Description of returned object.

        """

        return len(self.list_seq)

    def __getitem__(self, index: int) -> torch.Tensor:

        """Short summary.

        Parameters
        ----------
        index : int
            Description of parameter `index`.

        Returns
        -------
        torch.Tensor
            Description of returned object.

        """

        # to subsample data  use numpy's slicing, simply (start:stop:step)
        data = torch.tensor(
            self.list_seq.iloc[index].values[0 : -1 : self.subsamble_coef],
            dtype=torch.float32,
        )
        data = torch.unsqueeze(data, 0)
        # Shape is [sequence_len,1]
        return data
