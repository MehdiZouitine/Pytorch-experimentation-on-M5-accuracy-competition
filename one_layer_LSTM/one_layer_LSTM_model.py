import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NoReturn


class LstmModel(nn.Module):
    """Short summary.

    Parameters
    ----------
    hidden_dim : int
        Description of parameter `hidden_dim`.
    batch_size : int
        Description of parameter `batch_size`.
    target_size : int
        Description of parameter `target_size`.
    sequence_length : int
        Description of parameter `sequence_length`.
    n_layers : int
        Description of parameter `n_layers`.
    device : str
        Description of parameter `device`.


    """

    def __init__(
        self,
        hidden_dim: int,
        batch_size: int,
        target_size: int,
        sequence_length: int,
        n_layers: int,
        device: str,
    ) -> NoReturn:

        """Short summary.

        Parameters
        ----------
        hidden_dim : int
            Description of parameter `hidden_dim`.
        batch_size : int
            Description of parameter `batch_size`.
        target_size : int
            Description of parameter `target_size`.
        sequence_length : int
            Description of parameter `sequence_length`.
        n_layers : int
            Description of parameter `n_layers`.
        device : str
            Description of parameter `device`.

        Returns
        -------
        NoReturn
            Description of returned object.

        """
        super(LstmModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.target_size = target_size

        self.batch_size = batch_size

        self.device = device

        self.fc = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=hidden_dim // 2, out_features=self.target_size),
        )
        self.sequence_length = sequence_length

        self.lstm_cell = nn.LSTMCell(input_size=1, hidden_size=hidden_dim, bias=True)

    def init_hidden(self, device: str, batch_size: int) -> torch.Tensor:

        """Short summary.

        Parameters
        ----------
        device : str
            Description of parameter `device`.
        batch_size : int
            Description of parameter `batch_size`.

        Returns
        -------
        torch.Tensor
            Description of returned object.

        """

        # Always give batch_size and device information when we create new tensor
        # initialize the hidden state and the cell state to zeros
        return (
            torch.zeros(batch_size, self.hidden_dim, requires_grad=True).to(
                device=self.device
            ),
            torch.zeros(batch_size, self.hidden_dim, requires_grad=True).to(
                device=self.device
            ),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Short summary.

        Parameters
        ----------
        sequence : torch.Tensor
            Description of parameter `sequence`.

        Returns
        -------
        torch.Tensor
            Description of returned object.

        """

        batch_size = sequence.size(0)
        hidden, cell = self.init_hidden(batch_size=batch_size, device=self.device)
        for i in range(self.sequence_length):
            hidden, cell = self.lstm_cell(sequence[:, i], (hidden, cell))

        out = self.fc(hidden)
        out = torch.unsqueeze(out, 2)
        # size : torch.Size([2048, 28, 1])

        ##AUTOREGRESSIVE PART (DONT WORK BECAUSE NOT DIFFERENTIABLE)
        # outputs = torch.zeros(self.target_size, batch_size, 1, requires_grad=True).to(
        #     device=self.device
        # )

        # outputs[0] = out
        # for target_idx in range(1, self.target_size):
        #     hidden, cell = self.lstm_cell(out, (hidden, cell))
        #     out = self.fc(hidden)
        #     outputs[target_idx] = out
        # outputs = outputs.view(outputs.shape[1], outputs.shape[0], outputs.shape[2])
        # print(outputs.shape)
        # Out have to be shapped like [batch_size,sequence_len,dim_of_xi]
        return out
