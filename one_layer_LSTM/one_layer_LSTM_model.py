import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmModel(nn.Module):
    def __init__(
        self, hidden_dim, batch_size, target_size, sequence_length, n_layers, device
    ):
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

    def init_hidden(self, device, batch_size):
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

    def forward(self, sequence):

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
