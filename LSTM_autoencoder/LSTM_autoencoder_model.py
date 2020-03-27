import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NoReturn


class Encoder_LSTM(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        n_features: int,
        hidden_dim: int,
        batch_size: int,
        n_layers: int,
        dropout: int,
        latent_dim: int,
        device: str,
    ) -> NoReturn:
        super().__init__()

        self.sequence_len = sequence_len

        self.n_features = n_features

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.n_layers = n_layers

        self.dropout = dropout

        self.latent_dim = latent_dim

        self.device = device

        # Here n_features will be 1 because we got 1D signal (we can add more features)
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            bias=True,
            dropout=self.dropout,
            batch_first=True,
        )
        # output,(hidden,cell) see more here https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
        # Output of LSTM will be (seq_len, batch_size, num_directions=1 * hidden_dim) num_directions = 1 because it's time series
        # if batch_first=True then  (batch_size,seq_len, num_directions=1 * hidden_dim)
        INPUT_FC = self.sequence_len * self.hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(in_features=INPUT_FC, out_features=self.hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.hidden_dim // 2, out_features=self.latent_dim),
        )

    def forward(
        self, sequence: torch.Tensor, hidden_0: torch.Tensor, cell_0: torch.Tensor
    ) -> torch.Tensor:

        # SIZE(sequence) = (batch_size, seq_len, n_features)

        output, (h_embedded, c_embedded) = self.lstm(sequence, (hidden_0, cell_0))

        # SIZE(output) = (seq_len, batch_size, num_directions * hidden_dim) if batch_first=False
        # SIZE(output) = (batch_size,seq_len, num_directions * hidden_dim) if batch_first=True
        # here num_directions = 1 because it's times series (times is ordonned)

        # SIZE(h_embeded) = SIZE(c_embeded) = (batch_size,hidden_dim)

        # output = output.view(-1)
        # fc need a flat vector
        print(output.shape)
        output = output.view(1, output.size(1) * output.size(2))

        lattent_vector = self.fc(output)
        # fc take seq_len*hidden_dim
        lattent_vector = torch.unsqueeze(lattent_vector, 2)
        # SIZE(lattent_vector) = batch_size,lattent_dim, n_features=1)

        # Check hidden_state size
        return {
            "lattent vector": lattent_vector,
            "h_embedded": h_embeded,
            "c_embedded": c_embedded,
        }


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################


class Decoder_LSTM(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        n_features: int,
        hidden_dim: int,
        batch_size: int,
        n_layers: int,
        dropout: int,
        latent_dim: int,
        device: str,
    ) -> NoReturn:

        super().__init__()

        self.sequence_len = sequence_len

        self.n_features = n_features

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.n_layers = n_layers

        self.dropout = dropout

        self.latent_dim = latent_dim

        self.device = device

        # Here we set n_features to 1 because the decoder take the latent vector which is 1 dimensional vector
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_dim,
            num_layers=self.n_layers,
            bias=True,
            dropout=self.dropout,
            batch_first=True,
        )
        # Output of LSTM will be (seq_len, batch_size, num_directions=1 * hidden_dim) num_directions=1 because it's time series
        # output,(hidden,cell) see more here https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm

        INPUT_FC = self.latent_dim * self.hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(in_features=INPUT_FC, out_features=self.hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.hidden_dim // 2, out_features=self.sequence_len),
        )

    def forward(
        self,
        lattent_vector: torch.Tensor,
        h_embedded: torch.Tensor,
        c_embedded: torch.Tensor,
    ) -> torch.Tensor:

        lattent_vector = lattent_vector.view()
        output, (h_decoded, c_embedded) = self.lstm(
            lattent_vector, (h_embedded, c_embedded)
        )

        # SIZE(output) = (lattent_vector, batch_size, num_directions * hidden_dim) if batch_first=False
        # SIZE(output) = (batch_size,lattent_vector, num_directions * hidden_dim) if batch_first=True
        # here num_directions = 1 because it's times series (times is ordonned)

        # SIZE(h_decoded) = SIZE(c_decoded) = (batch_size,hidden_dim)

        # output = output.view(-1)

        decoded_sequence = self.fc(output)
        decoded_sequence = torch.unsqueeze(decoded_sequence, 2)

        # SIZE(out) : (batch_size,sequence_len, n_features=1)
        ## SIZE(h_decoded) = SIZE(c_decoded) = (batch_size,hidden_dim)

        return decoded_sequence


################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
class AutoEncoder_LSTM(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        n_features: int,
        hidden_dim: int,
        batch_size: int,
        n_layers: int,
        dropout: int,
        latent_dim: int,
        device: str,
    ) -> NoReturn:

        super().__init__()

        self.sequence_len = sequence_len

        self.n_features = n_features

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.n_layers = n_layers

        self.dropout = dropout

        self.latent_dim = latent_dim

        self.device = device

        self.encoder = Encoder_LSTM(
            sequence_len,
            n_features,
            hidden_dim,
            batch_size,
            n_layers,
            dropout,
            latent_dim,
            device,
        )

        self.decoder = Decoder_LSTM(
            sequence_len,
            n_features,
            hidden_dim,
            batch_size,
            n_layers,
            dropout,
            latent_dim,
            device,
        )

    def init_hidden(self, device: str, batch_size: int) -> torch.Tensor:
        # Always give batch_size and device information when we create new tensor
        # initialize the hidden state and the cell state to zeros
        # h_0 and c_0 of shape (num_layers * num_directions, batch, hidden_size)
        return (
            torch.zeros(
                self.n_layers, batch_size, self.hidden_dim, requires_grad=True
            ).to(device=self.device),
            torch.zeros(
                self.n_layers, batch_size, self.hidden_dim, requires_grad=True
            ).to(device=self.device),
        )

    def forward(self, sequence):

        batch_size = sequence.size(0)

        hidden_0, cell_0 = self.init_hidden(batch_size=batch_size, device=self.device)
        encoded_sequence = self.encoder(sequence, hidden_0, cell_0)
        decoded_sequence = self.decoder(
            encoded_sequence["lattent vector"],
            encoded_sequence["h_embedded"],
            encoded_sequence["c_embedded"],
        )
        return decoded_sequence
