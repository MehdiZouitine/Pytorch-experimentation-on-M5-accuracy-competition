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
            num_layer=self.n_layers,
            bias=True,
            dropout=self.dropout,
        )
        # Output of LSTM will be (seq_len, batch_size, num_directions=1 * hidden_dim) num_directions=1 because it's time series

        INPUT_FC = self.sequence_len * self.hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(in_features=INPUT_FC, out_features=self.hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.hidden_dim // 2, out_features=self.latent_dim),
        )

    def forward(
        self, sequence: torch.Tensor, hidden_0: torch.Tensor, cell_0: torch.Tensor
    ) -> torch.Tensor:

        batch_size = sequence.size(0)

        output, hidden_state = self.lstm(sequence[:, i], (hidden_0, cell_0))
        # Output : output,(hidden,cell) see more here https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
        # size hidden_state : (batch_size, self.hidden_dim),(batch_size, self.hidden_dim)

        lattent_vector = self.fc(output)
        lattent_vector = torch.unsqueeze(lattent_vector, 2)
        # size of lattent_vector : torch.Size([batch_size,lattent_dim, 1])

        # Check hidden_state size
        return {
            "lattent vector": lattent_vector,
            "hidden embedded": hidden_state[0],
            "cell hidden_embedded ": hidden_state[1],
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
            num_layer=self.n_layers,
            bias=True,
            dropout=self.dropout,
        )
        # Output of LSTM will be (seq_len, batch_size, num_directions=1 * hidden_dim) num_directions=1 because it's time series

        INPUT_FC = self.latent_dim * self.hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(in_features=INPUT_FC, out_features=self.hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.hidden_dim // 2, out_features=self.sequence_len),
        )

    def forward(
        self,
        lattent_vector: torch.Tensor,
        hidden_embedded: torch.Tensor,
        cell_embedded: torch.Tensor,
    ) -> torch.Tensor:

        output, hidden_state = self.lstm(
            lattent_vector[:, i], (hidden_embedded, cell_embedded)
        )
        # Output : output,(hidden,cell) see more here https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm

        decoded_sequence = self.fc(output)
        decoded_sequence = torch.unsqueeze(decoded_sequence, 2)

        # size of out : torch.Size([batch_size,lattent_dim, 1])
        # size hidden_state : (batch_size, self.hidden_dim),(batch_size, self.hidden_dim)

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
        hidden_0, cell_0 = self.init_hidden(batch_size=batch_size, device=self.device)
        encoded_sequence = encoder(sequence, hidden_O, cell_0)
        decoded_sequence = decoder(
            encoded_sequence["lattent vector"],
            encoded_sequence["hidden embedded"],
            encoded_sequence["cell embedded"],
        )
        return decoded_sequence
