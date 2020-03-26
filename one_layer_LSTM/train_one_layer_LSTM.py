import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm_notebook
from tensorboard_logger import TensorboardLogger
import numpy as np
from typing import Dict, NoReturn
from one_layer_LSTM_model import LstmModel


def train_lstm(
    model: LstmModel,
    criterion: torch.nn.modules.loss,
    optimizer: torch.optim,
    train_loader: torch.utils.data.dataloader.DataLoader,
    val_loader: torch.utils.data.dataloader.DataLoader,
    device: str,
    verbose: bool,
    n_epochs: int,
    kwargs_writer: Dict[str, str] = None,
) -> NoReturn:

    """Short summary.

    Parameters
    ----------
    model : LstmModel
        Description of parameter `model`.
    criterion : torch.nn.modules.loss
        Description of parameter `criterion`.
    optimizer : torch.optim
        Description of parameter `optimizer`.
    train_loader : torch.utils.data.dataloader.DataLoader
        Description of parameter `train_loader`.
    val_loader : torch.utils.data.dataloader.DataLoader
        Description of parameter `val_loader`.
    device : str
        Description of parameter `device`.
    verbose : bool
        Description of parameter `verbose`.
    n_epochs : int
        Description of parameter `n_epochs`.
    kwargs_writer : Dict[str, str]
        Description of parameter `kwargs_writer`.

    Returns
    -------
    NoReturn
        Description of returned object.

    """

    model = model.to(device)
    dict_loader = {"fit": train_loader, "val": val_loader}

    writer = TensorboardLogger(kwargs_writer)
    global_step_fit = 0
    glob_step_val = 0
    for epoch in tqdm_notebook(range(1, n_epochs + 1)):
        # monitor training loss
        fit_loss = 0.0

        ###################
        # train the model #
        ###################
        epoch_losses = {}
        for phase in ["fit", "val"]:

            for chunk in dict_loader[phase]:

                data = chunk["data"].to(device)
                target = chunk["target"].to(device)
                total_loss = 0

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "fit"):

                    outputs = model(data)

                    loss = criterion(outputs, target)
                    if phase == "fit":
                        loss.backward()
                        optimizer.step()
                        writer.add(
                            fit_loss=loss.item(),
                            val_loss=None,
                            model_for_gradient=None,
                            step=global_step_fit,
                        )
                        global_step_fit += 1

                    else:
                        writer.add(
                            fit_loss=None,
                            val_loss=loss.item(),
                            model_for_gradient=None,
                            step=glob_step_val,
                        )
                        glob_step_val += 1

                total_loss += loss.item() * data.size(0)

            epoch_losses.update(
                {f"{phase} loss": total_loss / len(dict_loader[phase].dataset)}
            )
            # if phase == "fit":
            #     writer.add(fit_loss=None, val_loss=None, model_for_gradient=model)

        # print avg training statistics
        if verbose:
            print(
                f'Fit loss: {epoch_losses["fit loss"]:.4f} and Val loss: {epoch_losses["val loss"]:.4f}'
            )
