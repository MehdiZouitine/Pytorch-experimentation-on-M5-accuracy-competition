import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm_notebook
from tensorboard_logger import TensorboardLogger
import numpy as np


def train_lstm(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    device,
    verbose,
    n_epochs,
    kwargs_writer=None,
):
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
