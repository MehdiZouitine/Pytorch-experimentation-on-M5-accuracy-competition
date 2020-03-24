import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm_notebook




def train_lstm(model,criterion,optimizer,train_loader,val_loader,device,verbose,n_epochs):
    model = model.to(device)
    dict_loader = {"fit": train_loader, "val": val_loader}
    fit_loss_hist = []
    val_loss_hist = []

    for epoch in tqdm_notebook(range(1, n_epochs + 1)):
        # monitor training loss
        train_loss = 0.0

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

                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(data)

                    loss = criterion(outputs, target)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                total_loss += loss.item() * data.size(0)
            epoch_losses.update(
                {f"{phase} loss": total_loss / len(dict_loader[phase].dataset)}
            )
        # print avg training statistics
        if verbose:
            fit_loss_hist.append(epoch_losses["fit loss"])
            val_loss_hist.append(epoch_losses["val loss"])
            print(
                f'Fit loss: {epoch_losses["fit loss"]:.4f} and Val loss: {epoch_losses["val loss"]:.4f}'
            )

    return fit_loss_hist,val_loss_hist
