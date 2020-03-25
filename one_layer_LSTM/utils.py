
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset,DataLoader

import torch.nn as nn

import matplotlib.pyplot as plt

import plotly.express as px




NB_DAYS = 1913
DAYS = range(1, 1913 + 1)




# def separate_and_scale(train,val_size,fit_size):
#
#     other_columns = train[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']]
#     ts_columns = [f'd_{i}' for i in DAYS]
#
#     scaler = MinMaxScaler()
#     scaler = scaler.fit(train[ts_columns])
#     train[ts_columns] = scaler.transform(train[ts_columns])
#
#     fit_days = train[ts_columns].iloc[:,NB_DAYS-val_size-fit_size:NB_DAYS-val_size]
#     val_days = train[ts_columns].iloc[:,NB_DAYS-val_size:NB_DAYS]
#
#     # fit = pd.concat([other_columns,fit_days],axis=1)
#     # val = pd.concat([other_columns,val_days],axis=1)
#
#     return fit_days,val_days
#
def ts_scale(train):
    ts_columns = [f'd_{i}' for i in DAYS]
    scaler = MinMaxScaler()
    scaler = scaler.fit(train[ts_columns])
    train[ts_columns] = scaler.transform(train[ts_columns])
    return train
#
#
#
# def rmsse(y_pred,y_true,n,h=28):
#     num = np.sum((y_true.iloc[n:n+h-1] - y_pred.iloc[n:n+h-1])**2)
#     den = 1/(n-1)*np.sum(y_true.iloc[1:n-1].shift(-1) - y_pred.iloc[1:n-1])**2
#     compose = (1/h)*num/den
#     return np.sqrt(compose)
####################################################################################
