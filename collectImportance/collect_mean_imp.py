# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch

from fit_hnn import *

#==============================================================================

data = pd.read_csv('../preprocessing/usa_new.csv')

info = ['id','eom','me']
y = ['ret_exc_lead1m']

char = data.columns[~data.columns.isin(info+y)].tolist()
data[char] = data.groupby("eom")[char].rank(pct = True)
data[char] = 2 * data[char] - 1

Hidden_dims = [[32,16,8,4,2], [32,16,8,4], [32,16,8], [32,16], [32]]
nn_Num = 10

#==============================================================================

results = pd.DataFrame([], index = char)

for hidden_dims in Hidden_dims:

    s1 = np.zeros((len(char), 20))
    s1_all = 0
    s2 = 0

    for i in range(20):
        train_data = data[(data['eom']>=19910131+i*10000) & (data['eom']<=19971231+i*10000)]
        train_data = train_data.drop(info, axis = 1)
        y_train = train_data[y].values.reshape(-1)

        print('\ncurrent testing year: {}\n'.format(2001+i))

        model_folder = "../hnn/results/"+str(2001+i)+"/"
        model_list = []
        for n in range(nn_Num):
            model_temp  = model_nn(train_data.shape[1]-1, hidden_dims)
            model_temp.load_state_dict(torch.load(f'{model_folder}/model_mu_{hidden_dims}_{n}.pth'))
            model_list.append(model_temp)

        for j, v in enumerate(char):
            print("ensemble: testing year {}, current variable: {}\n".format(2001+i, v))
            x_train = train_data.drop(y, axis = 1)
            x_train[v] = 0
            x_train = torch.tensor(x_train.values, dtype = torch.float32)
            with torch.no_grad():
                mu_ensem = np.zeros((len(train_data), nn_Num))
                for n in range(nn_Num):
                    mu_ensem[:, n] = model_list[n](x_train).detach().numpy().reshape(-1)
            pred = mu_ensem.mean(axis = 1)
            s1[j, i] = ((pred - y_train) ** 2).sum()

        x_train = train_data.drop(y, axis = 1)
        x_train = torch.tensor(x_train.values, dtype = torch.float32)
        with torch.no_grad():
            mu_ensem = np.zeros((len(train_data), nn_Num))
            for n in range(nn_Num):
                mu_ensem[:, n] = model_list[n](x_train).detach().numpy().reshape(-1)
        pred = mu_ensem.mean(axis = 1)
        s1_all += ((pred - y_train) ** 2).sum()
        s2 += (y_train ** 2).sum()

    s1 = s1.sum(axis = 1)
    results[str(hidden_dims)] = (s1 - s1_all) / s2

results.to_csv("mean_imp.csv", index=True, index_label='char')