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

    nll = np.zeros((len(char), 20))
    nll_all = 0

    for i in range(20):
        train_data = data[(data['eom']>=19910131+i*10000) & (data['eom']<=19971231+i*10000)]
        train_data = train_data.drop(info, axis = 1)
        y_train = train_data[y].values.reshape(-1)

        print('\ncurrent testing year: {}\n'.format(2001+i))

        model_folder = "../hnn/results/"+str(2001+i)+"/"
        model_mu_list = []
        model_var_list = []
        for n in range(nn_Num):
            model_mu_temp  = model_nn(train_data.shape[1]-1, hidden_dims)
            model_mu_temp.load_state_dict(torch.load(f'{model_folder}/model_mu_{hidden_dims}_{n}.pth'))
            model_mu_list.append(model_mu_temp)

            model_var_temp  = variance_nn(train_data.shape[1]-1, hidden_dims)
            model_var_temp.load_state_dict(torch.load(f'{model_folder}/model_var_{hidden_dims}_{n}.pth'))
            model_var_list.append(model_var_temp)

        x_train = train_data.drop(y, axis = 1)
        x_train = torch.tensor(x_train.values, dtype = torch.float32)
        with torch.no_grad():
            mu_ensem = np.zeros((len(train_data), nn_Num))
            for n in range(nn_Num):
                mu_ensem[:, n] = model_mu_list[n](x_train).detach().numpy().reshape(-1)
        pred = mu_ensem.mean(axis = 1)

        for j, v in enumerate(char):
            print("ensemble: testing year {}, current variable: {}\n".format(2001+i, v))
            x_train = train_data.drop(y, axis = 1)
            x_train[v] = 0
            x_train = torch.tensor(x_train.values, dtype = torch.float32)
            with torch.no_grad():
                var_ensem = np.zeros((len(train_data), nn_Num))
                for n in range(nn_Num):
                    var_ensem[:, n] = model_var_list[n](x_train).detach().numpy().reshape(-1)
            var = var_ensem.mean(axis = 1)
            nll[j, i] = (np.log(var) + (y_train - pred) ** 2 / var).mean()

        x_train = train_data.drop(y, axis = 1)
        x_train = torch.tensor(x_train.values, dtype = torch.float32)
        with torch.no_grad():
            var_ensem = np.zeros((len(train_data), nn_Num))
            for n in range(nn_Num):
                var_ensem[:, n] = model_var_list[n](x_train).detach().numpy().reshape(-1)
        var = var_ensem.mean(axis = 1)
        nll_all += (np.log(var) + (y_train - pred) ** 2 / var).mean()

    results[str(hidden_dims)] = nll.sum(axis = 1) - nll_all

results.to_csv("var_imp.csv", index=True, index_label='char')