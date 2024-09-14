# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd
import torch

from fit_hnn import *

#==============================================================================
# Performance Evaluation

def OOS(pred, real):
    pred = np.array(pred).reshape(-1)
    real = np.array(real).reshape(-1)
    s1 = ((pred - real) ** 2).sum()
    s2 = (real ** 2).sum()
    return (1 - s1 / s2) * 100

#==============================================================================

def pred(data_name, info, y, hidden_dims, lr, penalty, num_iter):
    
    data_dir = "results/"
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    data = pd.read_csv('../preprocessing/'+data_name+'.csv')
    
    char = data.columns[~data.columns.isin(info+y)].tolist()
    data[char] = data.groupby("eom")[char].rank(pct = True)
    data[char] = 2 * data[char] - 1
    # data[y] = data[y].clip(-0.5, 1)

#==============================================================================   
    
    num_epochs = 1000
    batch_size = 10000
    set_seed = 10
    nn_Num = 10
    
#==============================================================================

    results_single = pd.DataFrame([], columns = ['mean', 'true', 'var', 'date', 'me', 'id'])
    results_ensemble = pd.DataFrame([], columns = ['mean', 'true', 'var', 'var2', 'date', 'me', 'id'])

    for i in range(20):

        train_data = data[(data['eom']>=19910131+i*10000) & (data['eom']<=19971231+i*10000)]
        train_data = train_data.drop(info, axis = 1)
        train_data[y] = train_data[y].clip(-0.5, 1)
        valid_data = data[(data['eom']>=19980131+i*10000) & (data['eom']<=20001231+i*10000)]
        valid_data = valid_data.drop(info, axis = 1)
        valid_data[y] = valid_data[y].clip(-0.5, 1)
        test_data = data[(data['eom']>=20010131+i*10000) & (data['eom']<=20011231+i*10000)]
        x_test = test_data.drop(info + y, axis = 1)
        x_test = torch.tensor(x_test.values, dtype = torch.float32)
        
        print('\ncurrent testing year: {}\n'.format(2001+i))

#============================================================================== 
# Single
#============================================================================== 

        model_folder = data_dir + str(2001 + i) + '/'
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        model_mu, model_var = fit_model_hnn(train_data, valid_data, y, num_iter, hidden_dims, num_epochs, batch_size, lr, set_seed, penalty)
        torch.save(model_mu.state_dict(), f'{model_folder}/model_mu_{hidden_dims}_0.pth')
        torch.save(model_var.state_dict(), f'{model_folder}/model_var_{hidden_dims}_0.pth')

        with torch.no_grad():
            mu_single = model_mu(x_test).detach().numpy().reshape(-1)
            var_single = model_var(x_test).detach().numpy().reshape(-1)

        results_temp = {
        'mean': mu_single,
        'true': test_data[y].values.reshape(-1),
        'var': var_single,
        'date': test_data['eom'].values,
        'me': test_data['me'].values,
        'id': test_data['id'].values,
        }
        results_temp = pd.DataFrame(results_temp)
        results_single = pd.concat([results_single, results_temp], ignore_index=True)

#============================================================================== 
# Ensemble
#============================================================================== 
        
        mu_ensem = np.zeros((len(test_data), nn_Num))
        var_ensem = np.zeros((len(test_data), nn_Num))
        mu_ensem[:, 0] = mu_single
        var_ensem[:, 0] = var_single
        for n in range(1, nn_Num):
            print("\ntesting year {}, ensemble +{}\n".format(2001+i,n))
            model_mu, model_var = fit_model_hnn(train_data, valid_data, y, num_iter, hidden_dims, num_epochs, batch_size, lr, set_seed + n * 1000, penalty)
            torch.save(model_mu.state_dict(), f'{model_folder}/model_mu_{hidden_dims}_{n}.pth')
            torch.save(model_var.state_dict(), f'{model_folder}/model_var_{hidden_dims}_{n}.pth')
            
            with torch.no_grad():
                mu_ensem[:, n] = model_mu(x_test).detach().numpy().reshape(-1)
                var_ensem[:, n] = model_var(x_test).detach().numpy().reshape(-1)
        pred = mu_ensem.mean(axis = 1)
        var = var_ensem.mean(axis = 1)
        var2 = (var_ensem + mu_ensem ** 2).mean(axis = 1) - mu_ensem.mean(axis = 1) ** 2

        results_temp = {
        'mean': pred,
        'true': test_data[y].values.reshape(-1),
        'var': var,
        'var2': var2,
        'date': test_data['eom'].values,
        'me': test_data['me'].values,
        'id': test_data['id'].values,
        }
        results_temp = pd.DataFrame(results_temp)
        results_ensemble = pd.concat([results_ensemble, results_temp], ignore_index=True)

#============================================================================== 
    
    results_single.to_csv(data_dir + "{}_{}_{}_{}_{}.csv".format(hidden_dims,lr,penalty,num_iter,1), index = False)
    results_ensemble.to_csv(data_dir + "{}_{}_{}_{}_{}.csv".format(hidden_dims,lr,penalty,num_iter,nn_Num), index = False)
    
    r2_single = OOS(results_single['mean'], results_single['true'])
    print('\nfinal single r2 : {}'.format(r2_single))

    r2_ensemble = OOS(results_ensemble['mean'], results_ensemble['true'])
    print('final ensemble r2 : {}\n'.format(r2_ensemble))

    if not os.path.isfile(data_dir + "oos_r2.csv"):
        with open(data_dir + "oos_r2.csv", 'a') as file:
            file.write('hidden_dims,lr,penalty,num_iter,nn_Num,r2\n')
    with open(data_dir + "oos_r2.csv", 'a') as file:
        file.write("\"{}\",{},{},{},{},{}\n".format(str(hidden_dims), lr, penalty, num_iter, 1, r2_single))
        file.write("\"{}\",{},{},{},{},{}\n".format(str(hidden_dims), lr, penalty, num_iter, nn_Num, r2_ensemble))
