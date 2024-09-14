# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

#==============================================================================
# Average Log Likelihood

def LogGaussian(y_true, pred_mean, pred_var):
    lg = - np.mean(np.log(2 * np.pi * pred_var) + ((y_true - pred_mean) ** 2) / pred_var) / 2
    return lg

#==============================================================================
# Performance Evaluation

Hidden_dims = [[32], [32, 16], [32, 16, 8], [32, 16, 8, 4], [32, 16, 8, 4, 2]]
results = pd.DataFrame(0, index=['hnn','hnn_top','hnn_bottom'], columns=[str(x) for x in Hidden_dims])

datatype='hnn'
for hidden_dims in Hidden_dims:
    data = pd.read_csv('../hnn/results/'+str(hidden_dims)+'_0.001_1e-05_1_10.csv')

    lg_value = LogGaussian(data['true'], data['mean'], data['var'])
    results.loc[datatype, str(hidden_dims)] = lg_value

datatype='hnn_top'
for hidden_dims in Hidden_dims:
    data = pd.read_csv('../hnn_top/results/'+str(hidden_dims)+'_0.001_1e-05_1_10_top.csv')

    lg_value = LogGaussian(data['true'], data['mean'], data['var'])
    results.loc[datatype, str(hidden_dims)] = lg_value

datatype='hnn_bottom'
for hidden_dims in Hidden_dims:
    data = pd.read_csv('../hnn_bottom/results/'+str(hidden_dims)+'_0.001_1e-05_1_10_bottom.csv')

    lg_value = LogGaussian(data['true'], data['mean'], data['var'])
    results.loc[datatype, str(hidden_dims)] = lg_value

results.to_csv("LogGaussian.csv")