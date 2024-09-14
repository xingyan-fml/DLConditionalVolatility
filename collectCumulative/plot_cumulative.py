# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#==============================================================================

def make_table(data, portfolios, model_name, sorting, weight, numGroups):
    if sorting == 'univariate':
        short = portfolios.loc[idx[:, 1], :].droplevel('group')
        long = portfolios.loc[idx[:, numGroups], :].droplevel('group')
        zeronet = long - short
    else:
        short = portfolios.loc[idx[:, 1, :], :].droplevel('group_mean')
        long = portfolios.loc[idx[:, numGroups, :], :].droplevel('group_mean')
        zeronet = (long - short).groupby('date').mean()

    short = short['true_' + weight].groupby('date').mean()
    long = long['true_' + weight].groupby('date').mean()
    zeronet = zeronet['true_' + weight]
    short.index = [np.int64(x / 100) for x in short.index]
    long.index = [np.int64(x / 100) for x in long.index]
    zeronet.index = [np.int64(x / 100) for x in zeronet.index]

    FF6 = pd.read_csv('../collectMeasures/F-F_Research_Data_5_Factors_2x3.csv')
    riskFree = FF6.set_index('date')['RF'].shift(-1).loc[zeronet.index] / 100
    short = short + riskFree
    long = long + riskFree
    zeronet = zeronet + riskFree

    short = - pd.concat([pd.Series([1]), (1 - short).cumprod()])
    long = pd.concat([pd.Series([1]), (1 + long).cumprod()])
    zeronet = pd.concat([pd.Series([1]), (1 + zeronet).cumprod()])
    short.index = short.index.astype(str)
    long.index = long.index.astype(str)
    zeronet.index = zeronet.index.astype(str)
    return short, long, zeronet

#==============================================================================

if not os.path.isdir('plots/'):
    os.mkdir('plots/')
short = {}
long = {}
zeronet = {}

Models = ['[32]', '[32, 16]', '[32, 16, 8]', '[32, 16, 8, 4]', '[32, 16, 8, 4, 2]']
Model_names = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']

for i, model in enumerate(Models):
    idx = pd.IndexSlice

    data = pd.read_csv('../hnn/results/'+model+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)

    #==============================================================================
    # univariate sorting

    numGroups = 10

    data['rank'] = data.groupby('date')['mean'].rank(pct = True)
    data['group'] = (data['rank'] * numGroups).apply(np.ceil)

    # equal-weighted
    data['weights'] = 1/data.groupby(['date', 'group'])['me'].transform('size')
    portfolios = data.groupby(['date', 'group'])[['mean', 'true']].mean()
    portfolios = portfolios.rename(columns={'mean':'mean_ew', 'true':'true_ew'})
    key = 'univariate' + 'ew'
    short[key], long[key], zeronet[key] = make_table(data, portfolios, Model_names[i], 'univariate', 'ew', numGroups)

    # value-wegihted
    data['weights'] = data['me']/data.groupby(['date', 'group'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['weights']
    data['true_vw'] = data['true'] * data['weights']
    portfolios = data.groupby(['date', 'group'])[['mean_vw', 'true_vw']].sum()
    key = 'univariate' + 'vw'
    short[key], long[key], zeronet[key] = make_table(data, portfolios, Model_names[i], 'univariate', 'vw', numGroups)

    #==============================================================================
    # independent sorting

    numGroups = 10
    data.drop(['rank', 'group', 'weights', 'mean_vw', 'true_vw'], axis = 1, inplace = True)

    data['rank_mean'] = data.groupby('date')['mean'].rank(pct = True)
    data['rank_std'] = data.groupby('date')['std'].rank(pct = True, ascending = False)
    data['group_mean'] = (data['rank_mean'] * numGroups).apply(np.ceil)
    data['group_std'] = (data['rank_std'] * numGroups).apply(np.ceil)

    # equal-weighted
    data['weights'] = 1/data.groupby(['date', 'group_mean', 'group_std'])['me'].transform('size')
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean', 'true']].mean()
    portfolios = portfolios.rename(columns={'mean':'mean_ew', 'true':'true_ew'})
    key = 'independent' + 'ew'
    short[key], long[key], zeronet[key] = make_table(data, portfolios, Model_names[i], 'independent', 'ew', numGroups)

    # value-weighted
    data['weights'] = data['me']/data.groupby(['date', 'group_mean', 'group_std'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['weights']
    data['true_vw'] = data['true'] * data['weights']
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean_vw', 'true_vw']].sum()
    key = 'independent' + 'vw'
    short[key], long[key], zeronet[key] = make_table(data, portfolios, Model_names[i], 'independent', 'vw', numGroups)

    #==============================================================================
    # dependent sorting

    numGroups = 10
    data.drop(['rank_mean', 'rank_std', 'group_mean', 'group_std', 'weights', 'mean_vw', 'true_vw'], axis = 1, inplace = True)

    data['rank_std'] = data.groupby('date')['std'].rank(pct = True, ascending = False)
    data['group_std'] = (data['rank_std'] * numGroups).apply(np.ceil)
    data['rank_mean'] = data.groupby(['date', 'group_std'])['mean'].rank(pct = True)
    data['group_mean'] = (data['rank_mean'] * numGroups).apply(np.ceil)

    # equal-weighted
    data['weights'] = 1/data.groupby(['date', 'group_mean', 'group_std'])['me'].transform('size')
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean', 'true']].mean()
    portfolios = portfolios.rename(columns={'mean':'mean_ew', 'true':'true_ew'})
    key = 'dependent' + 'ew'
    short[key], long[key], zeronet[key] = make_table(data, portfolios, Model_names[i], 'dependent', 'ew', numGroups)

    # value-weighted
    data['weights'] = data['me']/data.groupby(['date', 'group_mean', 'group_std'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['weights']
    data['true_vw'] = data['true'] * data['weights']
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean_vw', 'true_vw']].sum()
    key = 'dependent' + 'vw'
    short[key], long[key], zeronet[key] = make_table(data, portfolios, Model_names[i], 'dependent', 'vw', numGroups)

#==============================================================================

    colors = ['blue', 'orange', 'green']
    labelString = {'univariate':'Single sorting', 'independent':'Independent double', 'dependent':'Dependent double'}
    for weight in ['ew', 'vw']:
        pdf = PdfPages('plots/{}_shortlong_{}.pdf'.format(Model_names[i], weight))
        plt.figure(figsize=(8, 4), dpi = 300)
        j = 0
        for sorting in ['univariate', 'independent', 'dependent']:
            plt.plot(long[sorting + weight], label = labelString[sorting] + ', long', color = colors[j])
            plt.plot(short[sorting + weight], label = labelString[sorting] + ', short', color = colors[j], linestyle = 'dashed')
            j = j + 1
        plt.xticks(['200101', '200401', '200701', '201001', '201301', '201601', '201901'])
        plt.grid(True)
        plt.legend(framealpha = 0.5)
        pdf.savefig(bbox_inches = 'tight')
        plt.close()
        pdf.close()

        pdf = PdfPages('plots/{}_zeronet_{}.pdf'.format(Model_names[i], weight))
        plt.figure(figsize=(8, 4), dpi = 300)
        j = 0
        for sorting in ['univariate', 'independent', 'dependent']:
            plt.plot(zeronet[sorting + weight], label = labelString[sorting] + ', zero-net', color = colors[j])
            j = j + 1
        plt.xticks(['200101', '200401', '200701', '201001', '201301', '201601', '201901'])
        plt.yticks([])
        plt.legend(framealpha = 0.5)
        pdf.savefig(bbox_inches = 'tight')
        plt.close()
        pdf.close()

