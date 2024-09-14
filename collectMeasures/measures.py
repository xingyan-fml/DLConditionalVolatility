# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pandas as pd

#==============================================================================

def cal_turnover(data, sorting, numGroups):
    allIndices = list(itertools.product(data['date'].unique(), data['id'].unique()))
    allData = data.set_index(['date', 'id'])
    restIndices = list(set(allIndices) - set(allData.index.tolist()))
    restIndices = pd.MultiIndex.from_tuples(restIndices, names = ('date', 'id'))
    allData = pd.concat([allData, pd.DataFrame(0, restIndices, allData.columns)]).sort_index()

    func = lambda x: -1 if x == 1 else np.floor(x / numGroups)
    if sorting == 'univariate':
        allData['position'] = allData['group'].apply(func)
        allData['weights'] = allData['weights'] * allData['position']
    else:
        allData['position'] = allData['group_mean'].apply(func)
        allData['weights'] = allData['weights'] * allData['position']
        tmp = allData.groupby(['date', 'position'])['weights'].transform('sum')
        tmp = tmp.apply(lambda x: 1 if x == 0 else abs(x))
        allData['weights'] = allData['weights'] / tmp

    allData['weights+1m'] = allData.groupby('id')['weights'].transform(lambda x: x.shift(-1))
    allData = allData[allData['weights+1m'].notna()]
    tmp = (allData['weights'] * allData['true']).groupby('date').transform('sum')
    tmp = allData['weights'] * (1 + allData['true']) / (1 + tmp)
    turnover = (allData['weights+1m'] - tmp).abs().groupby('date').sum().mean()
    return turnover

def make_table(data, portfolios, model_name, sorting, weight, numGroups):
    idx = pd.IndexSlice
    with open("measures.txt", 'a') as file:

        if sorting == 'univariate':
            short = portfolios.loc[idx[:, 1], :].droplevel('group')
            long = portfolios.loc[idx[:, numGroups], :].droplevel('group')
            zeronet = long - short
        else:
            short = portfolios.loc[idx[:, 1, :], :].droplevel('group_mean')
            long = portfolios.loc[idx[:, numGroups, :], :].droplevel('group_mean')
            zeronet = (long - short).groupby('date').mean()

        returns = zeronet['true_' + weight]
        returns.index = [np.int64(x / 100) for x in returns.index]
        FF6 = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv').set_index('date')
        riskFree = FF6['RF'].shift(-1).loc[returns.index] / 100
        returns = returns + riskFree

        riskFree = riskFree.to_frame().reset_index()
        riskFree.columns = ['date_m', 'RF']
        newData = pd.DataFrame(data, copy = True)
        newData['date_m'] = newData['date'].apply(lambda x: np.int64(x / 100))
        newData = pd.merge(newData, riskFree, on = 'date_m')
        newData['mean'] = newData['mean'] + newData['RF']
        newData['true'] = newData['true'] + newData['RF']

        nav = pd.concat([pd.Series([1]), (1 + returns).cumprod()])
        dd = nav / nav.cummax() - 1
        mdd = - dd.min() * 100

        m1ml = - returns.min() * 100
        turnover = cal_turnover(newData, sorting, numGroups) * 100
        file.write(f'{model_name},{sorting},{weight},{mdd:.2f},{m1ml:.2f},{turnover:.2f}\n')

#==============================================================================

with open("measures.txt", 'w') as file:
    file.write('model_name,sorting,weight,MDD,M1ML,turnover\n')

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
    make_table(data, portfolios, Model_names[i], 'univariate', 'ew', numGroups)

    # value-wegihted
    data['weights'] = data['me']/data.groupby(['date', 'group'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['weights']
    data['true_vw'] = data['true'] * data['weights']
    portfolios = data.groupby(['date', 'group'])[['mean_vw', 'true_vw']].sum()
    make_table(data, portfolios, Model_names[i], 'univariate', 'vw', numGroups)

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
    make_table(data, portfolios, Model_names[i], 'independent', 'ew', numGroups)

    # value-weighted
    data['weights'] = data['me']/data.groupby(['date', 'group_mean', 'group_std'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['weights']
    data['true_vw'] = data['true'] * data['weights']
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean_vw', 'true_vw']].sum()
    make_table(data, portfolios, Model_names[i], 'independent', 'vw', numGroups)

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
    make_table(data, portfolios, Model_names[i], 'dependent', 'ew', numGroups)

    # value-weighted
    data['weights'] = data['me']/data.groupby(['date', 'group_mean', 'group_std'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['weights']
    data['true_vw'] = data['true'] * data['weights']
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean_vw', 'true_vw']].sum()
    make_table(data, portfolios, Model_names[i], 'dependent', 'vw', numGroups)



