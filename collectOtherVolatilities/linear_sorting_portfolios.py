# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#==============================================================================

def performance(mean_series, true_series):
    return [100 * mean_series.mean(), 
            100 * true_series.mean(), 
            100 * true_series.std(), 
            ( true_series.mean() / true_series.std() ) * np.sqrt(12)]

def make_table(portfolios, model_name, sorting, weight, numGroups):
    idx = pd.IndexSlice
    with open("linear_portfolio_" + weight + ".txt", 'a') as file:

        for p in range(numGroups):
            if sorting == 'univariate':
                temp = portfolios.loc[idx[:, p+1], :].droplevel('group')
            else:
                temp = portfolios.loc[idx[:, p+1, :], :].droplevel('group_mean')
                temp = temp.groupby('date').mean()
            result = performance(temp['mean_' + weight], temp['true_' + weight])
            if p == 0:
                file.write(f'{model_name},{sorting},Low(L),{result[0]:.2f},{result[1]:.2f},{result[2]:.2f},{result[3]:.2f}\n')
            elif p == numGroups - 1:
                file.write(f'{model_name},{sorting},High(H),{result[0]:.2f},{result[1]:.2f},{result[2]:.2f},{result[3]:.2f}\n')
            else:
                file.write(f'{model_name},{sorting},{p+1},{result[0]:.2f},{result[1]:.2f},{result[2]:.2f},{result[3]:.2f}\n')

        if sorting == 'univariate':
            short = portfolios.loc[idx[:, 1], :].droplevel('group')
            long = portfolios.loc[idx[:, numGroups], :].droplevel('group')
            zeronet = long - short
        else:
            short = portfolios.loc[idx[:, 1, :], :].droplevel('group_mean')
            long = portfolios.loc[idx[:, numGroups, :], :].droplevel('group_mean')
            zeronet = (long - short).groupby('date').mean()
        result = performance(zeronet['mean_' + weight], zeronet['true_' + weight])
        file.write(f'{model_name},{sorting},H-L,{result[0]:.2f},{result[1]:.2f},{result[2]:.2f},{result[3]:.2f}\n')

#==============================================================================

with open("linear_portfolio_ew.txt", 'w') as file:
    file.write('model_name,sorting,group,Pred,Real,Std,SR\n')
with open("linear_portfolio_vw.txt", 'w') as file:
    file.write('model_name,sorting,group,Pred,Real,Std,SR\n')

Models = ['[32]', '[32, 16]', '[32, 16, 8]', '[32, 16, 8, 4]', '[32, 16, 8, 4, 2]']
Model_names = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']

for i, model in enumerate(Models):
    idx = pd.IndexSlice

    data = pd.read_csv('linear/results/'+model+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)

    #==============================================================================
    # univariate sorting

    numGroups = 10

    data['rank'] = data.groupby('date')['mean'].rank(pct = True)
    data['group'] = (data['rank'] * numGroups).apply(np.ceil)

    # equal-weighted
    portfolios = data.groupby(['date', 'group'])[['mean', 'true']].mean()
    portfolios = portfolios.rename(columns={'mean':'mean_ew', 'true':'true_ew'})
    make_table(portfolios, Model_names[i], 'univariate', 'ew', numGroups)

    # value-wegihted
    data['value_weights'] = data['me']/data.groupby(['date', 'group'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['value_weights']
    data['true_vw'] = data['true'] * data['value_weights']
    portfolios = data.groupby(['date', 'group'])[['mean_vw', 'true_vw']].sum()
    make_table(portfolios, Model_names[i], 'univariate', 'vw', numGroups)

    #==============================================================================
    # independent sorting

    numGroups = 10
    data.drop(['rank', 'group', 'value_weights', 'mean_vw', 'true_vw'], axis = 1, inplace = True)

    data['rank_mean'] = data.groupby('date')['mean'].rank(pct = True)
    data['rank_std'] = data.groupby('date')['std'].rank(pct = True, ascending = False)
    data['group_mean'] = (data['rank_mean'] * numGroups).apply(np.ceil)
    data['group_std'] = (data['rank_std'] * numGroups).apply(np.ceil)

    # equal-weighted
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean', 'true']].mean()
    portfolios = portfolios.rename(columns={'mean':'mean_ew', 'true':'true_ew'})
    make_table(portfolios, Model_names[i], 'independent', 'ew', numGroups)

    # value-weighted
    data['value_weights'] = data['me']/data.groupby(['date', 'group_mean', 'group_std'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['value_weights']
    data['true_vw'] = data['true'] * data['value_weights']
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean_vw', 'true_vw']].sum()
    make_table(portfolios, Model_names[i], 'independent', 'vw', numGroups)

    #==============================================================================
    # dependent sorting

    numGroups = 10
    data.drop(['rank_mean', 'rank_std', 'group_mean', 'group_std', 'value_weights', 'mean_vw', 'true_vw'], axis = 1, inplace = True)

    data['rank_std'] = data.groupby('date')['std'].rank(pct = True, ascending = False)
    data['group_std'] = (data['rank_std'] * numGroups).apply(np.ceil)
    data['rank_mean'] = data.groupby(['date', 'group_std'])['mean'].rank(pct = True)
    data['group_mean'] = (data['rank_mean'] * numGroups).apply(np.ceil)

    # equal-weighted
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean', 'true']].mean()
    portfolios = portfolios.rename(columns={'mean':'mean_ew', 'true':'true_ew'})
    make_table(portfolios, Model_names[i], 'dependent', 'ew', numGroups)

    # value-weighted
    data['value_weights'] = data['me']/data.groupby(['date', 'group_mean', 'group_std'])['me'].transform('sum')
    data['mean_vw'] = data['mean'] * data['value_weights']
    data['true_vw'] = data['true'] * data['value_weights']
    portfolios = data.groupby(['date', 'group_mean', 'group_std'])[['mean_vw', 'true_vw']].sum()
    make_table(portfolios, Model_names[i], 'dependent', 'vw', numGroups)



