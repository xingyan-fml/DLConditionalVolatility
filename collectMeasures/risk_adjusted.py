# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm

#==============================================================================

FF6 = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv')
FF_mom = pd.read_csv('F-F_Momentum_Factor.csv')
FF6 = pd.merge(FF6, FF_mom, on = 'date')
riskFree = FF6[['date', 'RF']].set_index('date')
FF6 = FF6.drop('RF', axis = 1).set_index('date')

def cal_FF6(returns, FF6):
    X = sm.add_constant(FF6.shift(-1).loc[returns.index, :])
    Y = returns
    model = sm.OLS(Y, X).fit()

    mean_ret = Y.mean() * 100
    alpha = model.params['const'] * 100
    t_alpha = model.tvalues['const']
    r_squared = model.rsquared * 100
    ir = model.params['const'] / model.resid.std()

    return mean_ret, alpha, t_alpha, r_squared, ir

#==============================================================================

def make_table(data, portfolios, model_name, sorting, weight, numGroups):
    idx = pd.IndexSlice
    with open("risk_adjusted.txt", 'a') as file:

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
        mean_ret, alpha, t_alpha, r_squared, ir = cal_FF6(returns, FF6)
        file.write(f'{model_name},{sorting},{weight},{mean_ret:.2f},{alpha:.2f},{t_alpha:.2f},{r_squared:.2f},{ir:.2f}\n')

#==============================================================================

with open("risk_adjusted.txt", 'w') as file:
    file.write('model_name,sorting,weight,mean_ret,alpha,t_alpha,r_squared,ir\n')

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



