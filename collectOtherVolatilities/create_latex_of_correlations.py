# -*- coding: utf-8 -*-

import itertools
import numpy as np
import pandas as pd
from scipy import stats

from arch import arch_model

#==============================================================================

Models = ['[32]', '[32, 16]', '[32, 16, 8]', '[32, 16, 8, 4]', '[32, 16, 8, 4, 2]']
Model_names = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']
vols = ['ours', 'ivol', 'ivol_ff3', 'rvol', 'garch', 'gjr-garch', 'linear']
corrs = pd.DataFrame(index = Model_names, columns = vols)
pvalues = pd.DataFrame(index = Model_names, columns = vols)

#==============================================================================

info = ['id','eom','me']
y = ['ret_exc_lead1m']
data = pd.read_csv('../preprocessing/usa_new.csv')[info+y]

allIndices = list(itertools.product(data['eom'].unique(), data['id'].unique()))
allData = data.set_index(['eom', 'id'])
restIndices = list(set(allIndices) - set(allData.index.tolist()))
restIndices = pd.MultiIndex.from_tuples(restIndices, names = ('eom', 'id'))
allData = pd.concat([allData, pd.DataFrame(None, restIndices, allData.columns)])
allData = allData.sort_index().reset_index()

collect = []
for sid, sdata in allData.groupby('id'):
    print(sid)
    
    for i in range(20):
        splitInt = 20010131+i*10000
        split = str(splitInt)
        split = split[:4]+'-'+split[4:6]+'-'+split[6:8]
        trainTest = pd.DataFrame(sdata[(sdata['eom']>=19910131+i*10000) & (sdata['eom']<=20011231+i*10000)], copy=True)
        
        trainTest['eom'] = trainTest['eom'].apply(lambda x: str(x))
        trainTest['eom'] = trainTest['eom'].apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:8])
        trainTest = trainTest.set_index('eom')['ret_exc_lead1m'].dropna() * 100
        trainTest.index = pd.to_datetime(trainTest.index)
        trainTest.index.name = 'date'
        if len(trainTest.loc[:split]) < 6 or trainTest.loc[split:].empty:
            continue
        
        am = arch_model(trainTest, mean='Constant', vol='GARCH', p=1, o=0, q=1, dist="Normal")
        res = am.fit(last_obs=split, update_freq = 0, disp = 'off', show_warning = False)
        tmp = np.sqrt(res.forecast().variance)
        
        tmp.columns = ['cv']
        tmp['id'] = sid
        tmp.reset_index(inplace = True)
        tmp['date'] = tmp['date'].apply(lambda x: int(x.strftime("%Y%m%d")))
        tmp = tmp[tmp['date'] >= splitInt]
        collect.append(tmp)

preds = pd.concat(collect, ignore_index = True)

for i, model in enumerate(Models):
    data = pd.read_csv('../hnn/results/'+model+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)
    
    data = pd.merge(data, preds, how = 'inner', on = ['date', 'id'])
    data['std'] = data['cv']
    
    res = stats.pearsonr(data['mean'], data['std'])
    corrs.loc[Model_names[i], 'garch'] = res.statistic
    pvalues.loc[Model_names[i], 'garch'] = res.pvalue

#==============================================================================

collect = []
for sid, sdata in allData.groupby('id'):
    print(sid)
    
    for i in range(20):
        splitInt = 20010131+i*10000
        split = str(splitInt)
        split = split[:4]+'-'+split[4:6]+'-'+split[6:8]
        trainTest = pd.DataFrame(sdata[(sdata['eom']>=19910131+i*10000) & (sdata['eom']<=20011231+i*10000)], copy=True)
        
        trainTest['eom'] = trainTest['eom'].apply(lambda x: str(x))
        trainTest['eom'] = trainTest['eom'].apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:8])
        trainTest = trainTest.set_index('eom')['ret_exc_lead1m'].dropna() * 100
        trainTest.index = pd.to_datetime(trainTest.index)
        trainTest.index.name = 'date'
        if len(trainTest.loc[:split]) < 6 or trainTest.loc[split:].empty:
            continue
        
        am = arch_model(trainTest, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist="Normal")
        res = am.fit(last_obs=split, update_freq = 0, disp = 'off', show_warning = False)
        tmp = np.sqrt(res.forecast().variance)
        
        tmp.columns = ['cv']
        tmp['id'] = sid
        tmp.reset_index(inplace = True)
        tmp['date'] = tmp['date'].apply(lambda x: int(x.strftime("%Y%m%d")))
        tmp = tmp[tmp['date'] >= splitInt]
        collect.append(tmp)

preds = pd.concat(collect, ignore_index = True)

for i, model in enumerate(Models):
    data = pd.read_csv('../hnn/results/'+model+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)
    
    data = pd.merge(data, preds, how = 'inner', on = ['date', 'id'])
    data['std'] = data['cv']
    
    res = stats.pearsonr(data['mean'], data['std'])
    corrs.loc[Model_names[i], 'gjr-garch'] = res.statistic
    pvalues.loc[Model_names[i], 'gjr-garch'] = res.pvalue

#==============================================================================

for i, model in enumerate(Models):
    data = pd.read_csv('linear/results/'+model+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)
    res = stats.pearsonr(data['mean'], data['std'])
    corrs.loc[Model_names[i], 'linear'] = res.statistic
    pvalues.loc[Model_names[i], 'linear'] = res.pvalue

#==============================================================================

for i, model in enumerate(Models):
    data = pd.read_csv('../hnn/results/'+model+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)
    res = stats.pearsonr(data['mean'], data['std'])
    corrs.loc[Model_names[i], 'ours'] = res.statistic
    pvalues.loc[Model_names[i], 'ours'] = res.pvalue
    
    usa = pd.read_csv('../preprocessing/usa_new.csv')[['eom', 'id', 'ivol_capm_252d', 'ivol_ff3_21d', 'rvol_21d']]
    usa.columns = ['date', 'id', 'ivol_capm_252d', 'ivol_ff3_21d', 'rvol_21d']
    data = pd.merge(data, usa, how = 'left', on = ['date', 'id'])
    
    res = stats.pearsonr(data['mean'], data['ivol_capm_252d'])
    corrs.loc[Model_names[i], 'ivol'] = res.statistic
    pvalues.loc[Model_names[i], 'ivol'] = res.pvalue
    
    res = stats.pearsonr(data['mean'], data['ivol_ff3_21d'])
    corrs.loc[Model_names[i], 'ivol_ff3'] = res.statistic
    pvalues.loc[Model_names[i], 'ivol_ff3'] = res.pvalue
    
    res = stats.pearsonr(data['mean'], data['rvol_21d'])
    corrs.loc[Model_names[i], 'rvol'] = res.statistic
    pvalues.loc[Model_names[i], 'rvol'] = res.pvalue

#==============================================================================

with open("create_latex_of_correlations.txt", 'w') as file:
    file.write("& NN$_k$ & ivol\\_capm\\_252d & ivol\\_ff3\\_21d & rvol\\_21d & GARCH & GJR-GARCH & Linear \\\\\n")
    for model in Model_names:
        file.write(model)
        for vol in corrs.columns:
            # file.write(" & \\makecell{{{:.2f}\\\\ ({:.2f})}}".format(corrs.loc[model, vol], pvalues.loc[model, vol]))
            file.write(" & {:.2f}".format(corrs.loc[model, vol]))
        file.write(" \\\\\n")



