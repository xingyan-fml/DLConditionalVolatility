# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import t

#==============================================================================

def make_table(results, table_name):
    with open(table_name + ".txt", 'w') as file:
        file.write(' & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 \\\\\n') 
        file.write("\\midrule\n")
        
        for idx, value in results.iterrows():
            file.write(str(value.iloc[0]))
            for i in np.arange(1, len(value)):
                if i > 1 and value.iloc[i] < value.iloc[i-1]:
                    file.write(' & {:.3f}$^*$'.format(value.iloc[i]))
                else:
                    file.write(' & {:.3f}'.format(value.iloc[i]))
            file.write(' \\\\\n')

#==============================================================================

Hidden_dims = [[32], [32, 16], [32, 16, 8], [32, 16, 8, 4], [32, 16, 8, 4, 2]]
numGroups = 10

if not os.path.isdir("tables/"):
    os.mkdir("tables/")

for hidden_dims in Hidden_dims:

    data = pd.read_csv('../hnn/results/'+str(hidden_dims)+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)
    
    data['rank'] = data.groupby('date')['std'].rank(pct = True)
    data['group'] = (data['rank'] * numGroups).apply(np.ceil)
    
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data['year'] = data['date'].dt.year
    
    portfolio = data.groupby(['year', 'group'])[['true']].std()
    portfolio = portfolio.rename(columns={'true':'real_std'})
    
    results = pd.DataFrame(columns = np.arange(numGroups) + 1)
    for idx, value in portfolio.iterrows():
        results.loc[idx] = value.item()
    
    results.to_csv("tables/" + str(hidden_dims) + '_vol.csv')
    make_table(results.reset_index(), "tables/" + str(hidden_dims) + '_vol')