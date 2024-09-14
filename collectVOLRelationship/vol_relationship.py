# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#==============================================================================

Hidden_dims = [[32], [32, 16], [32, 16, 8], [32, 16, 8, 4], [32, 16, 8, 4, 2]]
Model_names = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']

results = pd.DataFrame(columns = ['date', 'model', 'gamma', 't_gamma', 'r_squared', 'number'])
for i in range(len(Hidden_dims)):
    
    data = pd.read_csv('../hnn/results/'+str(Hidden_dims[i])+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)
    
    for date, oneData in data.groupby('date')[['mean', 'std']]:
        X = sm.add_constant(oneData[['std']])
        Y = oneData['mean']
        model = sm.OLS(Y, X).fit()
        
        gamma = model.params['std'] * 100
        t_gamma = model.tvalues['std']
        r_squared = model.rsquared * 100
        results.loc[len(results)] = [str(int(date / 100)), Model_names[i], gamma, t_gamma, r_squared, len(Y)]

idx = pd.IndexSlice
results = results.set_index(['date', 'model'])

if not os.path.isdir("plots/"):
    os.mkdir("plots/")
for model in Model_names:
    for col in ['gamma', 'r_squared']:
        pdf = PdfPages("plots/{}_{}.pdf".format(model, col))
        plt.figure(figsize=(7, 3), dpi = 300)
        plt.plot(results.loc[idx[:, model], col].droplevel('model'), color = 'black')
        plt.xticks(['200101', '200401', '200701', '201001', '201301', '201601', '201901'])
        if col == 'gamma':
            plt.axhline(y = 0, color = 'black', linewidth = 0.8)
            plt.yticks([-20, -10, 0, 10])
        else:
            plt.yticks([0, 10, 20, 30, 40])
        pdf.savefig(bbox_inches = 'tight')
        plt.close()
        pdf.close()

with open("latex.txt", 'w') as file:
    for col in ['gamma', 't_gamma', 'r_squared']:
        for model in Model_names:
            file.write(model)
            data = results.loc[idx[:, model], col].droplevel('model')
            if col == 't_gamma':
                data = data[data < 0]
            numbers = [data.mean(), data.std(), data.min(), np.percentile(data, 25), \
                       data.median(), np.percentile(data, 75), data.max()]
            for number in numbers:
                file.write(f' & {number:.2f}')
            file.write(' \\\\\n')
        file.write('\n')



