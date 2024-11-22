# -*- coding: utf-8 -*-

import pandas as pd

#==============================================================================
# Generate Latex Code

models = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']
vols = ['ours', 'ivol', 'ivol_ff3', 'rvol', 'garch', 'gjr-garch', 'linear']

results = {'ew': {}, 'vw': {}}
with open("create_latex.txt", 'w') as file:
    for weight in ['ew', 'vw']:
        for vol in vols:
            if vol == 'ours':
                results[weight][vol] = pd.read_csv("../collectPortfolios/portfolio_" + weight + ".txt").set_index(['model_name', 'sorting', 'group'])
            else:
                results[weight][vol] = pd.read_csv(vol + "_portfolio_" + weight + ".txt").set_index(['model_name', 'sorting', 'group'])
        
        for double in ['independent', 'dependent']:
            if weight == 'ew':
                file.write("& Single & \\multicolumn{{7}}{{c}}{{Equal-weighted {} double sorting}} \\\\\n".format(double))
            else:
                file.write("& Single & \\multicolumn{{7}}{{c}}{{Value-weighted {} double sorting}} \\\\\n".format(double))
            
            numbers = pd.DataFrame(None, index = models, columns = ['single'] + vols)
            for model in models:
                for vol in numbers.columns:
                    if vol == 'single':
                        numbers.loc[model, vol] = results[weight]['ours'].loc[(model, 'univariate', 'H-L'), 'SR']
                    else:
                        numbers.loc[model, vol] = results[weight][vol].loc[(model, double, 'H-L'), 'SR']
            numbers.loc['Average'] = numbers.mean()
            
            file.write("& sorting & NN$_k$ & ivol\\_capm\\_252d & ivol\\_ff3\\_21d & rvol\\_21d & GARCH & GJR-GARCH & Linear \\\\\n")
            file.write("\\midrule\n")
            for model in numbers.index:
                file.write(model)
                for num in numbers.loc[model]:
                    if num > numbers.loc[model].max() - 0.02:
                        file.write(' & {:.2f}$^{{**}}$'.format(num))
                    else:
                        file.write(' & {:.2f}'.format(num))
                file.write(' \\\\\n')
            # file.write("&  &  &  &  &  &  &  &  \\\\\n\n")
            file.write("\\midrule\n")