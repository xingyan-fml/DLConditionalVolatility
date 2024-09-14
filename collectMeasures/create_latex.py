# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#==============================================================================

measures = pd.read_csv("measures.txt").set_index(['model_name', 'sorting', 'weight'])
risk_adjusted = pd.read_csv("risk_adjusted.txt").set_index(['model_name', 'sorting', 'weight'])

with open("create_latex.txt", 'w') as file:
    for model_name in ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']:
        for sorting in ['univariate', 'independent', 'dependent']:
            sorting2 = 'sin' if sorting == 'univariate' else sorting[:3]
            file.write(f" & \\makecell{{{model_name}\\\\{sorting2}}}")
    file.write(" \\\\\n")
    file.write("\\midrule\n")
    
    file.write('\\multicolumn{9}{l}{Drawdowns and turnover (equal-weighted)} \\\\\n')
    for col in ['MDD', 'M1ML', 'turnover']:
        col_text = {'MDD':'Max DD (\\%)', 'M1ML':'Max 1M loss (\\%)', 'turnover':'Turnover (\\%)'}
        file.write(col_text[col])
        for model_name in ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']:
            for sorting in ['univariate', 'independent', 'dependent']:
                number = measures.loc[(model_name, sorting, 'ew'), col]
                file.write(f' & {number:.2f}')
        file.write(' \\\\\n')
    file.write(' &  &  &  &  &  &  &  &  & \\\\\n')
    
    file.write('\\multicolumn{9}{l}{Drawdowns and turnover (value-weighted)} \\\\\n')
    for col in ['MDD', 'M1ML', 'turnover']:
        col_text = {'MDD':'Max DD (\\%)', 'M1ML':'Max 1M loss (\\%)', 'turnover':'Turnover (\\%)'}
        file.write(col_text[col])
        for model_name in ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']:
            for sorting in ['univariate', 'independent', 'dependent']:
                number = measures.loc[(model_name, sorting, 'vw'), col]
                file.write(f' & {number:.2f}')
        file.write(' \\\\\n')
    file.write(' &  &  &  &  &  &  &  &  & \\\\\n')
    
    file.write('\\multicolumn{9}{l}{Risk-adjusted performance (equal-weighted)} \\\\\n')
    for col in ['mean_ret', 'alpha', 't_alpha', 'r_squared', 'ir']:
        col_text = {'mean_ret':'Mean return', 'alpha':'FF5+Mom $\\alpha$', \
                    't_alpha':'$t(\\alpha$)', 'r_squared':'$R^2$', 'ir':'IR'}
        file.write(col_text[col])
        for model_name in ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']:
            for sorting in ['univariate', 'independent', 'dependent']:
                number = risk_adjusted.loc[(model_name, sorting, 'ew'), col]
                file.write(f' & {number:.2f}')
        file.write(' \\\\\n')
    file.write(' &  &  &  &  &  &  &  &  & \\\\\n')
    
    file.write('\\multicolumn{9}{l}{Risk-adjusted performance (value-weighted)} \\\\\n')
    for col in ['mean_ret', 'alpha', 't_alpha', 'r_squared', 'ir']:
        col_text = {'mean_ret':'Mean return', 'alpha':'FF5+Mom $\\alpha$', \
                    't_alpha':'$t(\\alpha$)', 'r_squared':'$R^2$', 'ir':'IR'}
        file.write(col_text[col])
        for model_name in ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']:
            for sorting in ['univariate', 'independent', 'dependent']:
                number = risk_adjusted.loc[(model_name, sorting, 'vw'), col]
                file.write(f' & {number:.2f}')
        file.write(' \\\\\n')