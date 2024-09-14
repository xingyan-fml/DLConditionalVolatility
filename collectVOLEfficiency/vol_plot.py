# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

#==============================================================================

year1 = range(2001, 2011)
year2 = range(2011, 2021)
numGroups = 10
Hidden_dims = [[32], [32, 16], [32, 16, 8], [32, 16, 8, 4], [32, 16, 8, 4, 2]]
colors = ['b','g','r','c','m','y','k','gray','pink','lime','lightblue','orange']

if not os.path.isdir("plots/"):
    os.mkdir("plots/")

for hidden_dims in Hidden_dims:
    data = pd.read_csv('../hnn/results/'+str(hidden_dims)+'_0.001_1e-05_1_10.csv')
    data['std'] = np.sqrt(data['var'])
    data = data.drop(['var', 'var2'], axis = 1)

    data['rank'] = data.groupby('date')['std'].rank(pct = True)
    data['group'] = (data['rank'] * numGroups).apply(np.ceil)

    portfolio = data.groupby(['date', 'group'])[['true']].std()
    portfolio = portfolio.rename(columns={'true':'real_std'})
    portfolio = portfolio.reset_index()
    portfolio['log_std'] = np.log(portfolio['real_std'])

    portfolio['date'] = pd.to_datetime(portfolio['date'], format='%Y%m%d')
    portfolio['year'] = portfolio['date'].dt.year
    portfolio['month'] = portfolio['date'].dt.month
    
    portfolio_year1 = portfolio[portfolio['year'].isin(year1)]
    portfolio_year2 = portfolio[portfolio['year'].isin(year2)]
    
    #plot
    pdf = PdfPages("plots/{}_{}.pdf".format(str(hidden_dims), str(1)))
    plt.figure(figsize=(20,8), dpi=300)
    
    i=0
    for g in portfolio_year1['group'].unique():
        temp = portfolio_year1[portfolio_year1['group']==g]
        plt.plot(temp['log_std'], marker='o', color=colors[i], label=int(g), linewidth=0.75)
        i = i+1
    
    log_ticks = np.array([-3, -2.5, -2, -1.5, -1, -0.5])
    real_ticks = np.round(np.exp(log_ticks), 2)
    plt.yticks(log_ticks, real_ticks, fontsize=15)
    
    month_ticks = np.arange(0, 120, 12)*10
    plt.xticks(month_ticks, year1, fontsize=15)
    
    plt.legend(bbox_to_anchor=(1.075, 1), loc='upper right', fontsize=15)
    #plt.grid(True)
    pdf.savefig(bbox_inches='tight')
    plt.close()
    pdf.close()
    
    #plot
    pdf = PdfPages("plots/{}_{}.pdf".format(str(hidden_dims), str(2)))
    plt.figure(figsize=(20,8), dpi=300)
    
    i=0
    for g in portfolio_year2['group'].unique():
        temp = portfolio_year2[portfolio_year2['group']==g]
        plt.plot(temp['log_std'], marker='o', color=colors[i], label=int(g), linewidth=0.75)
        i = i+1
    
    log_ticks = np.array([-3, -2.5, -2, -1.5, -1, -0.5])
    real_ticks = np.round(np.exp(log_ticks), 2)
    plt.yticks(log_ticks, real_ticks, fontsize=15)
    
    month_ticks = np.arange(0, 120, 12)*10 + 1200
    plt.xticks(month_ticks, year2, fontsize=15)
    
    plt.legend(bbox_to_anchor=(1.075, 1), loc='upper right', fontsize=15)
    #plt.grid(True)
    pdf.savefig(bbox_inches='tight')
    plt.close()
    pdf.close()