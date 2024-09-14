# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#==============================================================================

category = ['Accruals', 'Debt Issuance', 'Investment', 'Low Leverage', 'Low Risk', 'Momentum',
           'Profit Growth', 'Profitability', 'Quality', 'Seasonality', 'Size', 'Short-Term Reversal',
           'Value']

char_category = [["cowc_gr1a","oaccruals_at","oaccruals_ni","seas_16_20na","taccruals_at","taccruals_ni"],
                ["capex_abn","debt_gr3","fnl_gr1a","ncol_gr1a","nfna_gr1a","ni_ar1","noa_at"],
                ["aliq_at","at_gr1","be_gr1a","capx_gr1","capx_gr2","capx_gr3","coa_gr1a","col_gr1a",
                "emp_gr1","inv_gr1","inv_gr1a","lnoa_gr1a","mispricing_mgmt","ncoa_gr1a","nncoa_gr1a",
                "noa_gr1a","ppeinv_gr1a","ret_60_12","sale_gr1","sale_gr3","saleq_gr1","seas_2_5na"],
                ["age","aliq_mat","at_be","bidaskhl_21d","cash_at","netdebt_me","ni_ivol","rd_sale",
                "rd5_at","tangibility","z_score"],
                ["beta_60m","beta_dimson_21d","betabab_1260d","betadown_252d","earnings_variability",
                "ivol_capm_21d","ivol_capm_252d","ivol_ff3_21d","ivol_hxz4_21d","ocfq_saleq_std",
                "rmax1_21d","rmax5_21d","rvol_21d","seas_6_10na","turnover_126d",
                "zero_trades_21d","zero_trades_126d","zero_trades_252d"],
                ["prc_highprc_252d","resff3_6_1","resff3_12_1","ret_3_1","ret_6_1","ret_9_1",
                "ret_12_1","seas_1_1na"],
                ["dsale_dinv","dsale_drec","dsale_dsga","niq_at_chg1","niq_be_chg1","niq_su",
                "ocf_at_chg1","ret_12_7","sale_emp_gr1","saleq_su","seas_1_1an","tax_gr1a"],
                ["dolvol_var_126d","ebit_bev","ebit_sale","f_score","ni_be","niq_be","o_score",
                "ocf_at","ope_be","ope_bel1","turnover_var_126d"],
                ["at_turnover","cop_at","cop_atl1","dgp_dsale","gp_at","gp_atl1","mispricing_perf",
                "ni_inc8q","niq_at","op_at","op_atl1","opex_at","qmj","qmj_growth","qmj_prof",
                "qmj_safety","sale_bev"],
                ["corr_1260d","coskew_21d","dbnetis_at","kz_index","lti_gr1a","pi_nix","seas_2_5an",
                "seas_6_10an","seas_11_15an","seas_11_15na","seas_16_20an","sti_gr1a"],
                ["ami_126d","dolvol_126d","market_equity","prc","rd_me"],
                ["iskew_capm_21d","iskew_ff3_21d","iskew_hxz4_21d","ret_1_0","rmax5_rvol_21d","rskew_21d"],
                ["at_me","be_me","bev_mev","chcsho_12m","debt_me","div12m_me","ebitda_mev","eq_dur",
                "eqnetis_at","eqnpo_12m","eqnpo_me","eqpo_me","fcf_me","ival_me","netis_at","ni_me","ocf_me",
                 "sale_me"]]

#==============================================================================

var_imp = pd.read_csv('var_imp.csv')
Hidden_dims = [[32], [32,16], [32,16,8], [32,16,8,4], [32,16,8,4,2]]
Network_name = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']

if not os.path.isdir('plots/'):
    os.mkdir('plots/')

var_imp['average'] = var_imp.drop('char', axis = 1).mean(axis = 1)
var_imp_sorted = var_imp.sort_values(['average', 'char'], ascending = [True, False]).iloc[-20:]

pdf = PdfPages('plots/var_importance.pdf')
var_imp_sorted.plot(kind='barh', x='char', y='average', figsize = (8, 6), width = 0.8)
for x in plt.xticks()[0]:
    plt.axvline(x, color='gray', linestyle='--', linewidth=1)
plt.ylabel('')
plt.legend().set_visible(False)
pdf.savefig(bbox_inches = 'tight')
plt.close()
pdf.close()

#==============================================================================

importance_all = pd.DataFrame([])
for i, hidden_dims in enumerate(Hidden_dims):
    importance_all[str(hidden_dims)] = var_imp[str(hidden_dims)].rank(pct = False)

importance_all.index = var_imp['char']
importance_all['total_rank'] = importance_all.sum(axis = 1)
importance_all = importance_all.sort_values(by = ['total_rank'], ascending = True)
importance_all = importance_all.drop(['total_rank'], axis = 1)

pdf = PdfPages('plots/var_importance_all.pdf')
plt.figure(figsize = (9, 30), dpi = 500)
plt.pcolor(importance_all, cmap= 'Blues', edgecolor = 'w')
plt.yticks(np.arange(0.5, len(importance_all.index), 1), importance_all.index)
plt.xticks(np.arange(0.5, len(importance_all.columns), 1), Network_name, fontsize = 15)
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()

#==============================================================================

conditions = [var_imp['char'].isin(char_cate) for char_cate in char_category]
var_imp['category'] = np.select(conditions, category, default = np.nan)
importance_cate = var_imp.groupby(['category'])[[str(x) for x in Hidden_dims]].mean()
importance_cate = importance_cate.rank(pct = False)
importance_cate['total_rank'] = importance_cate.sum(axis = 1)
importance_cate = importance_cate.sort_values(by = ['total_rank'], ascending = True)
importance_cate = importance_cate.drop(['total_rank'], axis = 1)

pdf = PdfPages('plots/var_importance_category.pdf')
plt.figure(figsize=(7, 6), dpi = 500)
plt.pcolor(importance_cate, cmap= 'Blues', edgecolor = 'w')
plt.yticks(np.arange(0.5, len(importance_cate.index), 1), importance_cate.index)
plt.xticks(np.arange(0.5, len(importance_cate.columns), 1), Network_name, fontsize = 12)
pdf.savefig(bbox_inches='tight')
plt.close()
pdf.close()