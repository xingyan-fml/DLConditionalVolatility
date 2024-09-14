# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fit_hnn import *

#==============================================================================

cols = ["age",                 "aliq_at",             "aliq_mat",            "ami_126d",           
  "at_be",               "at_gr1",              "at_me",               "at_turnover",        
  "be_gr1a",             "be_me",               "beta_60m",            "beta_dimson_21d",    
  "betabab_1260d",       "betadown_252d",       "bev_mev",             "bidaskhl_21d",       
  "capex_abn",           "capx_gr1",            "capx_gr2",            "capx_gr3",           
  "cash_at",             "chcsho_12m",          "coa_gr1a",            "col_gr1a",           
  "cop_at",              "cop_atl1",            "corr_1260d",          "coskew_21d",         
  "cowc_gr1a",           "dbnetis_at",          "debt_gr3",            "debt_me",            
  "dgp_dsale",           "div12m_me",           "dolvol_126d",         "dolvol_var_126d",    
  "dsale_dinv",          "dsale_drec",          "dsale_dsga",          "earnings_variability",
  "ebit_bev",            "ebit_sale",           "ebitda_mev",          "emp_gr1",            
  "eq_dur",              "eqnetis_at",          "eqnpo_12m",           "eqnpo_me",           
  "eqpo_me",             "f_score",             "fcf_me",              "fnl_gr1a",           
  "gp_at",               "gp_atl1",             "ival_me",             "inv_gr1",            
  "inv_gr1a",            "iskew_capm_21d",      "iskew_ff3_21d",       "iskew_hxz4_21d",     
  "ivol_capm_21d",       "ivol_capm_252d",      "ivol_ff3_21d",        "ivol_hxz4_21d",      
  "kz_index",            "lnoa_gr1a",           "lti_gr1a",            "market_equity",      
  "mispricing_mgmt",     "mispricing_perf",     "ncoa_gr1a",           "ncol_gr1a",          
  "netdebt_me",          "netis_at",            "nfna_gr1a",           "ni_ar1",             
  "ni_be",               "ni_inc8q",            "ni_ivol",             "ni_me",              
  "niq_at",              "niq_at_chg1",         "niq_be",              "niq_be_chg1",        
  "niq_su",              "nncoa_gr1a",          "noa_at",              "noa_gr1a",           
  "o_score",             "oaccruals_at",        "oaccruals_ni",        "ocf_at",             
  "ocf_at_chg1",         "ocf_me",              "ocfq_saleq_std",      "op_at",              
  "op_atl1",             "ope_be",              "ope_bel1",            "opex_at",            
  "pi_nix",              "ppeinv_gr1a",         "prc",                 "prc_highprc_252d",   
  "qmj",                 "qmj_growth",          "qmj_prof",            "qmj_safety",         
  "rd_me",               "rd_sale",             "rd5_at",              "resff3_12_1",        
  "resff3_6_1",          "ret_1_0",             "ret_12_1",            "ret_12_7",           
  "ret_3_1",             "ret_6_1",             "ret_60_12",           "ret_9_1",            
  "rmax1_21d",           "rmax5_21d",           "rmax5_rvol_21d",      "rskew_21d",          
  "rvol_21d",            "sale_bev",            "sale_emp_gr1",        "sale_gr1",           
  "sale_gr3",            "sale_me",             "saleq_gr1",           "saleq_su",           
  "seas_1_1an",          "seas_1_1na",          "seas_11_15an",        "seas_11_15na",       
  "seas_16_20an",        "seas_16_20na",        "seas_2_5an",          "seas_2_5na",         
  "seas_6_10an",         "seas_6_10na",         "sti_gr1a",            "taccruals_at",       
  "taccruals_ni",        "tangibility",         "tax_gr1a",            "turnover_126d",      
  "turnover_var_126d",   "z_score",             "zero_trades_126d",    "zero_trades_21d",    
  "zero_trades_252d"]

if not os.path.isdir('plots/'):
    os.mkdir('plots/')

main_char = 'market_equity'
Marginal_chars = ['zero_trades_21d', 'ret_1_0', 'turnover_126d', 'prc_highprc_252d']

Hidden_dims = [[32], [32, 16], [32, 16, 8], [32, 16, 8, 4], [32, 16, 8, 4, 2]]
Network_names = ['NN1', 'NN2', 'NN3', 'NN4', 'NN5']

N = 1000
nn_Num = 10

for marginal_char in Marginal_chars:
    data = pd.DataFrame(np.zeros((N, len(cols))), columns = cols)
    np.random.seed(42)
    data[marginal_char] = np.random.uniform(-1, 1, size = N)
    data[main_char] = np.repeat([-1, -0.5, 0, 0.5, 1], N / 5)
    data_x = torch.tensor(data.values, dtype = torch.float32)
    
    results = pd.DataFrame([])
    results[main_char] = data[main_char]
    results[marginal_char] = data[marginal_char]
    
    hidden_dims = [32, 16, 8]
    pred = np.zeros((len(data), 20))
    for i in range(20):
        model_folder = "../hnn/results/"+str(2001+i)+"/"
        model_list = []
        for n in range(nn_Num):
            model_temp  = model_nn(data.shape[1], hidden_dims)
            model_temp.load_state_dict(torch.load(f'{model_folder}/model_mu_{hidden_dims}_{n}.pth'))
            model_list.append(model_temp)
        
        with torch.no_grad():
            mu_ensem = np.zeros((len(data), nn_Num))
            for n in range(nn_Num):
                mu_ensem[:, n] = model_list[n](data_x).detach().numpy().reshape(-1)
        pred[:, i] = mu_ensem.mean(axis = 1)
    
    results[str(hidden_dims)] = pred.mean(axis = 1)
    results = results.sort_values(by=marginal_char).reset_index(drop=True)
    
    pdf = PdfPages(f'plots/{marginal_char}_mean.pdf')
    plt.figure(figsize=(7,5),dpi=300)
    for i in [-1, -0.5, 0, 0.5, 1]:
        results_temp = results[results[main_char] == i]
        plt.plot(results_temp[marginal_char], results_temp[str(hidden_dims)] * 100, label = f'{main_char} = ${i}$')
    plt.legend(loc='upper right')
    plt.xlim(-1, 1)
    for x in plt.xticks()[0]:
        plt.axvline(x, color='gray', linestyle='--', linewidth=1)
    pdf.savefig(bbox_inches = 'tight')
    plt.close()
    pdf.close()