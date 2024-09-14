# -*- coding: utf-8 -*-

import pandas as pd

#==============================================================================

data = pd.read_csv('usa_new.csv')

p = 0.3 
data_top = data.groupby('eom').apply(lambda x: x.nlargest(int(len(x)*p), 'me')).reset_index(drop = True)  
data_bottom = data.groupby('eom').apply(lambda x: x.nsmallest(int(len(x)*p), 'me')).reset_index(drop = True) 

data_top.to_csv('usa_new_top.csv', index = False)
data_bottom.to_csv('usa_new_bottom.csv', index = False)