# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from evaluation_hnn import *

#==============================================================================
# usa_new

# hidden_dims = [32,16,8,4,2], [32,16,8,4], [32,16,8], [32,16], [32]

data_name = 'usa_new'
lr = 0.001
penalty = 0.00001
num_iter = 1

info = ['id','eom','me']
y = ['ret_exc_lead1m']

for hidden_dims in [[32,16,8,4,2], [32,16,8,4], [32,16,8], [32,16], [32]]:
    pred(data_name, info, y, hidden_dims, lr, penalty, num_iter)



