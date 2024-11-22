# -*- coding: utf-8 -*-
"""
Authors: Wenxuan Ma & Xing Yan @ RUC
mawenxuan@ruc.edu.cn
xingyan@ruc.edu.cn
"""

import copy
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#==============================================================================

class model_nn(nn.Module):
    def __init__(self, in_dim, hiddens):
        super(model_nn, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [1]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.activation = nn.ReLU()
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        X = self.linears[-1](X)
        return X

class variance_nn(nn.Module):
    def __init__(self, in_dim, hiddens):
        super(variance_nn, self).__init__()
        
        self.dims = [in_dim] + list(hiddens) + [1]
        
        self.linears = nn.ModuleList()
        for i in range(1, len(self.dims)):
            self.linears.append(nn.Linear(self.dims[i-1], self.dims[i]))
        
        self.activation = nn.ReLU()
        self.positive = nn.Softplus()
    
    def forward(self, X):
        for i in range(len(self.linears) - 1):
            X = self.activation(self.linears[i](X))
        X = self.linears[-1](X)
        X = self.positive(X) + 1e-8
        return X

#==============================================================================

# Negative log-likelihood loss function
def NLLloss(y, mean, var):
    return (torch.log(var) + (y - mean).pow(2) / var).mean()

def MSELoss(input, target, weight):
    return (weight * (input - target) ** 2).mean()

def initialize(model, weight_seed):
    torch.manual_seed(weight_seed)
    for linear in model.linears:
        nn.init.xavier_normal_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

#==============================================================================

def fit_weight_nn(train_data, valid_data, y, v_hat_train, v_hat_valid, hidden_dims, num_epochs, batch_size, lr, set_seed, penalty):
    model = model_nn(train_data.shape[1] - 1, hidden_dims)
    model.to(device)
    
    x_train, y_train = train_data.drop(y, axis = 1), train_data[y]
    x_train = torch.tensor(x_train.values, dtype = torch.float32).cuda()
    y_train = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1).cuda()
    
    x_valid, y_valid = valid_data.drop(y, axis = 1), valid_data[y]
    x_valid = torch.tensor(x_valid.values, dtype = torch.float32).cuda()
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32).reshape(-1, 1).cuda()
            
    weight_train = 1 / torch.tensor(v_hat_train, dtype = torch.float32).reshape(-1, 1).cuda()
    weight_valid = 1 / torch.tensor(v_hat_valid, dtype = torch.float32).reshape(-1, 1).cuda()
    
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    best_valid = torch.tensor(1e8).cuda()
    epoch_valids = []
    
    random.seed(set_seed)
    indexSet = list(range(x_train.shape[0]))
    for epoch in range(num_epochs):
        batch_valids = []
        random.shuffle(indexSet)
        for batch_i in range(len(indexSet) // batch_size):
            batch_index = indexSet[batch_i * batch_size : (batch_i + 1) * batch_size]
            x_batch, y_batch = x_train[batch_index], y_train[batch_index]
            weight_batch = weight_train[batch_index]
            
            model.train()
            out = model(x_batch)

            loss = MSELoss(out, y_batch, weight_batch)
            l1_lambda = torch.tensor(penalty).cuda()
            l1_norm = sum(linear.weight.abs().sum() for linear in model.linears)
            loss_p = loss + l1_lambda * l1_norm

            optimizer.zero_grad()
            loss_p.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                valid_loss = MSELoss(model(x_valid), y_valid, weight_valid).detach()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())
        
        mean_batch_valids = torch.stack(batch_valids).mean()
        print("epoch {}:".format(epoch), mean_batch_valids)
        epoch_valids.append(mean_batch_valids)
        if len(epoch_valids) >= 25 and torch.stack(epoch_valids[-5:]).mean() > torch.stack(epoch_valids[-25:-5]).mean():
            break
    
    model.load_state_dict(best_dict)
    model.eval()
    
    Y_hat_train = model(x_train).detach().cpu().numpy().reshape(-1)
    Y_hat_valid = model(x_valid).detach().cpu().numpy().reshape(-1)
    
    return model, Y_hat_train, Y_hat_valid

#==============================================================================

def fit_variance_nn(train_data, valid_data, y, Y_hat_train, Y_hat_valid, hidden_dims, num_epochs, batch_size, lr, set_seed, penalty):
    model = variance_nn(train_data.shape[1] - 1, hidden_dims)
    model.to(device)
    
    x_train, y_train = train_data.drop(y, axis = 1), train_data[y]
    x_train = torch.tensor(x_train.values, dtype = torch.float32).cuda()
    y_train = torch.tensor(y_train.values, dtype = torch.float32).reshape(-1, 1).cuda()
    
    x_valid, y_valid = valid_data.drop(y, axis = 1), valid_data[y]
    x_valid = torch.tensor(x_valid.values, dtype = torch.float32).cuda()
    y_valid = torch.tensor(y_valid.values, dtype = torch.float32).reshape(-1, 1).cuda()
    
    y_hat_train = torch.tensor(Y_hat_train, dtype = torch.float32).reshape(-1, 1).cuda()
    y_hat_valid = torch.tensor(Y_hat_valid, dtype = torch.float32).reshape(-1, 1).cuda()
    
    initialize(model, set_seed)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    best_valid = torch.tensor(1e8).cuda()
    epoch_valids = []
    
    random.seed(set_seed)
    indexSet = list(range(x_train.shape[0]))
    for epoch in range(num_epochs):
        batch_valids = []
        random.shuffle(indexSet)
        for batch_i in range(len(indexSet) // batch_size):
            batch_index = indexSet[batch_i * batch_size : (batch_i + 1) * batch_size]
            x_batch, y_batch = x_train[batch_index], y_train[batch_index]
            y_hat_batch = y_hat_train[batch_index]
            
            model.train()
            out = model(x_batch)
            
            loss = NLLloss(y_batch, y_hat_batch, out)
            l1_lambda = torch.tensor(penalty).cuda()
            l1_norm = sum(linear.weight.abs().sum() for linear in model.linears)
            loss_p = loss + l1_lambda * l1_norm

            optimizer.zero_grad()
            loss_p.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                valid_loss = NLLloss(y_valid, y_hat_valid, model(x_valid)).detach()
            batch_valids.append(valid_loss)
            if valid_loss < best_valid:
                best_valid = valid_loss
                best_dict = copy.deepcopy(model.state_dict())
        
        mean_batch_valids = torch.stack(batch_valids).mean()
        print("epoch {}:".format(epoch), mean_batch_valids)
        epoch_valids.append(mean_batch_valids)
        if len(epoch_valids) >= 25 and torch.stack(epoch_valids[-5:]).mean() > torch.stack(epoch_valids[-25:-5]).mean():
            break
    
    model.load_state_dict(best_dict)
    model.eval()

    var_hat_train = model(x_train).detach().cpu().numpy().reshape(-1)
    var_hat_valid = model(x_valid).detach().cpu().numpy().reshape(-1)
    
    return model, var_hat_train, var_hat_valid

#==============================================================================

def fit_model_hnn(train_data, valid_data, y, num_iter, hidden_dims, num_epochs, batch_size, lr, set_seed, penalty):
    v_hat_train = np.ones(len(train_data))
    v_hat_valid = np.ones(len(valid_data))
    
    for n in range(num_iter):
        model_mu, Y_hat_train, Y_hat_valid = fit_weight_nn(train_data, valid_data, y, v_hat_train, v_hat_valid, hidden_dims, num_epochs, batch_size, lr, set_seed + n * 101, penalty)
        model_var, v_hat_train, v_hat_valid = fit_variance_nn(train_data, valid_data, y, Y_hat_train, Y_hat_valid, [], num_epochs, batch_size, lr, set_seed + n * 101 + 2, penalty)
    
    model_mu.to('cpu')
    model_var.to('cpu')

    return model_mu, model_var
