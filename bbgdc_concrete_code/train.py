from __future__ import print_function
from __future__ import division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# import libraries
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
import torch
import math
import torch.nn.functional as F
import time
from model import BBDGCN
from utils import load_data, accuracy_mrun_np, normalize_torch


# torch.cuda.device_count()

seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Load data

adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')


# hyper-parameters

nfeat = features.shape[1]
nclass = labels.max().item() + 1
nfeat_list = [nfeat, 128, 128, 128, nclass]
nlay = 4
nblock = 2
# num_edges = int(adj.nnz/2)
num_edges = int(adj.shape[0] ** 2)
dropout = 0
lr = 0.005
weight_decay = 5e-3
mul_type='norm_first'


# defining model

model = BBDGCN(nfeat_list=nfeat_list
               , dropout=dropout
               , nblock=nblock
               , nlay=nlay
               , num_edges=num_edges)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Model Summary:")
print(model)
print('----------------------')


adj = torch.FloatTensor(adj.todense())
adj_normt = normalize_torch(adj + torch.eye(adj.shape[0]))

if torch.cuda.is_available():
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    adj = adj.cuda()
    adj_normt = adj_normt.cuda()


labels_np = labels.cpu().numpy().astype(np.int32)
idx_train_np = idx_train.cpu().numpy()
idx_val_np = idx_val.cpu().numpy()
idx_test_np = idx_test.cpu().numpy()



# training

nepochs = 2000
num_run = 20

for i in range(nepochs):
    # t = time.time()
    optimizer.zero_grad()
    wup = np.min([1.0, (i+1)/20])
    # wup = 1.0
    output, tot_loss, nll_loss, kld_loss, drop_rates = model(x=features
                                                             , labels=labels
                                                             , adj=adj
                                                             , obs_idx=idx_train
                                                             , warm_up=wup
                                                             , adj_normt=adj_normt
                                                             , training=True
                                                             , mul_type=mul_type)
    
    l2_reg = None
    block_index = 0
    for layer in range(nlay):
        l2_reg_lay = None
        if layer==0:
            for param in model.gcs[str(block_index)].parameters():
                if l2_reg_lay is None:
                    l2_reg_lay = (param**2).sum()
                else:
                    l2_reg_lay = l2_reg_lay + (param**2).sum()
            block_index += 1
            
        else:
            for iii in range(nblock):
                for param in model.gcs[str(block_index)].parameters():
                    if l2_reg_lay is None:
                        l2_reg_lay = (param**2).sum()
                    else:
                        l2_reg_lay = l2_reg_lay + (param**2).sum()
                block_index += 1
                
        l2_reg_lay = (1-drop_rates[layer])*l2_reg_lay
        
        if l2_reg is None:
            l2_reg = l2_reg_lay
        else:
            l2_reg += l2_reg_lay
    
    main_loss = tot_loss + weight_decay * l2_reg
    main_loss.backward()
    optimizer.step()
    
    outs = [None]*num_run
    for j in range(num_run):
        outstmp, _, _, _,_ = model(x=features
                                   , labels=labels
                                   , adj=adj
                                   , obs_idx=idx_train
                                   , warm_up=wup
                                   , adj_normt=adj_normt
                                   , training=False
                                   , samp_type='rel_ber'
                                   , mul_type=mul_type)
        
        outs[j] = outstmp.cpu().data.numpy()
    
    outs = np.stack(outs)
    acc_val_tr = accuracy_mrun_np(outs, labels_np, idx_val_np)
    acc_test_tr = accuracy_mrun_np(outs, labels_np, idx_test_np)
    
    print('Epoch: {:04d}'.format(i+1)
          , 'nll: {:.4f}'.format(nll_loss.item())
          , 'kld: {:.4f}'.format(kld_loss.item())
          , 'acc_val: {:.4f}'.format(acc_val_tr)
          , 'acc_test: {:.4f}'.format(acc_test_tr))
    print('----------------------')
