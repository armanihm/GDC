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
from model import BBGDCGCN
from utils import load_data, accuracy_mrun_np, normalize_torch


# torch.cuda.device_count()

seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


# Load data

adj, features, labels, idx_train, idx_val, idx_test = load_data('citeseer')
adj=adj-sp.diags((adj.diagonal()>0).astype('int'))

# hyper-parameters

nfeat = features.shape[1]
nclass = labels.max().item() + 1
nfeat_list = [nfeat, 128, 128, 128, nclass]
nlay = 4
nblock = 2
num_nodes = int(adj.shape[0])
dropout = 0
lr = 0.005 #0.01
weight_decay = 5e-3
mul_type='norm_first'
con_type='reg'
sym=True

adj = adj + sp.eye(adj.shape[0])
if sym:
    num_edges = int((adj.count_nonzero()+adj.shape[0])/2.0)
else:
    num_edges = adj.count_nonzero()

# finding NZ indecies
if sym:
  adju = sp.triu(adj)
  nz_idx_tupl = adju.nonzero()
  nz_idx_list = []
  for i in range(len(nz_idx_tupl[0])):
    nz_idx_list.append([nz_idx_tupl[0][i], nz_idx_tupl[1][i]])

  nz_idx = torch.LongTensor(nz_idx_list).T
else:
  nz_idx_tupl = adj.nonzero()
  nz_idx_list = []
  for i in range(len(nz_idx_tupl[0])):
    nz_idx_list.append([nz_idx_tupl[0][i], nz_idx_tupl[1][i]])

  nz_idx = torch.LongTensor(nz_idx_list).T


# defining model

model = BBGDCGCN(nfeat_list=nfeat_list
                 , dropout=dropout
                 , nblock=nblock
                 , adj=adj
                 , nlay=nlay
                 , num_edges=num_edges
                 , num_nodes=num_nodes
                 , sym=sym)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Model Summary:")
print(model)
print('----------------------')


adj = torch.FloatTensor(adj.todense())
adj_normt = normalize_torch(adj)

if torch.cuda.is_available():
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    nz_idx = nz_idx.cuda()
    adj = adj.cuda()
    adj_normt = adj_normt.cuda()


labels_np = labels.cpu().numpy().astype(np.int32)
idx_train_np = idx_train.cpu().numpy()
idx_val_np = idx_val.cpu().numpy()
idx_test_np = idx_test.cpu().numpy()



# training

nepochs = 1700
num_run = 12

for i in range(nepochs):
    # t = time.time()
    optimizer.zero_grad()
    wup = np.min([1e-4, (i+1)/200e4])
    # wup = 1.0
    output, tot_loss, nll_loss, kld_loss, tot_loss2 = model(x=features
                                                          , labels=labels
                                                          , adj=adj
                                                          , nz_idx=nz_idx
                                                          , obs_idx=idx_train
                                                          , warm_up=wup
                                                          , adj_normt=adj_normt
                                                          , training=True
                                                          , mul_type=mul_type
                                                          , con_type=con_type
                                                          )
    
    index = 0
    l2_reg = None
    for param in model.parameters():
        if index<(2):
            index += 1
            continue
        else:
            if l2_reg is None:
                index += 1
                l2_reg = (param**2).sum()
            else:
                index += 1
                l2_reg = l2_reg + (param**2).sum()
    
    main_loss = tot_loss + weight_decay * l2_reg

    main_loss.backward()

    for layy in range(len(model.drpedgs)):
        model.drpedgs[layy].a_uc.grad=1.0*model.drpedgs[layy].get_derivative_a(torch.sum((model.drpedgs[layy].u_samp-0.5)*(tot_loss2-tot_loss),dim=0)) + 1.0*wup * model.drpedgs[layy].get_reg_deriv_a()#without mask
        model.drpedgs[layy].b_uc.grad=1.0*model.drpedgs[layy].get_derivative_b(torch.sum((model.drpedgs[layy].u_samp-0.5)*(tot_loss2-tot_loss),dim=0)) + 1.0*wup * model.drpedgs[layy].get_reg_deriv_b()#without mask
    
    optimizer.step()
    
    outs = [None]*num_run
    for j in range(num_run):
        outstmp, _, _, _, _ = model(x=features
                                      , labels=labels
                                      , adj=adj
                                      , nz_idx=nz_idx
                                      , obs_idx=idx_train
                                      , warm_up=wup
                                      , adj_normt=adj_normt
                                      , training=True
                                      , mul_type=mul_type
                                      , con_type=con_type
                                      )
        
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