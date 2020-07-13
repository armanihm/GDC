from __future__ import print_function
from __future__ import division

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# import libraries
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from utils import normalize_torch


# torch.cuda.device_count()


# seed = 5
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        torch.nn.init.normal_(tensor=self.weight, std=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, inp, adj):
        support = torch.mm(inp, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class BBDropedge(nn.Module):
    def __init__(self, num_pars, alpha=0.8, kl_scale=1.0):
        super(BBDropedge, self).__init__()
        self.num_pars = num_pars
        self.alpha = alpha
        self.kl_scale = kl_scale
        self.a_uc = nn.Parameter(torch.FloatTensor(self.num_pars))
        self.b_uc = nn.Parameter(torch.FloatTensor(self.num_pars))
        self.a_uc.data.uniform_(1.0, 1.5)
        self.b_uc.data.uniform_(0.49, 0.51)
    
    def get_params(self):
        a = F.softplus(self.a_uc.clamp(min=-10.))
        b = F.softplus(self.b_uc.clamp(min=-10., max=50.))
        return a, b
    
    def sample_pi(self):
        a, b = self.get_params()
        u = torch.rand(self.num_pars).clamp(1e-6, 1-1e-6)
        if torch.cuda.is_available():
            u = u.cuda()
        return (1 - u.pow_(1./b)).pow_(1./a)
    
    def get_weight(self, num_samps, training, samp_type='rel_ber'):
        temp = torch.Tensor([0.67])
        if torch.cuda.is_available():
            temp = temp.cuda()
        
        if training:
            pi = self.sample_pi()
            p_z = RelaxedBernoulli(temp, probs=pi)
            z = p_z.rsample(torch.Size([num_samps]))
        else:
            if samp_type=='rel_ber':
                pi = self.sample_pi()
                p_z = RelaxedBernoulli(temp, probs=pi)
                z = p_z.rsample(torch.Size([num_samps]))
            elif samp_type=='ber':
                pi = self.sample_pi()
                p_z = torch.distributions.Bernoulli(probs=pi)            
                z = p_z.sample(torch.Size([num_samps]))
        return z, pi
    
    def get_reg(self):
        a, b = self.get_params()
        kld = (1 - self.alpha/a)*(-0.577215664901532 - torch.digamma(b) - 1./b) + torch.log(a*b + 1e-10) - math.log(self.alpha) - (b-1)/b
        kld = (self.kl_scale) * kld.sum()
        return kld



class BBDGCN(nn.Module):
    def __init__(self, nfeat_list, dropout, nblock, nlay, num_edges):
        super(BBDGCN, self).__init__()
        
        assert len(nfeat_list)==nlay+1
        self.nlay = nlay
        self.nblock = nblock
        self.num_edges = num_edges
        self.num_nodes = int(np.sqrt(num_edges))
        self.drpedg_list = []
        self.dropout = dropout
        gcs_list = []
        idx = 0
        for i in range(nlay):
            if i==0:
                self.drpedg_list.append(BBDropedge(1))
                gcs_list.append([str(idx), GraphConvolution(nfeat_list[i], nfeat_list[i+1])])
                idx += 1
            else:
                self.drpedg_list.append(BBDropedge(1))
                for j in range(self.nblock):
                    gcs_list.append([str(idx), GraphConvolution(int(nfeat_list[i]/self.nblock), nfeat_list[i+1])])
                    idx += 1
        
        self.drpedgs = nn.ModuleList(self.drpedg_list)
        self.gcs = nn.ModuleDict(gcs_list)
        self.nfeat_list = nfeat_list
    
    def forward(self, x, labels, adj, obs_idx, warm_up, adj_normt, training=True
                , mul_type='norm_first', samp_type='rel_ber'):
        h_perv = x
        kld_loss = 0.0
        drop_rates = []
        for i in range(self.nlay):
            mask_vec, drop_prob = self.drpedgs[i].get_weight(self.nblock*self.num_edges, training, samp_type)
            mask_vec = torch.squeeze(mask_vec)
            drop_rates.append(drop_prob)
            if i==0:
                mask_mat = torch.reshape(mask_vec[:self.num_edges], (self.num_nodes, self.num_nodes)).cuda()
                
                if mul_type=='norm_sec':
                    adj_lay = normalize_torch(torch.mul(mask_mat, adj) + torch.eye(adj.shape[0]).cuda())
                elif mul_type=='norm_first':
                    adj_lay = torch.mul(mask_mat, adj_normt).cuda()
                
                x = F.relu(self.gcs[str(i)](x, adj_lay))
                x = F.dropout(x, self.dropout, training=training)
            
            else:
                feat_pblock = int(self.nfeat_list[i]/self.nblock)
                for j in range(self.nblock):
                    mask_mat = torch.reshape(mask_vec[j*self.num_edges:(j+1)*self.num_edges]
                                             , (self.num_nodes, self.num_nodes)).cuda()
                    
                    if mul_type=='norm_sec':
                        adj_lay = normalize_torch(torch.mul(mask_mat, adj) + torch.eye(adj.shape[0]).cuda())
                    elif mul_type=='norm_first':
                        adj_lay = torch.mul(mask_mat, adj_normt).cuda()
                    
                    if i<(self.nlay-1):
                        if j==0:
                            x_out = self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                        else:
                            x_out = x_out + self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                    else:
                        if j==0:
                            out = self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                        else:
                            out = out + self.gcs[str((i-1)*self.nblock+j+1)](x[:,j*feat_pblock:(j+1)*feat_pblock], adj_lay)
                
                if i<(self.nlay-1):
                    x = x_out
                    x = F.dropout(F.relu(x), self.dropout, training=training)
            
            
            kld_loss += self.drpedgs[i].get_reg()
            
        
        output = F.log_softmax(out, dim=1)
        
        nll_loss = self.loss(labels, output, obs_idx)
        tot_loss = nll_loss + warm_up * kld_loss
        drop_rates = torch.stack(drop_rates)
        return output, tot_loss, nll_loss, kld_loss, drop_rates
    
    def loss(self, labels, preds, obs_idx):
        return F.nll_loss(preds[obs_idx], labels[obs_idx])
