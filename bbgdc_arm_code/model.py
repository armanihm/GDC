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
from torch.distributions.uniform import Uniform
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
        torch.nn.init.normal_(tensor=self.weight)
        if self.bias is not None:
            torch.nn.init.normal_(tensor=self.bias)
    
    def forward(self, inp, adj):
        support = torch.mm(inp, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class BBGDC(nn.Module):
    def __init__(self, num_pars,num_edges,alpha=0.8, a_uc_init=-1.0, thres=1e-3, kl_scale=1.0):
        super(BBGDC, self).__init__()
        self.num_pars = num_pars
        self.num_samps = num_edges
        self.alpha = alpha
        self.thres = thres
        self.kl_scale = kl_scale
        
        self.a_uc = nn.Parameter(torch.FloatTensor(self.num_pars),requires_grad=False)
        self.b_uc = nn.Parameter(torch.FloatTensor(self.num_pars),requires_grad=False)
        self.a_uc.data.uniform_(1.0, 1.5)
        self.b_uc.data.uniform_(0.49, 0.51)
        
        self.u_samp = Uniform(0.0,1.0).rsample(torch.Size([self.num_samps,self.num_pars])).cuda()
        self.mask_samp = Uniform(0.0,1.0).rsample(torch.Size([self.num_samps,self.num_pars])).cuda()
        self.u = torch.rand(self.num_pars).clamp(1e-6, 1-1e-6).cuda()
    
    def get_params(self):
        a = F.softplus(self.a_uc)
        b = F.softplus(self.b_uc)
        return a, b
    
    def sample_pi(self):
        a, b = self.get_params()
        self.u = torch.rand(self.num_pars).clamp(1e-6, 1-1e-6).cuda()

        return (1 - self.u.pow(1./b)).pow(1./a)

    def get_Epi(self):
        a, b = self.get_params()
        Epi = b*torch.exp(torch.lgamma(1+1./a) + torch.lgamma(b) - torch.lgamma(1+1./a + b))
        return Epi
    
    def get_params_kl(self):
        log_po = F.logsigmoid(self.p_post)
        log_po_ = F.logsigmoid(-self.p_post)
        log_pr = F.logsigmoid(self.p_pri)
        log_pr_ = F.logsigmoid(-self.p_pri)        
        
        return log_po,log_po_,log_pr,log_pr_
    
    def get_u_samp(self):
        return self.u_samp
    
    def get_u(self):
        return self.u
    
    def get_weight(self, training):
        if training:
            pi = self.sample_pi()
            p_z=Uniform(0.0,1.0)
            self.u_samp = p_z.rsample(torch.Size([self.num_samps,self.num_pars])).cuda()
            mask_11 = (-self.u_samp > -pi).float()
            mask_12 = (self.u_samp > 1.-pi).float()
            
            self.mask_samp=mask_11-mask_12

        else:
            Epi = self.get_Epi()
            mask = self.get_mask(Epi=Epi)
            mask_11 = Epi * mask
            mask_12 = Epi * mask
        return mask_11,mask_12
    
    def get_mask(self, Epi=None):
        Epi = self.get_Epi() if Epi is None else Epi
        return (Epi >= self.thres).float()
    
    def get_derivative(self,grad):
        a, b = self.get_params()
        da = torch.log((1 - self.u.pow(1./b))+1e-12) * (1 - self.u.pow(1./b)).pow(1./a) * (-1.) * (1./a**2)
        da = da * torch.sigmoid(self.a_uc)
        db = (1./a) * (1 - self.u.pow(1./b)).pow(1./a - 1.) * torch.log(self.u) * self.u.pow(1./b) * (1./b**2)
        db = db * torch.sigmoid(self.b_uc)
        derivs = torch.stack([da,db]).squeeze_()

        return grad*derivs
    
    def get_derivative_a(self,grad):
        a, b = self.get_params()
        da = torch.log((1 - self.u.pow(1./b))+1e-12) * (1 - self.u.pow(1./b)).pow(1./a) * (-1.) * (1./a**2)
        da = da * torch.sigmoid(self.a_uc)
        return grad*da
    
    def get_derivative_b(self,grad):
        a, b = self.get_params()
        db = (1./a) * (1 - self.u.pow(1./b)).pow(1./a - 1.) * torch.log(self.u) * self.u.pow(1./b) * (1./b**2)
        db = db * torch.sigmoid(self.b_uc)
        return grad*db
    
    def get_reg_deriv(self):
        a, b = self.get_params()
        dkl_a = (self.alpha/a**2)*(-0.577215664901532 - torch.digamma(b) - 1./b)  + a
        dkl_a = dkl_a * torch.sigmoid(self.a_uc)
        dkl_b = (1 - self.alpha/a) * (1./b**2 - torch.polygamma(1,b)) + b - (1./b**2)
        dkl_b = dkl_b * torch.sigmoid(self.a_uc)
        derivs_kl = torch.stack([dkl_a,dkl_b]).squeeze_()
        return derivs_kl
    
    def get_reg_deriv_a(self):
        a, b = self.get_params()
        dkl_a = (self.alpha/a**2)*(-0.577215664901532 - torch.digamma(b) - 1./b)  + a
        return dkl_a * torch.sigmoid(self.a_uc)
    
    def get_reg_deriv_b(self):
        a, b = self.get_params()
        dkl_b = (1 - self.alpha/a) * (1./b**2 - torch.polygamma(1,b)) + b - (1./b**2)
        return dkl_b * torch.sigmoid(self.b_uc)

    def get_reg(self):
        a, b = self.get_params()
        kld = (1 - self.alpha/a)*(-0.577215664901532 - torch.digamma(b) - 1./b)                 + torch.log(a*b + 1e-10) - math.log(self.alpha)                 - (b-1)/b
        kld = (self.kl_scale) * kld.sum()
        return kld




class BBGDCGCN(nn.Module):
    def __init__(self, nfeat_list, dropout, nblock, adj, nlay, num_edges,num_nodes,sym):
        super(BBGDCGCN, self).__init__()      
        assert len(nfeat_list)==nlay+1
        self.nlay = nlay
        self.sym=sym
        self.nblock = nblock
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.drpedg_list = []
        self.dropout = dropout
        gcs_list = []
        idx = 0
        for i in range(nlay):
            if i==0:
                self.drpedg_list.append(BBGDC(1,self.nblock*self.num_edges))
                gcs_list.append([str(idx), GraphConvolution(nfeat_list[i], nfeat_list[i+1])])
                idx += 1
            else:
                self.drpedg_list.append(BBGDC(1,self.nblock*self.num_edges))
                for j in range(self.nblock):
                    gcs_list.append([str(idx), GraphConvolution(int(nfeat_list[i]/self.nblock)
                                                                , nfeat_list[i+1])])
                    idx += 1
    
        self.drpedgs = nn.ModuleList(self.drpedg_list)
        self.gcs = nn.ModuleDict(gcs_list)
        self.nfeat_list = nfeat_list
    
    def forward(self, x, labels, adj, nz_idx, obs_idx, warm_up, adj_normt, training=True
                , mul_type='norm_sec', con_type='res_sum', samp_type='rel_ber'):
        h_perv1 = x
        h_perv2 = x
        x1 = x
        x2 = x
        kld_loss = 0.0
        for i in range(self.nlay):
            mask_vec1,mask_vec2 = self.drpedgs[i].get_weight(training)
            mask_vec1=torch.squeeze(mask_vec1)
            mask_vec2=torch.squeeze(mask_vec2)
            if i==0:
                
                if self.sym:
                    mask_mat1 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                    mask_mat1[nz_idx[0], nz_idx[1]] = mask_vec1[:self.num_edges]
                    mask_mat1 = (mask_mat1 + mask_mat1.T) / 2.    
                else:
                    mask_mat1 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                    mask_mat1[nz_idx[0], nz_idx[1]] = mask_vec1[:self.num_edges]
                                
                if mul_type=='norm_sec':
                    adj_lay1 = normalize_torch(torch.mul(mask_mat1, adj) + torch.eye(adj.shape[0]).cuda())
                elif mul_type=='norm_first':
                    adj_lay1 = torch.mul(mask_mat1, adj_normt).cuda()
                    
                if self.sym:
                    mask_mat2 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                    mask_mat2[nz_idx[0], nz_idx[1]] = mask_vec2[:self.num_edges]
                    mask_mat2 = (mask_mat2 + mask_mat2.T) / 2.    
                else:
                    mask_mat2 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                    mask_mat2[nz_idx[0], nz_idx[1]] = mask_vec2[:self.num_edges]
                                
                if mul_type=='norm_sec':
                    adj_lay2 = normalize_torch(torch.mul(mask_mat2, adj) + torch.eye(adj.shape[0]).cuda())
                elif mul_type=='norm_first':
                    adj_lay2 = torch.mul(mask_mat2, adj_normt).cuda()
                
                if con_type=='reg':
                    x1 = F.relu(self.gcs[str(i)](x1, adj_lay1))
                    x1 = F.dropout(x1, self.dropout, training=training)
                    x2 = F.relu(self.gcs[str(i)](x2, adj_lay2))
                    x2 = F.dropout(x2, self.dropout, training=training)
                
            else:
                feat_pblock = int(self.nfeat_list[i]/self.nblock)
                for j in range(self.nblock):
         
                    if self.sym:
                        mask_mat1 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                        mask_mat1[nz_idx[0], nz_idx[1]] = mask_vec1[j*self.num_edges:(j+1)*self.num_edges]
                        mask_mat1 = (mask_mat1 + mask_mat1.T) / 2.    
                    else:
                        mask_mat1 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                        mask_mat1[nz_idx[0], nz_idx[1]] = mask_vec1[j*self.num_edges:(j+1)*self.num_edges]
                    
                    if mul_type=='norm_sec':
                        adj_lay1 = normalize_torch(torch.mul(mask_mat1, adj) + torch.eye(adj.shape[0]).cuda())
                    elif mul_type=='norm_first':
                        adj_lay1 = torch.mul(mask_mat1, adj_normt).cuda()
                        
                    if self.sym:
                        mask_mat2 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                        mask_mat2[nz_idx[0], nz_idx[1]] = mask_vec2[j*self.num_edges:(j+1)*self.num_edges]
                        mask_mat2 = (mask_mat2 + mask_mat2.T) / 2.    
                    else:
                        mask_mat2 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                        mask_mat2[nz_idx[0], nz_idx[1]] = mask_vec2[j*self.num_edges:(j+1)*self.num_edges]
                    
                    if mul_type=='norm_sec':
                        adj_lay2 = normalize_torch(torch.mul(mask_mat2, adj) + torch.eye(adj.shape[0]).cuda())
                    elif mul_type=='norm_first':
                        adj_lay2 = torch.mul(mask_mat2, adj_normt).cuda()
                    
                    if con_type=='reg':
                        if i<(self.nlay-1):
                            if j==0:
                                x_out1 = self.gcs[str((i-1)*self.nblock+j+1)](x1[:,j*feat_pblock:(j+1)*feat_pblock]
                                                                             , adj_lay1)
                                x_out2 = self.gcs[str((i-1)*self.nblock+j+1)](x2[:,j*feat_pblock:(j+1)*feat_pblock]
                                                                             , adj_lay2)
                            else:
                                x_out1 = x_out1 + self.gcs[str((i-1)*self.nblock+j+1)](x1[:,j*feat_pblock:(j+1)*feat_pblock]
                                                                                     , adj_lay1)
                                x_out2 = x_out2 + self.gcs[str((i-1)*self.nblock+j+1)](x2[:,j*feat_pblock:(j+1)*feat_pblock]
                                                                                     , adj_lay2)
                        else:
                            if j==0:
                                out1 = self.gcs[str((i-1)*self.nblock+j+1)](x1[:,j*feat_pblock:(j+1)*feat_pblock]
                                                                           , adj_lay1)
                                out2 = self.gcs[str((i-1)*self.nblock+j+1)](x1[:,j*feat_pblock:(j+1)*feat_pblock]
                                                                           , adj_lay2)
                            else:
                                out1 = out1 + self.gcs[str((i-1)*self.nblock+j+1)](x2[:,j*feat_pblock:(j+1)*feat_pblock]
                                                                                 , adj_lay1)
                                out2 = out2 + self.gcs[str((i-1)*self.nblock+j+1)](x2[:,j*feat_pblock:(j+1)*feat_pblock]
                                                                                 , adj_lay2)
                        
                if con_type=='reg':
                    if i<(self.nlay-1):
                        x1 = x_out1
                        x1 = F.dropout(F.relu(x1), self.dropout, training=training)
                        x2 = x_out2
                        x2 = F.dropout(F.relu(x2), self.dropout, training=training)
            
            
            kld_loss += self.drpedgs[i].get_reg()
          
        output1 = F.log_softmax(out1, dim=1)
        output2 = F.log_softmax(out2, dim=1)
        
        nll_loss1 = self.loss(labels, output1, obs_idx)
        nll_loss2 = self.loss(labels, output2, obs_idx)
        tot_loss1 = nll_loss1 
        tot_loss2 = nll_loss2 
        return output1, tot_loss1, nll_loss1, kld_loss, tot_loss2
    
    def loss(self, labels, preds, obs_idx):
        return F.nll_loss(preds[obs_idx], labels[obs_idx])

class BBDE(nn.Module):
    def __init__(self, num_pars,num_edges,alpha=0.8, a_uc_init=-1.0, thres=1e-3, kl_scale=1.0):
        super(BBDE, self).__init__()
        self.num_pars = num_pars
        self.num_samps = num_edges
        self.alpha = alpha
        self.thres = thres
        self.kl_scale = kl_scale
        
        self.a_uc = nn.Parameter(torch.FloatTensor(self.num_pars),requires_grad=False)
        self.b_uc = nn.Parameter(torch.FloatTensor(self.num_pars),requires_grad=False)
        self.a_uc.data.uniform_(1.0, 1.5)
        self.b_uc.data.uniform_(0.49, 0.51)
        
        self.u_samp = Uniform(0.0,1.0).rsample(torch.Size([self.num_samps,self.num_pars])).cuda()
        self.mask_samp = Uniform(0.0,1.0).rsample(torch.Size([self.num_samps,self.num_pars])).cuda()
        self.u = torch.rand(self.num_pars).clamp(1e-6, 1-1e-6).cuda()

    
    def get_params(self):
        a = F.softplus(self.a_uc)
        b = F.softplus(self.b_uc)
        return a, b
    
    def sample_pi(self):
        a, b = self.get_params()
        self.u = torch.rand(self.num_pars).clamp(1e-6, 1-1e-6).cuda()
        return (1 - self.u.pow(1./b)).pow(1./a)

    def get_Epi(self):
        a, b = self.get_params()
        Epi = b*torch.exp(torch.lgamma(1+1./a) + torch.lgamma(b) - torch.lgamma(1+1./a + b))
        return Epi
    
    def get_params_kl(self):
        log_po = F.logsigmoid(self.p_post)
        log_po_ = F.logsigmoid(-self.p_post)
        log_pr = F.logsigmoid(self.p_pri)
        log_pr_ = F.logsigmoid(-self.p_pri)        
        
        return log_po,log_po_,log_pr,log_pr_
    
    def get_u_samp(self):
        return self.u_samp
    
    def get_u(self):
        return self.u

    def get_weight(self, training):
        if training:
            pi = self.sample_pi()
            p_z=Uniform(0.0,1.0)
            self.u_samp = p_z.rsample(torch.Size([self.num_samps,self.num_pars])).cuda()
            mask_11 = (-self.u_samp > -pi).float()
            mask_12 = (self.u_samp > 1.-pi).float()
            
            self.mask_samp=mask_11-mask_12

        else:
            Epi = self.get_Epi()
            mask = self.get_mask(Epi=Epi)
            mask_11 = Epi * mask
            mask_12 = Epi * mask
        return mask_11,mask_12
      
    def get_mask(self, Epi=None):
        Epi = self.get_Epi() if Epi is None else Epi
        return (Epi >= self.thres).float()
    
    def get_derivative(self,grad):
        a, b = self.get_params()
        da = torch.log((1 - self.u.pow(1./b))+1e-12) * (1 - self.u.pow(1./b)).pow(1./a) * (-1.) * (1./a**2)
        da = da * torch.sigmoid(self.a_uc)
        db = (1./a) * (1 - self.u.pow(1./b)).pow(1./a - 1.) * torch.log(self.u) * self.u.pow(1./b) * (1./b**2)
        db = db * torch.sigmoid(self.b_uc)
        derivs = torch.stack([da,db]).squeeze_()

        return grad*derivs
    
    def get_derivative_a(self,grad):
        a, b = self.get_params()
        da = torch.log((1 - self.u.pow(1./b))+1e-12) * (1 - self.u.pow(1./b)).pow(1./a) * (-1.) * (1./a**2)
        da = da * torch.sigmoid(self.a_uc)
        return grad*da
    
    def get_derivative_b(self,grad):
        a, b = self.get_params()
        db = (1./a) * (1 - self.u.pow(1./b)).pow(1./a - 1.) * torch.log(self.u) * self.u.pow(1./b) * (1./b**2)
        db = db * torch.sigmoid(self.b_uc)
        return grad*db
    
    def get_reg_deriv(self):
        a, b = self.get_params()
        dkl_a = (self.alpha/a**2)*(-0.577215664901532 - torch.digamma(b) - 1./b)  + a
        dkl_a = dkl_a * torch.sigmoid(self.a_uc)
        dkl_b = (1 - self.alpha/a) * (1./b**2 - torch.polygamma(1,b)) + b - (1./b**2)
        dkl_b = dkl_b * torch.sigmoid(self.a_uc)
        derivs_kl = torch.stack([dkl_a,dkl_b]).squeeze_()
        return derivs_kl
    
    def get_reg_deriv_a(self):
        a, b = self.get_params()
        dkl_a = (self.alpha/a**2)*(-0.577215664901532 - torch.digamma(b) - 1./b)  + a
        return dkl_a * torch.sigmoid(self.a_uc)
    
    def get_reg_deriv_b(self):
        a, b = self.get_params()
        dkl_b = (1 - self.alpha/a) * (1./b**2 - torch.polygamma(1,b)) + b - (1./b**2)
        return dkl_b * torch.sigmoid(self.b_uc)

    def get_reg(self):
        a, b = self.get_params()
        kld = (1 - self.alpha/a)*(-0.577215664901532 - torch.digamma(b) - 1./b)                 + torch.log(a*b + 1e-10) - math.log(self.alpha)                 - (b-1)/b
        kld = (self.kl_scale) * kld.sum()
        return kld

class BBDEGCN(nn.Module):
    def __init__(self, nfeat_list, dropout, adj, nlay, num_edges,num_nodes,sym):
        super(BBDEGCN, self).__init__()
        
        assert len(nfeat_list)==nlay+1
        self.nlay = nlay
        self.sym=sym
        self.num_edges = num_edges
        self.num_nodes = num_nodes
        self.drpedg = BBDE(nlay,num_edges)
        self.dropout = dropout
        gcs_list = []
        for i in range(nlay):
            gcs_list.append([str(i), GraphConvolution(nfeat_list[i], nfeat_list[i+1])])
        
        self.gcs = nn.ModuleDict(gcs_list)
    
    def forward(self, x, labels, adj, nz_idx, obs_idx, warm_up, adj_normt, training=True,mul_type='norm_sec', con_type='res_sum',sym=False):
        mask_vec1,mask_vec2 = self.drpedg.get_weight(training)
        mask_vec1=torch.squeeze(mask_vec1)
        mask_vec2=torch.squeeze(mask_vec2)
        h_perv1 = x
        h_perv2 = x
        x1 = x
        x2 = x
        adj_reg1 = 0.0
        adj_reg2 = 0.0
        for i in range(self.nlay):
            
            if self.sym: 
                mask_mat1 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                mask_mat1[nz_idx[0], nz_idx[1]] = mask_vec1[:,i]
                mask_mat1 = (mask_mat1 + mask_mat1.T) / 2.  
            else:
                mask_mat1 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                mask_mat1[nz_idx[0], nz_idx[1]] = mask_vec1[:,i]
            
            if mul_type=='norm_sec':
                adj_lay1 = normalize_torch(torch.mul(mask_mat1, adj) + torch.eye(adj.shape[0]).cuda())
            elif mul_type=='norm_first':
                adj_lay1 = torch.mul(mask_mat1, adj_normt).cuda()       
            
            if self.sym:
                mask_mat2 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                mask_mat2[nz_idx[0], nz_idx[1]] = mask_vec2[:, i]
                mask_mat2 = (mask_mat2 + mask_mat2.T) / 2.
        
            else:
                mask_mat2 = torch.zeros((self.num_nodes, self.num_nodes)).cuda()
                mask_mat2[nz_idx[0], nz_idx[1]] = mask_vec2[:,i]            
            if mul_type=='norm_sec':
                adj_lay2 = normalize_torch(torch.mul(mask_mat2, adj) + torch.eye(adj.shape[0]).cuda())
            elif mul_type=='norm_first':
                adj_lay2 = torch.mul(mask_mat2, adj_normt).cuda() 

            if con_type=='reg':
                if i<(self.nlay-1):
                    x1 = F.relu(self.gcs[str(i)](x1, adj_lay1))
                    x1 = F.dropout(x1, self.dropout, training=training)
                    x2 = F.relu(self.gcs[str(i)](x2, adj_lay2))
                    x2 = F.dropout(x2, self.dropout, training=training)
                else:
                    out1 = self.gcs[str(i)](x1, adj_lay1)
                    out2 = self.gcs[str(i)](x2, adj_lay2)
            


        
        output1 = F.log_softmax(out1, dim=1)
        
        output2 = F.log_softmax(out2, dim=1)
        
        nll_loss = self.loss(labels, output1, obs_idx)
 

        kld_loss = self.drpedg.get_reg()

        tot_loss = nll_loss 

        nll_loss2 = self.loss(labels, output2, obs_idx)

        tot_loss2 = nll_loss2 
        
        return output1, tot_loss, nll_loss, kld_loss, tot_loss2
    
    def loss(self, labels, preds, obs_idx):
        return F.nll_loss(preds[obs_idx], labels[obs_idx])
