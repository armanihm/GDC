from __future__ import print_function
from __future__ import division

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# import libraries
import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy import stats
import torch
import math


# torch.cuda.device_count()


# seed = 5
# np.random.seed(seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)



# util functions

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_data(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, 1))
    
    idx_train = torch.LongTensor(np.where(train_mask)[0])
    idx_val = torch.LongTensor(np.where(val_mask)[0])
    idx_test = torch.LongTensor(np.where(test_mask)[0])

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_torch(mx):
    rowsum = torch.sum(mx, 1)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_inv[torch.isnan(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = torch.matmul(r_mat_inv, mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy_mrun(outputs, labels, inds):
    preds = outputs.max(2)[1].mode(0)[0].type_as(labels)
    correct = preds[inds].eq(labels[inds]).double()
    correct = correct.sum()
    return correct / len(inds)

def accuracy_mrun_np(outputs, labels_np, inds):
    preds_np = stats.mode(np.argmax(outputs, axis=2), axis=0)[0].reshape(-1).astype(np.int32)
    correct = np.equal(labels_np[inds], preds_np[inds]).astype(np.float32)
    correct = correct.sum()
    return correct / len(inds)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def uncertainty_mrun(outputs, labels, inds):
    softmax_mean = torch.exp(outputs).sum(0)/float(outputs.shape[0])
    log_softmax_mean = torch.log(softmax_mean + 1e-12)
    pred_entropy = -torch.sum(softmax_mean * log_softmax_mean,1)
    preds = outputs.max(2)[1].mode(0)[0].type_as(labels)
    entropies_list = pred_entropy[inds]
    preds_list = preds[inds]
    labels_list = labels[inds]

    num_thr = 10.
    thr = (entropies_list.max().data - entropies_list.min().data)/num_thr
    thr_start = entropies_list.min().data
    p_ac = [None]*int(num_thr + 1)
    p_ui = [None]*int(num_thr + 1)
    pavpu = [None]*int(num_thr + 1)
    for i in range(int(num_thr + 1)):
        tr = thr_start + thr*i
        mask_certain=(entropies_list < tr).double()
        accurate = preds_list.eq(labels_list).double()
        inaccurate = 1. - accurate
        ac = torch.sum(accurate*mask_certain).double()
        iu = torch.sum(inaccurate*(1.-mask_certain)).double()

        p_ac[i] = ac/torch.sum(mask_certain).double()
        p_ui[i] = iu/torch.sum(inaccurate).double()
        pavpu[i] = (ac+iu)/(torch.sum(mask_certain)+torch.sum(1.-mask_certain))

    return torch.stack(pavpu)