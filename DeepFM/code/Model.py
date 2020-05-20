#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author:hui zhang
@file: DeepFM.py
@time: 2020/05/20
FM/DeepFM/DeepFM的pytorch code
"""

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class fm_model(nn.Module):

    def __init__(self, name='ffm', slot_max_num={}, dim=32, bias_max_norm=0.1,
                 vec_max_norm=0.1, use_fm=True, use_ffm=True,use_deep=True, hidden_layers=[16, 8]):
        super(Model,self).__init__()
        self.embedding_v = nn.ModuleDict()   #embedding
        self.embedding_b = nn.ModuleDict()  #embedding bias
        self.b = nn.Parameter(torch.zeros(1))
        self.slot_max_num = slot_max_num   #FFM中field的个数
        self.dim = dim   #embedding的size
        self.use_fm = use_fm
        self.use_ffm = use_ffm
        self.use_deep = use_deep
        self.hidden_layers = hidden_layers

        #input layer -> fc layer1
        self.deep_fc1 = nn.Linear(len(self.slot_max_num) * self.dim, self.hidden_layers[0])
        #fc layer1 -> fc layer2
        self.deep_fc2 = nn.Linear(self.hidden_layers[0], self.hidden_layers[1])
        #fc layer2 -> output layer
        self.deep_fc3 = nn.Linear(self.hidden_layers[1], 1)

        for k in self.slot_max_num:
            #对每个field构造一个embedding矩阵：feature_size*embedding_size
            max_num = self.slot_max_num[k]
            self.embedding_v[k] = nn.Embedding(max_num, self.dim, max_norm=vec_max_norm, padding_idx=0)
            self.embedding_b[k] = nn.Embedding(max_num, 1, max_norm=bias_max_norm, padding_idx=0)


    def forward(self, inputs):
        tf_bias = {}
        tf_vec = {}
        for slot in self.slot_max_num:
            x = torch.tensor([i[slot] for i in inputs])  # [32]
            tf_bias[slot] = self.embedding_b[slot](x)
            tf_vec[slot] = self.embedding_v[slot](x)  # [32*8]

        sum_bias = torch.stack([tf_bias[i] for i in tf_bias]).sum(dim=0) + self.b

        u_keys = ['user_id'] #user feature
        g_keys = ['movie_id']   #item feature

        user = torch.stack([tf_vec[i] for i in u_keys if i in tf_vec]).sum(dim=0)
        group = torch.stack([tf_vec[i] for i in g_keys if i in tf_vec]).sum(dim=0)

        # fm
        fm = torch.sum(torch.mul(user, group), dim=1, keepdim=True)

        # deepfm
        deep_fm = nn.functional.relu(self.deep_fc1(torch.cat([tf_vec[i] for i in tf_vec], dim=1)))
        deep_fm = nn.functional.relu(self.deep_fc2(deep_fm))
        deep_fm = nn.functional.relu(self.deep_fc3(deep_fm))

        # use deepfm
        if self.use_fm and self.use_deep:
            y = torch.sigmoid(sum_bias + fm + deep_fm)

        else:
            # use fm
            y = torch.sigmoid(sum_bias + fm)

        return y


    def test(self, inputs, need_p=True, name=""):
        y = np.array([i['label'] for i in inputs], dtype=np.float32)
        y = y.reshape((len(y), 1))
        p = self.forward(inputs)
        p = p.detach().numpy()
        y_avg = np.mean(y)
        p_avg = np.mean(p)
        auc = roc_auc_score(y, p)
        if need_p:
            print('%s auc:%.4f, y_avg=%.3f p_avg=%.3f' % (name, auc, y_avg, p_avg))
        return auc

