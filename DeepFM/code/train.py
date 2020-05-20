#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author:hui.zhang
@file: train.py
@time: 2020/05/20
"""

from data import *
from Model import fm_model
from sklearn.metrics import roc_auc_score
import torch
import time


if __name__=='__main__':
    train_data=load_data("../data/ua.base")
    test_data=load_data("../data/ua.test")

    slot_max_num=get_slot_max_num("../data/ua.base","../data/ua.test")  #两个field
    model=fm_model(slot_max_num=slot_max_num, dim=64, bias_max_norm=0.1, vec_max_norm=0.1,use_fm=True,use_deep=False,hidden_layers=[8, 4])

    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
    criterion = torch.nn.BCELoss()  # 交叉熵损失函数

    t0 = time.time()
    print("start time",t0)

    iter_time=5
    for i in range(iter_time):
        batch_size = 32
        loss = 0
        st = 0
        index = 0

        while 1:
            tt = time.time()
            inputs = [d for d in train_data[st:st + batch_size]]
            if len(inputs) < 1:
                break

            y = np.array([i['label'] for i in inputs], dtype=np.float32)
            y = y.reshape((len(y), 1))

            model.zero_grad()
            p = model(inputs)

            loss = criterion(p, torch.tensor(y))
            loss.backward()
            optimizer.step()

            st += batch_size
        #每迭代一次，对测试集计算AUC

        model.test(test_data)

    print('finish train. %.2f' % (time.time() - t0))

    # 保存训练好的模型
    #torch.save(model.state_dict(), 'deepfm_params.pt')