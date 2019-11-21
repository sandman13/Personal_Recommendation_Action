#-*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from scipy.sparse import csr
from itertools import count
from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer


def create_csr_matrix(dic,index=None,dim=None):
    '''
    将数据集的原始列表输入转为一个csr矩阵
    :param dic:
    :param index:
    :param dim:
    :return:
    '''
    if index==None:
        d=count(0)   #创建一个从0开始，step为1的无限迭代器
        index=defaultdict(lambda :next(d))   #defaultdict的作用是当key不存在的时候，不报错而返回默认值
    sample_num=len(list(dic.values())[0])  #样本数:90570
    feature_num=len(list(dic.keys()))      #特征数:2
    total_num=sample_num*feature_num

    col_ix=np.empty(total_num,dtype=int)
    i=0
    for k,lis in dic.items():
        col_ix[i::feature_num]=[index[str(k)+str(el)] for el in lis]
        i+=1

    row_ix=np.repeat(np.arange(sample_num),feature_num)  #每一个元素重复feature_num次
    data=np.ones(total_num)

    if dim is None:
        dim=len(index)

    left_data_index=np.where(col_ix<dim)
    return csr.csr_matrix((data[left_data_index],(row_ix[left_data_index],col_ix[left_data_index])),
                          shape=(sample_num,dim)),index


def load_dataset():
    cols=['user','item','rating','timestamp']
    train=pd.read_csv("../data/ua.base",delimiter='\t',names=cols)
    test=pd.read_csv("../data/ua.test",delimiter='\t',names=cols)
    #print("train shape:",len(train))
    #print("test shape:",len(test))
    x_train,label_index=create_csr_matrix({'users':train.user.values,'items':train.item.values})
    x_test,label_index=create_csr_matrix({'users':test.user.values,'items':test.item.values},label_index,x_train.shape[1])
    print(len(label_index))
    y_train=train.rating.values
    y_test=test.rating.values

    x_train=x_train.todense()
    x_test=x_test.todense()

    return x_train,x_test,y_train,y_test


