#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author:hui.zhang
@file: data.py
@time: 2020/05/20
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#针对movielens数据集，利用user_id和movie_id特征

def load_data(filename):
    '''
    return train_data: 每一个样本是一个字典：{'user':*,'movie:'*'}，id编号从0开始
    '''
    data=[]
    with open(filename,'r')as infile:
        for line in infile.readlines():
            user_id,item_id,rate,timestamp=line.split('\t')

            instance={}
            #这里原本的数据集id是从1开始的，并且有序，所以只需要减1就可以了
            instance['user_id']=int(user_id)-1
            instance['movie_id']=int(item_id)-1

            instance['label']=1 if float(rate)>3.5 else 0
            data.append(instance)
    return data



def get_slot_max_num(train_file,test_file):
    '''
    获取每个域的取值的个数
    '''
    slot_max_num={}
    user_count=[]
    item_count=[]
    with open(train_file,'r')as infile:
        for line in infile.readlines():
            user_id, item_id, rate, timestamp = line.split('\t')
            if user_id not in user_count:
                user_count.append(user_id)
            if item_id not in item_count:
                item_count.append(item_id)
    with open(test_file,'r')as infile:
        for line in infile.readlines():
            user_id, item_id, rate, timestamp = line.split('\t')
            if user_id not in user_count:
                user_count.append(user_id)
            if item_id not in item_count:
                item_count.append(item_id)

    slot_max_num['user_id']=len(user_count)
    slot_max_num['movie_id']=len(item_count)

    return slot_max_num



if __name__=='__main__':
    print(load_data('../data/ua.base'))