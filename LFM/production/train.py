#-*- coding:utf-8 -*-
#LFM model train
import numpy as np
import sys
sys.path.append("../utils")
import utils.read as read
import operator

def lfm_train(train_data,F,alpha,beta,step):
    '''
    训练模型
    :param train_data:
    :param F: 隐向量维度
    :param alpha: 正则化系数
    :param beta: 学习速率
    :param step: 迭代次数
    :return: dict：np.ndarray
    '''
    user_vec={}
    item_vec={}
    for step_index in range(step):
        #每一次迭代中，从训练样本中获取实例
        for data_instance in train_data:
            userid,itemid,label=data_instance
            if userid not in user_vec:
                #如果当前用户是一次出现，则需要进行初始化
                user_vec[userid]=init_model(F)
            if itemid not in item_vec:
                item_vec[itemid]=init_model(F)
        delta=label-model_predict(user_vec[userid],item_vec[itemid])
        for index in range(F):
            user_vec[userid][index]+=beta*(delta*item_vec[itemid][index]-alpha*user_vec[userid][index])
            item_vec[itemid][index]+=beta*(delta*user_vec[userid][index]-alpha*item_vec[itemid][index])
        beta=beta*0.9   #对学习率进行一个衰减，目的是让模型在接近收敛时变化的慢一点
    return user_vec,item_vec

def init_model(F):
    '''
    初始化隐向量
    :param F:
    :return: ndarray
    '''
    return np.random.randn(F)    #使用标准正态分布进行初始化

def model_predict(user_vector,item_vector):
    '''
    模型产生的表征用户和item的向量
    :param user_vector:
    :param item_vector:
    :return:
    '''
    #矩阵相乘：np.dot
    #np.linalg.norm：二范数
    res=np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res

def model_train_process():
    '''
    test model train
    :return:
    '''
    train_data=read.get_train_data("../data/ratings.csv")
    user_vec,item_vec=lfm_train(train_data,50,0.01,0.1,50)
    recom_list=give_recom_result(user_vec,item_vec,user_id='24')
    ana_recom_result(train_data,'24',recom_list)


def give_recom_result(user_vec,item_vec,user_id):
    '''
    通过lfm模型得到用户的推荐候选集
    :param user_vec:
    :param item_vec:
    :param user_id:
    :return: list:(itemid,score)
    '''
    fix_num=10   #候选集大小
    if user_id not in user_vec:
        return []
    record={}
    recom_list=[]
    user_vector=user_vec[user_id]
    for itemid in item_vec:
        #计算用户向量和item向量之间的距离
        item_vector=item_vec[itemid]
        res=np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
        record[itemid]=res
    for instance in sorted(record.items(),key=operator.itemgetter(1),reverse=True)[:fix_num]:
        itemid=instance[0]
        score=round(instance[1],3)
        recom_list.append((itemid,score))
    return recom_list

def ana_recom_result(train_data,user_id,recom_list):
    '''
    评价推荐结果
    :param train_data:
    :param user_id:
    :param recom_list:
    :return:
    '''
    item_info=read.get_item_info("../data/movies.csv")
    for data_instance in train_data:
        tmp_userid,itemid,label=data_instance
        if tmp_userid==user_id and label==1:
            print(item_info[itemid])
    print("recommend result")
    for instance in recom_list:
        print(item_info[instance[0]])

if __name__=='__main__':
   model_train_process()