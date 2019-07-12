#-*- coding:utf-8 -*-
#check the performance on test file
#准确率和AUC
#AUC表示了模型预估样本的一种序关系：分母是正负样本对 分子是看正样本比负样本预测高的个数之和
from __future__ import division
import numpy as np
from sklearn.externals import joblib
import math

def get_test_data(input_file):
    '''
    读入测试数据
    :param input_file:
    :return: np.array test_feature test_label
    '''
    total_feature_num=118
    test_label=np.genfromtxt(input_file,dtype=np.float32,delimiter=",",usecols=-1)
    feature_list=list(range(total_feature_num))
    test_feature=np.genfromtxt(input_file,dtype=np.float32,delimiter=",",usecols=feature_list)
    return test_feature,test_label

def predict_by_lr_model(test_feature,lr_model):
    #直接调用模型函数
    prob_list=lr_model.predict_proba(test_feature)  #既包含了样本为0(第0列）的概率，也包含了样本为1的概率
    result_list=[]  #将每个样本为1的概率返回
    for index in range(len(prob_list)):
        result_list.append(prob_list[index][1])
    return result_list


def run_check_core(test_feature,test_label,model,score_func):
    '''
    使用不同的打分模型计算评价指标,通过结果可以发现不同打分模型下AUC的值是不变的，但是accuracy的值不同
    :param test_feature:
    :param test_label:
    :param model:
    :param score_func:
    :return:
    '''
    predict_list=score_func(test_feature,model)  #计算每个样本预估为1的概率
    get_auc(predict_list,test_label)
    get_accuracy(predict_list,test_label)

def get_auc(predict_list,test_label):
    '''

    :param predict_list:
    :param test_label:
    :return:       auc=(sum(pos_index)-pos_num(pos_num+1)/2)/pos_num*neg_num
    '''
    total_list=[]
    for index in range(len(predict_list)):
        predict_score=predict_list[index]
        label=test_label[index]
        total_list.append((label,predict_score))
    sorted_total_list=sorted(total_list,key=lambda ele:ele[1])
    neg_num=0
    pos_num=0
    count=1
    total_pos_index=0
    for instance in sorted_total_list:
        label,predict_score=instance
        if label==0:
            neg_num+=1
        else:
            pos_num+=1
            total_pos_index+=count
        count+=1
    auc_score=(total_pos_index-(pos_num)*(pos_num+1)/2)/(pos_num*neg_num)
    print("auc_score is:" ,auc_score)


def get_accuracy(predict_list,test_label):
    '''
    计算准确率
    :param predict_list:
    :param test_label:
    :return:
    '''
    score_thr=0.5
    right_num=0
    for index in range(len(predict_list)):
        predict_score=predict_list[index]
        if predict_score>=score_thr:
            predict_label=1
        else:
            predict_label=0
        if predict_label==test_label[index]:
            right_num+=1
    total_num=len(predict_list)
    accuracy_score=right_num/total_num
    print("accuracy is:",accuracy_score)

def sigmoid(x):
    return 1/(1+math.exp(-x))

def predict_by_lr_coef(test_feature,lr_coef):
    sigmoid_func=np.frompyfunc(sigmoid,1,1)   #创建通用函数，1个输入，1个返回值
    return sigmoid_func(np.dot(test_feature,lr_coef))

def run_check(test_file,lr_coef_file,lr_model_file):
    '''

    :param test_file: 测试文件
    :param lr_coef_file: 参数化的模型
    :param lr_model_file: 实例化的模型
    :return:
    '''
    test_feature,test_label=get_test_data(test_file)
    lr_coef=np.genfromtxt(lr_coef_file,dtype=np.float32,delimiter=",")
    lr_model=joblib.load(lr_model_file)
    run_check_core(test_feature,test_label,lr_model,predict_by_lr_model)
    run_check_core(test_feature,test_label,lr_coef,predict_by_lr_coef)

if __name__=='__main__':
     run_check("../data/test_preprocess.txt","../data/lr_coef.txt","../data/lr_mode_file.txt")