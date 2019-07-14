#-*- coding:utf-8 -*-
from __future__ import division
import numpy as np
import sys
import xgboost as xgb
import math
sys.path.append("../")
import train as TA
from scipy.sparse import csc_matrix

def get_test_data(test_file,feature_num_file):
	"""
		:param test_file: file to check performance
		:param feature_num_file: the file record total num of feature
		:return:
			two np.array:test_feature,test_label
		"""
	total_feature_num =103
	test_label = np.genfromtxt(test_file, dtype=np.float32, delimiter=",", usecols=-1)
	feature_list = range(total_feature_num)
	test_feature = np.genfromtxt(test_file, dtype=np.float32, delimiter=",", usecols=feature_list)
	return test_feature, test_label

def predict_by_tree(test_feature,tree_model):
    '''
    测试特征和树模型
    :param test_feature:
    :param tree_model:
    :return:
    '''
    predict_list=tree_model.predict(xgb.DMatrix(test_feature))#先转换一下数据结构，得到预测列表
    return predict_list

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

def run_check_core(test_feature,test_label,model,score_func):
    '''

    :param test_feature:
    :param test_label:
    :param model:
    :param score_func:
    :return:
    '''
    predict_list=score_func(test_feature,model)
    get_auc(predict_list,test_label)
    get_accuracy(predict_list,test_label)



def run_check(test_file,tree_model_file,feature_num_file):
    '''
    测试树模型在测试集上的表现
    :param test_file:
    :param tree_model_file:
    :param feature_num_file:
    :return:
    '''
    test_feature,test_label=get_test_data(test_file,feature_num_file)
    tree_model=xgb.Booster(model_file=tree_model_file)
    run_check_core(test_feature,test_label,tree_model,predict_by_tree)

def run_check_lr_gdbt(test_file,tree_mix_model_file,lr_coef_mix_model_file,feature_num_file):
    '''
    GBDT混合模型
    :param test_file:
    :param tree_mix_model_file:
    :param lr_coef_mix_model_file:
    :param feature_num_file:
    :return:
    '''
    test_feature,test_label=get_test_data(test_file,feature_num_file)
    mix_tree_model=xgb.Booster(model_file=tree_mix_model_file)
    mix_lr_model=np.genfromtxt(lr_coef_mix_model_file,dtype=np.float32,delimiter=",")
    tree_info=(4,10,0.3)
    run_check_lr_gbdt_core(test_feature,test_label,mix_tree_model,mix_lr_model,tree_info,predict_by_mix)


def predict_by_mix(test_feature,mix_tree_model,mix_lr_model,tree_info):
    tree_leaf=mix_tree_model.predict(xgb.DMatrix(test_feature),pred_leaf=True)  #首先预测每个样本在GBDT后落在哪一个节点上
    (tree_depth,tree_num,step_size)=tree_info
    total_feature_list=TA.get_gbdt_and_lr_feature(tree_leaf,tree_depth=tree_depth,tree_num=tree_num)  #得到的特征是稀疏矩阵的形式
    result_list=np.dot(csc_matrix(mix_lr_model),total_feature_list.tocsc().T).toarray()[0]
    sigmoid_ufunc=np.frompyfunc(sigmoid,1,1)
    return sigmoid_ufunc(result_list)


def run_check_lr_gbdt_core(test_feature,test_label,mix_tree_model,mix_lr_model,tree_info,score_func):
    predict_list=score_func(test_feature,mix_tree_model,mix_lr_model,tree_info)
    get_auc(predict_list,test_label)
    get_accuracy(predict_list,test_label)

if __name__=="__main__":
    #run_check("../data/test_preprocess.txt","../data/tree_model.txt","../data/feature_num.txt")
    run_check_lr_gdbt("../data/test_preprocess.txt","../data/xgb_mix_model","../data/lr_coef_mix_model","../data/feature_num.txt")