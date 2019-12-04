#-*-coding:utf-8 -*-

#train lr model
import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LRCV    #使用交叉验证的LR
from sklearn.utils import shuffle
from sklearn.externals import joblib

def train_lr_model(train_file,model_coef,model_file):
    '''
    训练模型
    :param train_file:
    :param model_coef:
    :param model_file:
    :return:
    '''
    total_feature_num=118
    train_label=np.genfromtxt(train_file,dtype=np.int32,delimiter=",",usecols=-1)  #标签是最后一列
    feature_list=list(range(total_feature_num-1))  #0-116
    print(feature_list)
    train_feature=np.genfromtxt(train_file,dtype=np.int32,delimiter=",",usecols=feature_list)
    #print(np.shape(train_feature))
    #迭代停止条件tol=0.0001,最大迭代次数为500，交叉验证为5，即每次选择20%作为测试,由于这里数据不是很多，默认最优化方法使用拟牛顿法
    lr_cf=LRCV(Cs=[1,10,100],penalty='l2',tol=0.0001,max_iter=500,cv=5,scoring="roc_auc").fit(train_feature,train_label)   #Cs的值的倒数是正则化的参数，这里设置多个即1,0.1,0.01，正则化选择l2
    scores=lr_cf.scores_.values()   #模型训练的准确率，scores是一个5行3列的数组，即每次交叉验证和不同正则化参数下的准确率
    print(scores)
    #1、将模型的参数输出
    coef=lr_cf.coef_[0]
    fw=open(model_coef,"w+")
    fw.write(",".join(str(ele) for ele in coef))
    fw.close()
    #2、将整个模型实例成一个对象输出
    joblib.dump(lr_cf,model_file)



if __name__=='__main__':
     train_lr_model("../data/train_preprocess.txt","../data/lr_coef.txt","../data/lr_mode_file.txt")