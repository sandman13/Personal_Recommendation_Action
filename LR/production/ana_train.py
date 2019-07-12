#-*-coding:utf-8 -*-
#feature selection and data selection

import os
import operator
import numpy as np
import pandas as pd
import sys

def get_input(input_train_file,input_test_file):
    '''

    :param input_train_file:
    :param input_test_file:
    :return:  two dataFrame
    '''
    dtype_dict={"age":np.int32,
                "education-num":np.int32,
                "capital-gain":np.int32,
                "capital-loss":np.int32,
                "hours-per-week":np.int32}
    use_list=list(range(15))
    #print(use_list)
    use_list.remove(2)   #id不需要
    #print(use_list)
    #指定一些列的类型，这里要注意设置缺省值的时候，数据集的？前面有一个空格，并且指定需要哪些列
    train_data_df=pd.read_csv(input_train_file,sep=",",header=0,dtype=dtype_dict,na_values=' ?',usecols=use_list)
    print(len(train_data_df))
    train_data_df=train_data_df.dropna(axis=0,how="any")
    print(len(train_data_df))
    test_data_df = pd.read_csv(input_test_file, sep=",", header=0, dtype=dtype_dict, na_values=' ?', usecols=use_list)
    print(len(test_data_df))
    test_data= test_data_df.dropna(axis=0, how="any")  # 删除所有含有缺省值的行
    print(len(test_data))
    return train_data_df,test_data_df

def train_label_trans(x):
    if x==" <=50K":
        return "0"
    if x==" >50K":
        return "1"
    return "0"

def test_label_trans(x):
    if x==" <=50K.":
        return "0"
    if x==" >50K.":
        return "1"
    return "0"

def process_label_feature(label_feature_str,train_df,test_df):
    '''
    将标签那一列转换为0/1
    :param label_feature_str:   "label"
    :param df_in:
    :return:
    '''
    train_df.loc[:,label_feature_str]=train_df.loc[:,label_feature_str].apply(train_label_trans)
    test_df.loc[:,label_feature_str]=test_df.loc[:,label_feature_str].apply(test_label_trans)
    return train_df,test_df


def dict_trans(dict_in):
    '''
    key:str value:int   根据出现的次数对特征进行排序
    :param dict_in:
    :return:
    '''
    output_dict={}
    index=0
    for instance in sorted(dict_in.items(),key=operator.itemgetter(1),reverse=True):
        output_dict[instance[0]]=index
        index+=1
    return output_dict

def dis_to_feature(x,feature_dict):
    '''
    将特征和数值对应，返回一个str:0,1,0,0,0,
    :param x:
    :param feature_dict:
    :return:
    '''
    output_list=[0]*len(feature_dict)
    if x not in feature_dict:
        return ",".join([str(ele) for ele in output_list])
    else:
        #对应位置置为1
        index=feature_dict[x]
        output_list[index]=1
    return ",".join([str(ele) for ele in output_list])

def process_dis_feature(feature_str,df_train,df_test):
    '''
    处理离散特征，这里训练数据要和测试数据一起处理
    :param label_feature_str:
    :param df_train:
    :param df_test:
    :return: dimension
    '''
    origin_dict=df_train.loc[:,feature_str].value_counts().to_dict()
    #print(origin_dict)
    #得到一个dictionary：key:feature取值 value：index
    feature_dict=dict_trans(origin_dict)
    #print(feature_dict)
    df_train.loc[:,feature_str]=df_train.loc[:,feature_str].apply(dis_to_feature,args=(feature_dict,))
    df_test.loc[:, feature_str] = df_test.loc[:, feature_str].apply(dis_to_feature, args=(feature_dict,))
    #print(df_train.loc[:2,feature_str])
    return len(feature_dict)   #返回每一个特征离散化后的维度

def list_trans(input_dict):
    '''

    :param input_dict:
    :return:
    '''
    output_list=[0]*5
    key_list=["min","25%","50%","75%","max"]
    for index in range(len(key_list)):
        fix_key=key_list[index]
        if fix_key not in input_dict:  #如果键值在输入字典中不存在，则直接报错返回
            print("error")
            sys.exit()
        else:
            output_list[index]=input_dict[fix_key]
    return output_list

def con_to_feature(x,feature_list):
    '''
    将特征离散化
    :param x:
    :param feature_list:
    :return:
    '''
    feature_len=len(feature_list)-1   #只要判断这个值位于哪一个区间内，则相应的位置置为1
    result=[0]*feature_len
    for index in range(feature_len):
        if x>=feature_list[index] and x<=feature_list[index+1]:
            result[index]=1
            return ",".join([str(ele)for ele in result])
    return ",".join([str(ele)for ele in result])

def process_con_feature(feature_str,df_train,df_test):
    '''
    处理连续特征
    :param feature_str:
    :param df_train:
    :param df_test:
    :return:
    '''
    #首先先确定分布，使用自带的describe()函数
    origin_dict=df_train.loc[:,feature_str].describe().to_dict()
    print(origin_dict)
    #根据分位点来处理连续特征
    feature_list=list_trans(origin_dict)
    df_train.loc[:,feature_str]=df_train.loc[:,feature_str].apply(con_to_feature,args=(feature_list,))
    df_test.loc[:,feature_str]=df_test.loc[:,feature_str].apply(con_to_feature,args=(feature_list,))
    return len(feature_list)-1

def ana_train_data(input_train_data,input_test_data,out_train_file,out_test_file):
    '''

    :param input_train_data:
    :param input_test_data:
    :param out_train_file:
    :param out_test_file:
    :return:
    '''
    train_data_df,test_data_df=get_input(input_train_data,input_test_data)
    label_feature_str="label"
    #数据集中的离散特征列表
    dis_feature_list=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    #处理label
    train_data_df,test_data_df=process_label_feature(label_feature_str,train_data_df,test_data_df)
    #处理离散化特征
    dis_feature_num=0
    for feature in dis_feature_list:
        dis_feature_num+=process_dis_feature(feature,train_data_df,test_data_df)
    con_feature_list=['education-num','age','capital-gain','capital-loss','hours-per-week']
    con_feature_num=0
    for feature in con_feature_list:
        con_feature_num+=process_con_feature(feature,train_data_df,test_data_df)
    print(train_data_df[:2])
    print(dis_feature_num)
    print(con_feature_num)
    output_file(train_data_df,out_train_file)
    output_file(test_data_df,out_test_file)

def output_file(df_in,out_file):
    '''
    将dataFrame写入文件
    :param df_in:
    :param out_file:
    :return:
    '''
    fw=open(out_file,"w+")
    for row_index in df_in.index:
        outline=",".join([str(ele) for ele in df_in.loc[row_index].values])
        fw.write(outline+"\n")
    fw.close()

if __name__=="__main__":
    ana_train_data("../data/train.txt","../data/test.txt","../data/train_preprocess.txt","../data/test_preprocess.txt")