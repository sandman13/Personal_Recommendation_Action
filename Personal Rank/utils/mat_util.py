#_-*-coding:utf-8 -*-
from scipy.sparse import coo_matrix
import numpy as np
from utils import *
from utils import read


def graph_to_matrix(graph):
    '''
    将二分图转换为矩阵
    :param graph:
    :return:稀疏矩阵
    '''
    vertex=list(graph.keys() ) #记录所有的顶点
    address_dict={}
    total_len=len(vertex)
    for index in range(total_len):
        address_dict[vertex[index]]=index
    row=[]
    col=[]
    data=[]
    for element_i in graph:
        weight=round(1/len(graph[element_i]),3)
        row_index=address_dict[element_i]
        for element_j in graph[element_i]:
            col_index=address_dict[element_j]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)
    row=np.array(row)
    col=np.array(col)
    data=np.array(data)
    m=coo_matrix((data,(row,col)),shape=(total_len,total_len))
    return m,vertex,address_dict


def mat_all_point(mat,vertex,alpha):
    '''
    计算E-alpha*mat.T
    :param mat:
    :param vertex:
    :param alpha:
    :return:
    '''
    total_len=len(vertex)
    row=[]
    col=[]
    data=[]
    for index in range(total_len):   #利用稀疏矩阵的方式初始化一个单位矩阵
        row.append(index)
        col.append(index)    #只有对角线的值为1
        data.append(1)
    row=np.array(row)
    col=np.array(col)
    data=np.array(data)
    eye_t=coo_matrix((data,(col,row)),shape=(total_len,total_len))
    return eye_t.tocsr()-alpha*mat.tocsr().transpose()   #使用csr计算会快一点

if __name__=='__main__':
    graph=read.get_graph_from_data("../data/test.txt")
    m,vertex,address_dict=graph_to_matrix(graph)
    print(m)
    print(mat_all_point(m,vertex,0.8))