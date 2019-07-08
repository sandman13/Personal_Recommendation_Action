#-*- coding:utf-8 -*-
from __future__ import division
import operator
import sys
sys.path.append("../utils")
from utils import read
from utils import mat_util
from scipy.sparse.linalg import gmres
import numpy as np

def personal_rank(graph,root,alpha,iter_step,recom_num=10):
    '''
    使用非矩阵化的方法实现personal rank算法
    :param graph: 根据用户行为得到的图
    :param root: 将要给哪一个用户推荐
    :param alpha: 游走概率
    :param iter_step: 迭代次数
    :param recom_num: 推荐数目
    :return:  dict: key:item_id, value:pr
    '''
    rank={}
    rank={point:0 for point in graph}  #将所有顶点的PR值初始为0
    rank[root]=1  #固定顶点的PR值为1
    recom_result={}
    for item_index in range(iter_step):
        tmp_rank={}    #临时存放结果
        tmp_rank={point:0 for point in graph}
        for out_point,out_dict in graph.items(): #对于graph中每一个字典对
            for inner_point,value in graph[out_point].items():
                #每一个节点的PR值等于所有有边指向当前节点的节点的PR值的等分
                tmp_rank[inner_point]+=round(alpha*rank[out_point]/len(out_dict),4)
                if inner_point==root:
                    tmp_rank[inner_point]+=round(1-alpha,4)
        if tmp_rank==rank:
            print("out")
            break    #提前结束迭代
        rank=tmp_rank
    right_num=0    #记录实际能推荐的item数
    for instance in sorted(rank.items(),key=operator.itemgetter(1),reverse=True):
        point,pr_score=instance[0],instance[1]
        if len(point.split('_'))<2:
            continue   #如果不是item，则跳过
        if point in graph[root]:
            continue   #如果这个item用户已经点击过，则跳过
        recom_result[point]=pr_score
        right_num+=1
        if right_num>recom_num:   #比较实际推荐的item数是否已经达到结果要求返回的item数
            break
    return recom_result

def personal_rank_mat(graph,root,alpha,recom_num=10):
    '''
    通过矩阵化计算
    :param graph:
    :param root:
    :param alpha:
    :param recom_num:
    :return:  A*r=r0
    '''
    m,vertex,address_dict=mat_util.graph_to_matrix(graph)
    if root not in address_dict:
        return {}
    mat_all=mat_util.mat_all_point(m,vertex,alpha)    #A
    score_dict={}
    recom_dict={}
    index=address_dict[root]
    initial_list=[[0] for i in range(len(vertex))]   #初始化r0矩阵  one-hot
    initial_list[index]=[1]
    r_zero=np.array(initial_list)
    res=gmres(mat_all,r_zero,tol=1e-8)[0]   #计算线性代数方程定义好误差
    for index in range(len(res)):
        point=vertex[index]
        if len(point.strip().split('_'))<2:
            continue
        if point in graph[root]:
            continue
        score_dict[point]=round(res[index],3)
        for instance in sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)[:recom_num]:
            point,score=instance[0],instance[1]
            recom_dict[point]=score
    return recom_dict




def get_one_user_recom():
    '''
    计算一个用户的推荐item的候选集
    :return:
    '''
    graph=read.get_graph_from_data("../data/ratings.csv")
    alpha=0.6
    user='1'
    iter_num=100
    recom_result=personal_rank(graph,user,alpha,iter_num,100)
    item_info=read.get_item_info("../data/movies.csv")
    for itemid in recom_result:
        pure_itemid=itemid.split("_")[1]
        print(item_info[pure_itemid])
        print(recom_result[itemid])
    return recom_result

def get_one_user_by_mat():
  '''
  矩阵化后测试结果
  '''
  user='1'
  alpha=0.8
  graph=read.get_graph_from_data("../data/ratings.csv")
  recom_result=personal_rank_mat(graph,user,alpha,100)
  return recom_result


if __name__=='__main__':
    recom_result_base=get_one_user_recom()
    recom_result_mat=get_one_user_by_mat()
    num=0
    for element in recom_result_base:
        if element in recom_result_mat:
            num+=1
    print(num)
