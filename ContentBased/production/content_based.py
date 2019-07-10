#-*- coding:utf-8 -*-

from __future__ import division
import os
import operator
import sys
sys.path.append("../")
import util.read as read


def get_up(item_cate,input_file):
    '''

    :param item_cate: key:item_id value{cate:ratio}
    :param input_file: 用户评分文件
    :return:
    '''
    if not os.path.exists(input_file):
        return{}
    linenum=0
    record={}
    score_thr=4.0
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(",")
        if len(item)<4:
            continue
        userid,itemid,rating,timestamp=item[0],item[1],float(item[2]),int(item[3])
        if rating<score_thr:
            continue
        if itemid not in item_cate:
            continue
        time_score=get_time_score(timestamp)
        if userid not in record:
            record[userid]={}
        for cate in item_cate[itemid]:
            if cate not in record[userid]:
                record[userid][cate]=0
            record[userid][cate]+=rating*time_score*item_cate[itemid][cate]  #用户对某一类别的偏好程度
    fp.close()
    up={}  #保存结果
    topK=2
    for userid in record:
        if userid not in up:
            up[userid]=[]
        total_score=0
        for instance in sorted(record[userid].items(),key=operator.itemgetter(1),reverse=True)[:topK]:
            up[userid].append((instance[0],instance[1]))
            total_score+=instance[1]    #用来对类别分数归一化
        for index in range(len(up[userid])):
            up[userid][index]=(up[userid][index][0],round(up[userid][index][1]/total_score,3))
    return up

def get_time_score(timestamp):
    '''

    :param timestamp: 时间戳
    :return: 时间的得分
    '''
    #已知最大时间戳是1537799250
    max_timestamp=1537799250
    total_sec=24*60*60
    delta=(max_timestamp-timestamp)/total_sec/100  #时间越近，即差距越小，分数越大
    return 1/(1+delta)

def recom(cate_item_sort,up,userid,topK=10):
    '''

    :param cate_item_sort:  类别中item的倒排结果
    :param up: 用户对类别的偏好程度
    :param userid:
    :param topK:
    :return:
    '''
    if userid not in up:
        return{}
    recom_result={}
    if userid not in recom_result:
        recom_result[userid]=[]
    for instance in up[userid]:
        cate=instance[0]
        ratio=instance[1]
        num=int(topK*ratio)+1
        if cate not in cate_item_sort:
            continue
        recom_list=cate_item_sort[cate][:num]
        print(recom_list)
        recom_result[userid]+=recom_list
    return recom_result

def run_main():
    ave_score=read.get_ave_score("../data/ratings.csv")
    item_cate,cate_item_sort=read.get_item_cate(ave_score,"../data/movies.csv")
    #print(item_cate)
    #print(cate_item_sort)
    up=get_up(item_cate,"../data/ratings.csv")
    print(len(up))
    print(recom(cate_item_sort,up,"609"))

if __name__=="__main__":
    run_main()