#-*-coding:utf-8-*-

import os
import numpy as np
import operator
def load_item_vec(input_file):
    '''
    将word2vec生成的item的向量进行整理
    :param input_file:
    :return: dict key:itemid value:vec
    '''
    if not os.path.exists(input_file):
        return {}
    linenum=0
    item_vec={}
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split()
        if len(item)<129:   #设置向量的长度为128
            continue
        itemid=item[0]
        if itemid=='</s>':    #过滤换行符
            continue
        item_vec[itemid]=np.array([float(ele)for ele in item[1:]])
    fp.close()
    return item_vec

def cal_item_sim(item_vec,itemid,output_file):
    '''

    :param item_vec:
    :param itemid:
    :param output_file:
    :return:
    '''
    if itemid not in item_vec:
        return
    topK=30
    score={}
    fix_item_vec=item_vec[itemid]
    for tmp_itemid in item_vec:
        if tmp_itemid ==itemid:
            continue
        tmp_itemvec=item_vec[tmp_itemid]
        denominator=np.linalg.norm(fix_item_vec)*np.linalg.norm(tmp_itemvec)   #计算分母
        if denominator==0:
            score[tmp_itemid]=0
        else:
            score[tmp_itemid]=round(np.dot(fix_item_vec,tmp_itemvec)/denominator,3)
    tmp_list=[]
    fw=open(output_file,"w+")
    outstr=itemid+"\t"
    for instance in sorted(score.items(),key=operator.itemgetter(1),reverse=True)[:topK]:
        tmp_list.append(instance[0]+"_"+str(instance[1]))
    outstr+=";".join(tmp_list)
    fw.write(outstr+"\n")
    fw.close()

def run_main(input_file,output_file):
    '''

    :param input_file:
    :param output_file:
    :return:
    '''
    item_vec=load_item_vec(input_file)
    cal_item_sim(item_vec,"27",output_file)


if __name__=="__main__":
    run_main("../data/item_vec.txt","../data/sim_result.txt")