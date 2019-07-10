#-*-coding:utf-8-*-

import os
def produce_train_data(input_file,out_file):
    '''
    得到用户点击序列
    :param input_file:用户评分文件
    :param out_file: 写出文件
    :return:
    '''
    if not os.path.exists(input_file):
        return
    linenum=0
    record={}
    score_thr=3.0
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split('::')
        if len(item)<4:
            continue
        userid,itemid,rating=item[0],item[1],float(item[2])
        if rating<score_thr:
            continue
        if userid not in record:
            record[userid]=[]
        record[userid].append(itemid)
    fp.close()
    fw=open(out_file,'w')
    for userid in record:
        fw.write(" ".join(record[userid])+"\n")
    fw.close()


if __name__=='__main__':
    produce_train_data("../data/ratings.dat","../data/train_data.txt")