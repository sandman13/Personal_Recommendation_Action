#-*- coding:utf-8 -*-
import os
def get_item_info(input_file):
  '''
  得到item的信息
  :param input_file:
  :return: a dict: key:itemid  value:[title,genre]
  '''
  if not os.path.exists(input_file):
      return {}
  linenum=0
  item_info={}
  fp=open(input_file, encoding='utf-8')
  for line in fp:
      if linenum==0:
          linenum+=1
          continue
      item=line.strip().split(',')
      if len(item)<3:   #观察数据集，发现一般由三列组成，但是有些电影title包含逗号，需要分开讨论
          continue;     #小于3，则直接过滤
      elif len(item)==3:
          itemid,title,genre=item[0],item[1],item[2]
      elif len(item)>3:
          itemid=item[0]
          genre=item[-1]
          title=','.join(item[1:-1])  #还原包含逗号被误分割的title
      item_info[itemid]=[title,genre]
  fp.close()
  return item_info


def get_ave_score(input_file):
    '''
    获取item的平均评分
    :param input_file: user rating file
    :return:
    '''
    if not os.path.exists(input_file):
        return {}
    linenum=0
    record_dict={}
    score_dict={}
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(',')
        if len(item)<4:
            continue;
        userid,itemid,rating=item[0],item[1],item[2]
        if itemid not in record_dict:
            record_dict[itemid]=[0,0]
        record_dict[itemid][0]+=1   #记录出现的次数
        record_dict[itemid][1]+=float(rating)
    fp.close()
    for itemid in record_dict:
        score_dict[itemid]=round(record_dict[itemid][1]/record_dict[itemid][0],3)  #精度为3
    return score_dict


def get_train_data(input_file):
    '''
    获得训练样本
    :param input_file:
    :return: list[user,item,label]
    '''
    if not os.path.exists(input_file):
        return []
    score_dict=get_ave_score(input_file)
    #负采样要保证正负样本均衡
    pos_dict={}
    neg_dict={}
    train_data=[]
    score_thr=4
    fp=open(input_file)
    linenum=0
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(',')
        if len(item)<4:
            continue
        userid,itemid,rating=item[0],item[1],float(item[2])
        if userid not in pos_dict:
            pos_dict[userid]=[]
        if userid not in neg_dict:
            neg_dict[userid]=[]
        if rating>=score_thr:        #大于阙值则看作正样本
            pos_dict[userid].append((itemid,1))
        else:
            score=score_dict.get(itemid,0)
            neg_dict[userid].append((itemid,score))
    fp.close()
    for userid in pos_dict:
            data_num=min(len(pos_dict[userid]),len(neg_dict.get(userid,[])))
            if data_num>0:
                train_data+=[(userid,temp[0],temp[1])for temp in pos_dict[userid]][:data_num]
            else:
                continue
            #对负样本按照平均评分进行排序，element是[itemid,score]
            sorted_neg_list=sorted(neg_dict[userid],key=lambda element:element[1],reverse=True)[:data_num]
            train_data+=[(userid,temp[0],0)for temp in sorted_neg_list]
    return train_data


if __name__=='__main__':
    '''
    item_dict=get_item_info("../data/movies.csv")
    print(len(item_dict))
    print(item_dict['11'])

    score_dict=get_ave_score("../data/ratings.csv")
    print(len(score_dict))
    print(score_dict['1'])
    '''
    train_data=get_train_data("../data/ratings.csv")
    print(len(train_data))
    print(train_data[:-20])