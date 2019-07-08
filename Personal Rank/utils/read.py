#-*- coding:utf-8 -*-
'''
get graph from data
'''
import os
def get_graph_from_data(input_file):
    '''

    :param input_file:ratings.csv
    :return: a dict : {userA{item1:1,item1:1},item1{userA:1}
    '''
    if not os.path.exists(input_file):
        return {}
    graph={}
    linenum=0
    score_thr=4.0
    fp=open(input_file)
    for line in fp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(',')
        if len(item)<3:
            continue
        userid,itemid,rating=item[0],"item_"+item[1],item[2]   #给itemid增加一个前缀
        if float(rating)<score_thr:
            continue
        if userid not in graph:
            graph[userid]={}
        graph[userid][itemid]=1
        if itemid not in graph:
            graph[itemid]={}
        graph[itemid][userid]=1
    fp.close()
    return graph

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

if __name__=='__main__':
     print(get_graph_from_data("../data/test.txt"))
