#-*- coding:utf-8 -*-
import re


def get_feature_num(input_file):
    fp = open(input_file)  # 测试模型参数输出是否正确，应该是包含118个参数
    for line in fp:
        line=str(line)
        count=re.sub("\D","",line)
        print(count)
        return count

if __name__=='__main__':
    get_feature_num("../data/feature_num.txt")