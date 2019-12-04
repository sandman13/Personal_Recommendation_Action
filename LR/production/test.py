#-*-coding:utf-8 -*-


if __name__=="__main__":
    fp=open("../data/lr_coef.txt")   #测试模型参数输出是否正确，应该是包含117个参数
    count=0
    for line in fp:
        item=line.strip().split(",")
        print(len(item))