#-*-coding:utf-8 -*-
from itertools import count
from collections import defaultdict
from scipy.sparse import csr
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer

cols=['user','item','rating','timestamp']
train=pd.read_csv("../data/ua.base",delimiter='\t',names=cols)
test=pd.read_csv("../data/ua.test",delimiter='\t',names=cols)
train=train.drop(['timestamp'],axis=1)
test=test.drop(['timestamp'],axis=1)

#由于DictVectorizer会自动的将数字识别为连续特征，所以这里将用户和物品的id表示为非数字形式
train['item']=train['item'].apply(lambda x:"c"+str(x))
train['user']=train['user'].apply(lambda x:"u"+str(x))

test['item']=test['item'].apply(lambda x:"c"+str(x))
test['user']=test['user'].apply(lambda x:"u"+str(x))

data=pd.concat([train,test])
print("all data shape is:",data.shape)
print(data.head(2))

vec=DictVectorizer()
vec.fit_transform(data.to_dict(orient='record')) #将特征转化为one-hot向量

x_train=vec.transform(train.to_dict(orient='record')).toarray()
x_test=vec.transform(test.to_dict(orient='record')).toarray()

print("x_train shape is:",x_train.shape)
print("x_test shape is:",x_test.shape)
print(x_train[0])

y_train=train['rating'].values.reshape(-1,1)   #这里-1表示没有指定具体的行数，1表示列数为1
y_test=test['rating'].values.reshape(-1,1)

print("y_train shape is:",y_train.shape)
print("y_test shape is:",y_test.shape)

n,p=x_train.shape
print("n=",n)
print("p=",p)

k=40

x=tf.placeholder('float',[None,p])
y=tf.placeholder('float',[None,1])

w0=tf.Variable(tf.zeros([1]))     #一维数组里放一个值

w=tf.Variable(initial_value=tf.random_normal(shape=[p],mean=0,stddev=0.1))   #生成p个随机数

v=tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))   #生成k*p个随机数

#线性部分
linear_part=tf.add(w0,tf.reduce_sum(tf.multiply(w,x),1,keep_dims=True))  #求和，1是指定的axis

#交叉特征部分,tf.substract表示x-y；tf.matmul表示矩阵相乘，而tf.multiply表示两个矩阵中对应元素相乘
cross_part=0.5*tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(x,tf.transpose(v)),2),
                                         tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))),axis=1,keep_dims=True)


#预测值
predictions=tf.add(linear_part,cross_part)

#正则化项
lambda_w=tf.constant(0.001,name='lambda_w')
lambda_v=tf.constant(0.001,name='lambda_v')
l2_norm=0.5*tf.reduce_sum(tf.add(tf.multiply(lambda_w,tf.pow(w,2)),tf.multiply(lambda_v,tf.pow(v,2))))

error=tf.reduce_mean(tf.square(y-predictions))
loss=tf.add(error,l2_norm)

#梯度下降迭代优化
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)

epochs=10

init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        _,t=sess.run([optimizer,loss],feed_dict={x:x_train,y:y_train})
        validate_loss=sess.run(error,feed_dict={x:x_test,y:y_test})
        print("epoch:%d train loss:%f validate loss:%s" %(epoch,t,validate_loss))

    loss=sess.run(loss,feed_dict={x:x_test,y:y_test})
    print("loss:",loss)
    error=sess.run(error,feed_dict={x:x_test,y:y_test})
    RMSE=np.sqrt(error)
    print("RMSE:",RMSE)

    pre=sess.run(predictions,feed_dict={x:x_test[0:10]})
    print("predict:",pre)
