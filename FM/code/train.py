#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

from FM.code import read

def batcher(X_, y_, batch_size=-1):

    assert X_.shape[0] == len(y_)

    n_samples = X_.shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = y_[i:upper_bound]
        yield(ret_x, ret_y)

def train():
    x_train,x_test,y_train,y_test=read.load_dataset()
    #print(x_train.shape)

    vec_dim=10
    batch_size=1000
    epochs=10
    learning_rate=0.001
    sample_num,feature_num=x_train.shape
    x = tf.placeholder(tf.float32, shape=[None, feature_num], name="input_x")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="ground_truth")

    w0 = tf.get_variable(name="bias", shape=(1), dtype=tf.float32)
    W = tf.get_variable(name="linear_w", shape=(feature_num), dtype=tf.float32)
    V = tf.get_variable(name="interaction_w", shape=(feature_num, vec_dim), dtype=tf.float32)

    linear_part = w0 + tf.reduce_sum(tf.multiply(x, W), axis=1, keep_dims=True)
    interaction_part = 0.5 * tf.reduce_sum(tf.square(tf.matmul(x, V)) - tf.matmul(tf.square(x), tf.square(V)), axis=1,
                                           keep_dims=True)
    y_hat = linear_part + interaction_part
    loss = tf.reduce_mean(tf.square(y - y_hat))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            step = 0
            print("epoch:{}".format(e))
            for batch_x, batch_y in batcher(x_train, y_train, batch_size):
                sess.run(train_op, feed_dict={x: batch_x, y: batch_y.reshape(-1, 1)})
                step += 1
                if step % 10 == 0:
                    for val_x, val_y in batcher(x_test, y_test):
                        train_loss = sess.run(loss, feed_dict={x: batch_x, y: batch_y.reshape(-1, 1)})
                        val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y.reshape(-1, 1)})
                        print("batch train_mse={}, val_mse={}".format(train_loss, val_loss))

        for val_x, val_y in batcher(x_test, y_test):
            val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y.reshape(-1, 1)})
            print("test set rmse = {}".format(np.sqrt(val_loss)))

if __name__=='__main__':
    train()