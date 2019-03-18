#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:59:59 2019

@author: xiaofang
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()

# 模型
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 模型参数
W = tf.Variable(tf.random_normal([784, 10])) # 正态分布的[784,10]的矩阵
b = tf.Variable(tf.zeros([10]))
print("b:", b, "W.shape:", W.shape)

# 正向传播的定义
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# cost, 反向传播 的定义 ,将生成的pred与样本标签y进行一次交叉熵运算最小化误差cost
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 参数 学习率 
learning_rate = 0.01

# 梯度下降优化cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 训练参数
training_epochs = 100
display_step = 1
batch_size = 100

saver = tf.train.Saver()
model_path = "log/mnist_model/mnist_model.ckpt"

#启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)#每一轮训练多少批次
        
        # 遍历全部数据集
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
             # 计算平均值以使误差值更加平均
            avg_cost += c / total_batch
#            print("I:",i,"eopch:", epoch +1, "avg_cost:", avg_cost)
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Eopch:", epoch, "avg_cost:", "{:.9f}".format(avg_cost))
    
    print("Train finished!")
    
    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确 率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
    # 存储模型 
    save_path = saver.save(sess, model_path)
    print("Model saved in file:%s" % save_path)
    
    
import pylab
#读取模型 
print("Starting 2nd session...")
with tf.Session() as sess2:
    # 初始化变量
    sess2.run(tf.global_variables_initializer())
    # 恢复模型
    saver.restore(sess2, model_path)
    
    # 测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    output = tf.argmax(pred, 1)
    test_range = 10
    batch_xs, batch_ys = mnist.train.next_batch(test_range) # 返回 两个手写数字图片
    outputval, predv = sess2.run([output, pred], feed_dict={x:batch_xs})
    print("outputval:", outputval, "predv:", predv, "batch_ys:", batch_ys)
    
    for i in range(test_range):
        im = batch_xs[i]
        im = im.reshape(-1, 28)
        pylab.imshow(im)
        pylab.show()