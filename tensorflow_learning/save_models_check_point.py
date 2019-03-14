#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:13:01 2019

@author: xiaofang
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = {"batchsize":[], "loss":[]}

''' 1 生成模拟数据 '''
train_X = np.linspace(-1, 1, 100) # 生成100个-1到1之前的数据点
print("train_X", train_X)
train_Y = 2 * train_X + np.random.randn(100) * 0.3 # y=2*x,加入了一个random随机数作为噪声

# 显示模拟数据点
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.legend()
plt.show()

# 初始化图
tf.reset_default_graph()

''' 2 搭建模型 '''
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
z = tf.multiply(X, W) + b

# 反向传播更新
# reduce_mean函数的作用是求平均值square是求平方这里的square(Y-z)就是求Y-z的误差平方值
cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01

# 使用TF的梯度下降优化器设定的学习率不断优化W和b使用代价函数cost最小,从而达到z与Y的误差最小化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

''' 3 迭代训练模型 '''
init = tf.global_variables_initializer()

training_epochs = 20 # 20轮
display_step = 2
saver = tf.train.Saver(max_to_keep=1)
savedir = "log/"

with tf.Session() as sess:
    sess.run(init)
    
    # 训练模型
    for epoch in range(training_epochs):
        for(x,y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x, Y:y}) # 使用x,y替换占位符X,Y
        
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch + 1, "loss=", loss, "W=", sess.run(W), "b=", sess.run(b))
            if not (loss == 'NA'):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
            saver.save(sess, savedir + "linearmodel.cpkt", global_step=epoch) # 保存模型到linearmodel.cpkt
            
    print("Finished!")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y:train_Y}),"W=", sess.run(W), "b=", sess.run(b))
    
    # 图形显示
    plt.plot(train_X, train_Y, 'ro', label="Original data")
    plt.plot(train_X, sess.run(W)*train_X + sess.run(b), label='Fitted linear regression')
    plt.legend()
    plt.show()
    
    def moving_average(a, w=10):
        print("a =", a, "w=", w)
        if len(a) < w:
            return a[:]
        for idx,val in enumerate(a):
            print("idx:",idx,"val:",val, "sum(a[(idx-w):idx])/w ", (sum(a[(idx-w):idx])/w))
        return [val if idx <w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]
    
    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel("Minibatch number")
    plt.ylabel("Loss")
    plt.title("Minibatch run vs. Training loss")
    plt.show()

#重启一个session    
load_epoch=18    
with tf.Session() as sess2:
    sess2.run(tf.global_variables_initializer())     
    saver.restore(sess2, savedir+"linearmodel.cpkt-" + str(load_epoch))
    print ("x=0.2，z=", sess2.run(z, feed_dict={X: 0.2}))
    
with tf.Session() as sess3:
    sess3.run(tf.global_variables_initializer()) 
    ckpt = tf.train.get_checkpoint_state(savedir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess3, ckpt.model_checkpoint_path)
        print ("x=0.2，z=", sess3.run(z, feed_dict={X: 0.2}))

with tf.Session() as sess4:
    sess4.run(tf.global_variables_initializer()) 
    kpt = tf.train.latest_checkpoint(savedir)
    if kpt!=None:
        saver.restore(sess4, kpt) 
        print ("x=0.2，z=", sess4.run(z, feed_dict={X: 0.2}))

