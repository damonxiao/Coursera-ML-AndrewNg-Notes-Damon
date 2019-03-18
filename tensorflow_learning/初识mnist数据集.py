#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:35:06 2019

@author: xiaofang
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

print("输入train数据:",mnist.train.images)
print("输入train数据shape:",mnist.train.images.shape)

import pylab # 图形打印类,见python

im = mnist.train.images[1]
im = im.reshape(-1,28)#将图由一行784个像素，转换成28*28像素的图以利于打印
pylab.imshow(im)
pylab.show()

print("test数据打印shape:", mnist.test.images.shape)
print("validation数据打印shape:", mnist.validation.images.shape)