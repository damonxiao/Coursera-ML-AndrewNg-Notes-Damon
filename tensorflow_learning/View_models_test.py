#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:03:52 2019

@author: xiaofang
"""

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
savedir = "log/"

print_tensors_in_checkpoint_file(savedir+"linearmodel.cpkt",None, True,True)

W = tf.Variable(1.0, name="weight")
b = tf.Variable(2.0, name="bias")

# 放到一个字典里
saver = tf.train.Saver({'weight': W, 'bias': b})

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, savedir+"linearmodel.cpkt")
print_tensors_in_checkpoint_file(savedir+"linearmodel.cpkt",None, True,True)