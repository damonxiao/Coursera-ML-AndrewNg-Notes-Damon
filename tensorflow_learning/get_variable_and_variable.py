#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:26:19 2019

@author: xiaofang
"""

import tensorflow as tf

tf.reset_default_graph()

var1 = tf.Variable(1.0, name="firstvar")
print("var1:", var1.name)
var1 = tf.Variable(2.0, name="firstvar")
print("var1 2.0:", var1.name)
var1 = tf.Variable(3.0, name="firstvar")
print("var1 3.0:", var1.name)



var2 = tf.Variable(3.0)
print("var2:", var2.name)
var2 = tf.Variable(4.0)
print("var2 4.0:", var2.name)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("var1=", var1.eval())
    print("var2=", var2.eval())
    
get_var1 = tf.get_variable("firstvar", [1], initializer=tf.constant_initializer(0.3))
print("get_var1:", get_var1.name)
# get_variable("firstvar"..)执行会报错,使用get_variable一个变量名不能重复使用,若使用tf.Variable函数来初始化
# 一个变量名可以重复使用,系统会自动给名字加后缀
'''
ValueError: Variable firstvar already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:
'''
#get_var1 = tf.get_variable("firstvar", [1], initializer=tf.constant_initializer(0.3))
#print("get_var1:", get_var1.name)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("get_var1=", get_var1.eval())