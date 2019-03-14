#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:04:12 2019

@author: xiaofang
"""


import tensorflow as tf
hello = tf.constant('Hello,TensorFlow!')

sess = tf.Session()
print(sess.run(hello))
sess.close()

a = tf.constant(3)
b = tf.constant(4)

with tf.Session() as sess:
    print ("相加: %i" % sess.run(a+b))
    print ("相乘: %i" % sess.run(a*b))

# 占位符placeholder使用
c = tf.placeholder(tf.int16)
d = tf.placeholder(tf.int16)

add = tf.add(c, d)
mul = tf.multiply(c, d)
divide = tf.divide(c, d)
with tf.Session() as sess:
    print("相加 : %i" % sess.run(add, feed_dict={c:3, d:4}))
    print("相乘 : %i" % sess.run(mul, feed_dict={c:3, d:4}))
    print("相除 : %i" % sess.run(divide, feed_dict={c:12, d:4}))
    print(sess.run([add,mul,divide], feed_dict={c:12, d:4}))