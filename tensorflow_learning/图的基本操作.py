#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:35:39 2019

@author: xiaofang
"""

import numpy as np
import tensorflow as tf

'''
1 .创建图的方法
'''
c = tf.constant(0.0) #在默认图上

g = tf.Graph() # 创建图g

with g.as_default():
    c1 = tf.constant(0.0)
    print("c1.graph:",c1.graph)
    print("g:",g)
    print("c.graph:",c.graph)

g2 = tf.get_default_graph()
print("g2:", g2)

tf.reset_default_graph()
g3 = tf.get_default_graph()
print("g3:", g3)

'''
2. 获取张量tensor
'''

print("c1.name:", c1.name)
t = g.get_tensor_by_name(name="Const:0")
print ("t:", t)

'''
3. 获取节点操作OP
'''
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])

tensor1 = tf.matmul(a, b, name = "exampleop")
print("tensor1.name:",tensor1.name, "tensor1:",tensor1)

test = g3.get_tensor_by_name("exampleop:0")
print("test:", test)

print("tensor1.op.name", tensor1.op.name)
testop = g3.get_operation_by_name("exampleop")
print("testop:", testop)

with tf.Session() as sess:
    test = sess.run(test)
    print("test after run", test)
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print("test get_tensor_by_name:", test)
    
'''
4. 获取所有列表
'''
# 返回图中的操作节点列表
tt2 = g.get_operations()
print("tt2:", tt2)

'''
5. 获取对象
'''
tt3 = g.as_graph_element(c1)
print("tt3:", tt3)
print("-------------------------\n")


#练习
with g.as_default():
  c1 = tf.constant(0.0)
  print(c1.graph)
  print(g)
  print(c.graph)
  g3 = tf.get_default_graph()
  print(g3)