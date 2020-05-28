#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/20 10:46
#@Author  :Zhang Yueping
#@FileName: tensor2.py

#@Software: PyCharm

#建立一维与二维张量

import tensorflow as tf

# ts_x = tf.Variable([0.4,0.2,0.4])
ts_x = tf.Variable([[0.4,0.2,0.4],[1,5,8]])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(ts_x)
    x = sess.run(ts_x)
    print(x.shape)

#矩阵基本运算
x = tf.Variable([[1.,1.,1.]])
w = tf.Variable([[1.,1.,4.],[2.,5.,9.],[7.,5.,3.]])
xw = tf.matmul(x,w)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    xw = sess.run(xw)
    print(xw,xw.shape)