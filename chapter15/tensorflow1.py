#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/17 17:13
#@Author  :Zhang Yueping
#@FileName: tensorflow1.py
#@Software: PyCharm

import tensorflow as tf
import numpy as np

# w = tf.Variable(tf.random_normal([3,2]),name='w')
# b = tf.Variable(tf.random_normal([1,2]),name='b')
# x=tf.placeholder('float',[None,3],name='x')
# y = tf.nn.sigmoid(tf.matmul(x,w)+b,'y')
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     x_array = np.array([[0.4,0.2,0.4],[0.3,0.4,0.5],[0.3,0.4,0.5]])
#     (_b,_w,_x,_y) = sess.run((b,w,x,y),feed_dict={x:x_array})
#     print(_b,_w,_x,_y)
#
#
# ts_c = tf.constant(2,name='ts_c')
# ts_x = tf.Variable(ts_c+5,name='ts_x')
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(ts_c)
#     print('ts_c=',sess.run(ts_c))
#     print('ts_x=',sess.run(ts_x))


width = tf.placeholder('int32',name='w')
height = tf.placeholder('int32',name='height')
area = tf.multiply(width,height,name='area')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area=',sess.run(area,feed_dict={width:6,height:8}))


tf.summary.merge_all()  #将所有要显示在tensorbord的数据整合
train_writer = tf.summary.FileWriter('log/area',sess.graph)  #写入文件
print(train_writer)
