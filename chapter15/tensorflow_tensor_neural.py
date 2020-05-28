#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/20 11:00
#@Author  :Zhang Yueping
#@FileName: tensorflow_tensor_neural.py

#@Software: PyCharm
import tensorflow as tf
import numpy as np
# x = tf.Variable([[0.4,0.2,0.4]])
# w = tf.Variable([[-0.5,-0.2],[-0.3,0.4],[-0.5,0.2]])
# b = tf.Variable([[0.1,0.2]])
#随机生成权重矩阵w,b
# w = tf.Variable(tf.random_normal([3,2]))
# b = tf.Variable(tf.random_normal([1,2]))
#
# #以placeholder传值
# x = tf.placeholder('float',[None,3])
#
# # x = tf.Variable([[0.4,0.2,0.4]])
# xwb = tf.matmul(x,w)+b
# # y = tf.nn.relu(tf.matmul(x,w)+b)
# y = tf.nn.sigmoid(xwb)
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     #执行计算图时，placeholder x以feed_dict传入x_array
#     x_array = np.array([[0.4,0.2,0.4],[1,5,7],[7,5,9]])
#     _b,_w,_x,_y = sess.run((b,w,x,y),feed_dict={x:x_array})
#     print('w=',_w)
#     print('b=',_b)
#
#     print('y=',_y)

def layer(output_dim,input_dim,inputs,activation=None):
    w = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    xwb = tf.matmul(inputs, w) + b
    if activation is None:
        outputs = xwb
    else:
        outputs = activation(xwb)
    return outputs

def layer_debug(output_dim,input_dim,inputs,activation=None):
    #比上一个函数多传回权重矩阵w,b
    w = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    xwb = tf.matmul(inputs, w) + b
    if activation is None:
        outputs = xwb
    else:
        outputs = activation(xwb)
    return outputs,w,b

#使用layer层建立3层神经网络
#输入层
x = tf.placeholder('float',[None,4])
#隐藏层
h,w1,b1 = layer_debug(3,4,x,activation=tf.nn.relu)
#输出层
y,w2,b2 = layer_debug(2,3,h)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    x_array = np.array([[0.4,0.4,0.5,0.3]])
    (layer_x,layer_h,layer_y,w1,b1,w2,b2) = sess.run((x,h,y,w1,b1,w2,b2),feed_dict={x:x_array})
    print(x_array,layer_x,'\n')
    print(w1,b1,w2,b2)
    print(layer_h,'\n')
    print(layer_y)



