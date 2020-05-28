#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/26 21:56
#@Author  :Zhang Yueping
#@FileName: test_gpu.py

#@Software: PyCharm

import tensorflow as tf
import time

# with tf.device('/cpu:0'):
#     size = 500
#     w = tf.random_normal([size,size],name='w')
#     x = tf.random_normal([size,size],name='x')
#     mul = tf.matmul(w,x,name='mul')
#     sum_result = tf.reduce_mean(mul,name='sum')
#
# #指定cpu或gpu设备
#
#
#
# #配置 显示设备信息
# tfconfig = tf.ConfigProto(log_device_placement=True)
# with tf.Session(config=tfconfig) as sess:
#     result = sess.run(sum_result)
# print('result=',result)

#测试cpu与gpu性能
def performanceTest(device_name,size):
    with tf.device(device_name):
        w = tf.random_normal([size, size], name='w')
        x = tf.random_normal([size, size], name='x')
        mul = tf.matmul(w, x, name='mul')
        sum_result = tf.reduce_mean(mul, name='sum')
    starttime = time.time()
    tfconfig = tf.ConfigProto(log_device_placement=True)
    with tf.Session(config=tfconfig) as sess:
        result = sess.run(sum_result)
    durationtime = time.time() - starttime
    print('device_name',device_name,'time',durationtime,'result=', result)
    return durationtime

# g = performanceTest('/gpu:0',100)
# c = performanceTest('/cpu:0',100)

#产生数据量与gpu,cpu运行速度关系
gpu_set = []
cpu_set = []
set = []
for i in range(0,5001,500):
    g = performanceTest('/gpu:0',i)
    c = performanceTest('/cpu:0',i)
    gpu_set.append(g)
    cpu_set.append(c)
    set.append(i)
print(set,gpu_set,cpu_set)

# set = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# gpu_set = [4.593356132507324, 0.2968475818634033, 0.4061906337738037, 0.5468018054962158, 0.6562261581420898, 0.7655751705169678, 0.921825647354126, 1.0624492168426514, 1.218670129776001,
#  1.328040599822998, 1.4842720031738281]
# cpu_set = [0.24997782707214355, 0.3593177795410156, 0.5312037467956543, 0.6561720371246338
# , 0.7967760562896729, 0.9998891353607178, 1.2186172008514404, 1.499847650527954, 1.843571662902832, 2.202908992767334, 2.4841372966766357]

import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(6,4)
plt.plot(set,gpu_set,label = 'gpu')
plt.plot(set,cpu_set,label = 'cpu')
plt.legend()
plt.show()

