#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/25 10:06
#@Author  :Zhang Yueping
#@FileName: tensorflow_mnist_mlp.py

#@Software: PyCharm

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

#建立模型
def layer(output_dim,input_dim,inputs,activation=None):
    w = tf.Variable(tf.random_normal([input_dim,output_dim]))
    b = tf.Variable(tf.random_normal([1,output_dim]))
    xwb = tf.matmul(inputs,w)+b
    if activation is None:
        outputs = xwb
    else:
        outputs = activation(xwb)
    return outputs

x = tf.placeholder('float',[None,784])

#隐藏层
h1 = layer(output_dim=1000,input_dim=784,inputs=x,activation=tf.nn.relu)
#第二个隐藏层
h2 = layer(output_dim=1000,input_dim=1000,inputs=h1,activation=tf.nn.relu)
#建立输出层
y_predict = layer(output_dim=10,input_dim=1000,inputs=h2,activation=None) #为什么加上激活函数才更不好呢？


#定义训练方式
y_label = tf.placeholder('float',[None,10])
#定义损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_label))
#定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
#计算每一项数据是否预测准确,相等返回1，不相等返回0
correct_prediction = tf.equal(tf.argmax(y_label,1),tf.argmax(y_predict,1))
#计算正确结果的平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))


#开始训练
trainEpochs = 15
batchsize = 100
totalBatchs = int(mnist.train.num_examples/batchsize)
print('需要批次数：',totalBatchs)
loss_list = []
epoch_list = []
accuracy_list = []
from time import time
starttime = time()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

with tf.device('/gpu:1'):
    for epoch in range(trainEpochs):
        for i in range(totalBatchs):
            batchx,batchy = mnist.train.next_batch(batchsize)  #迭代器
            sess.run(optimizer,feed_dict={x:batchx,y_label:batchy})
            #验证数据集
        loss,acc = sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y_label:mnist.validation.labels})
        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)
        print('train epoch:',epoch+1,'loss=',loss,'accuracy=',acc)

duration = time()-starttime
print('train time:',duration)


#画出误差执行结果
# import matplotlib.pyplot as plt
# fig = plt.gcf()
# fig.set_size_inches(4,2)
# plt.plot(epoch_list,loss_list,label = 'loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss'],loc='upper left')
# plt.show()

#评估模型准确率
# print('accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images,y_label:mnist.test.labels}))
# #进行预测
# prediction_result = sess.run(tf.argmax(y_predict,1),feed_dict={x:mnist.test.images})
# print(prediction_result[:10])


