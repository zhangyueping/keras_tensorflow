#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/25 12:03
#@Author  :Zhang Yueping
#@FileName: tensorflow_mnist_cnn.py

#@Software: PyCharm
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

#建立共享参数
def weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.1),name='w')

def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape),name='b')

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#输入层
with tf.name_scope('input_layer'):
    x = tf.placeholder('float',shape=[None,784],name='x')
    x_image = tf.reshape(x,[-1,28,28,1])

#卷积层
with tf.name_scope('c1_conv'): #28*28*16
    w1 = weights([5,5,1,16])
    b1 = bias([16])
    conv1 = conv2d(x_image,w1)+b1
    c1_conv = tf.nn.relu(conv1)

#池化层
with tf.name_scope('c1_pool'):  #14*14*16
    c1_pool = max_pool_2x2(c1_conv)

#建立卷积层2
#卷积过程中，输入层有多少个通道，滤波器就有多少个通道，
#但是滤波器的数量是任意的，滤波器的数量决定了卷积后 featuremap 的通道数

with tf.name_scope('c2_conv'):
    w2 = weights([5,5,16,36])
    b2 = bias([36])
    conv2 = conv2d(c1_pool,w2) + b2
    c2_conv = tf.nn.relu(conv2)
#池化层
with tf.name_scope('c2_pool'):
    c2_pool = max_pool_2x2(c2_conv)

#建立平坦层
with tf.name_scope('D_flat'):
    D_flat = tf.reshape(c2_pool,[-1,1764]) #36*7*7

#隐藏层
with tf.name_scope('D_hidden'):
    w3 = weights([1764,128])
    b3 = bias([128])
    D_hidden = tf.nn.relu(tf.matmul(D_flat,w3)+b3)
    D_hidden_dropout = tf.nn.dropout(D_hidden,keep_prob=0.8)

#输出层
with tf.name_scope('output_layer'):
    w4 = weights([128,10])
    b4 = bias([10])
    y_predict = tf.nn.softmax(tf.matmul(D_hidden_dropout,w4)+b4)


#定义训练方式
with tf.name_scope('optimizer'):
    y_label = tf.placeholder('float',shape=[None,10],name='y_label')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

#定义评估模型准确率的方式
with tf.name_scope('evaluate_model'):
    correct_predict = tf.equal(tf.argmax(y_predict,1),tf.argmax(y_label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,'float'))

#给定参数
trainEpoches = 10
batchsize = 100
totalBatchs = int(mnist.train.num_examples/batchsize)
epoch_list = []
accuracy_list = []
loss_list = []
from time import time
start = time()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#训练
for epoch in range(trainEpoches):
    for i in range(totalBatchs):
        batch_x,batch_y = mnist.train.next_batch(batchsize)
        sess.run(optimizer,feed_dict={x:batch_x,y_label:batch_y})
    loss,acc = sess.run([loss_function,accuracy],feed_dict={x:mnist.validation.images,y_label:mnist.validation.labels})

    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)

    print('train epoch:', epoch + 1, 'loss=', loss, 'accuracy=', acc)
duration = time() - start
print('time:',duration)

import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list,accuracy_list,label = 'acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc'],loc='upper left')
plt.show()

#评估模型准确率
print('accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images,y_label:mnist.test.labels}))
#进行预测
prediction_result = sess.run(tf.argmax(y_predict,1),feed_dict={x:mnist.test.images})
print(prediction_result[:10])


#启动tensorbord

merged = tf.summary.merge_all()  #merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
train_writer = tf.summary.FileWriter('log/CNN',sess.graph) #写入文件
