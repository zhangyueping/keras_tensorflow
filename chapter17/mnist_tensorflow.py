#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/25 8:57
#@Author  :Zhang Yueping
#@FileName: mnist_tensorflow.py

#@Software: PyCharm

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

print('train',mnist.train.num_examples,'validation',mnist.validation.num_examples,'test',mnist.test.num_examples)
print('train images',mnist.train.images.shape,'labels',mnist.train.labels.shape)

#绘图
def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='binary')
    plt.show()



#查看多项训练数据
def plot_images_labels_prediction(images,labels,prediction,idx,num = 10):

    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25:
        num = 25
    for i in range(0,num):
        ax = plt.subplot(5,5,1+i)
        ax.imshow(np.reshape(images[idx],(28,28)),cmap = 'binary')

        title = 'label='+str(np.argmax(labels[idx]))
        if len(prediction) > 0:
            title += 'prediction='+str(prediction[idx])
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()




if __name__ == '__main__':
    plot_image(mnist.train.images[0])
    print(mnist.train.labels[0])
    print(np.argmax(mnist.train.labels[0]))
    plot_images_labels_prediction(mnist.train.images, mnist.train.labels, [], 0)

    #批次读取mnist数据
    batch_images_xs,batch_labels_ys = mnist.train.next_batch(batch_size=100)
    print(len(batch_images_xs),len(batch_labels_ys))
    plot_images_labels_prediction(batch_images_xs,batch_labels_ys,[],0)