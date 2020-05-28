#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/4 16:16
#@Author  :Zhang Yueping
#@FileName: minist_1000.py

#@Software: PyCharm

#修改隐藏层单元数为1000，增加dropout避免过拟合

import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)
# 定义plot_image函数显示数字图像

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

class Data(object):
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def plot_image(self,image):
        fig = plt.gcf()
        fig.set_size_inches(2, 2)
        plt.imshow(image, cmap='binary')
        plt.show()

    def plot_images_labels_predictions(self,images,labels,prediction,idx,num = 10):
        #显示多个图像和标签
        #prediction显示序号为从idx开始的10张图片

        fig = plt.gcf()
        fig.set_size_inches(10,10)
        if num > 25:
            num = 25
        for i in range(0,num):
            ax = plt.subplot(3,5,i+5)   #将图像划分为三行五列，所处位置在第i+5个
            ax.imshow(images[idx],cmap='binary')
            title = 'label'+str(labels[idx])
            if len(prediction)>0:
                print(prediction[idx])
                title += ',predict='+str(prediction[idx])

            ax.set_title(title,fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        plt.show()

    def changeX(self):
        #将图像转换为一维向量
        x_train = self.x_train.reshape(60000,784).astype('float32')
        x_test = self.x_test.reshape(10000,784).astype('float32')

        # print(x_train.shape,x_test.shape)

        #将数字图像标准化(介于0-1之间),标签转为one-hot向量化
        x_train_norm = x_train/255
        x_test_norm = x_test/255
        return x_train_norm,x_test_norm

        # print(x_train_norm[0])

    def changeY(self):
        y_trainOnehot = np_utils.to_categorical(self.y_train)
        y_testOnehot = np_utils.to_categorical(self.y_test)
        return y_trainOnehot,y_testOnehot
        # print(y_testOnehot[0])

#构建模型
    def CreateModel(self,x_train_norm,y_trainOnehot,x_test_norm,y_testOnehot):
        model = Sequential()
        model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
        print('model.summary()',model.summary())  #查看摘要

        #定义训练方式
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        #开始训练
        train_history = model.fit(x=x_train_norm,y=y_trainOnehot,validation_split=0.2,
                                  epochs=10,batch_size=200,verbose=2)
        #评估模型准确率
        scores = model.evaluate(x_test_norm,y_testOnehot)
        print('accuracy=',scores)

        #预测
        prediction = model.predict_classes(x_test_norm)
        # print(prediction)

        #显示10项测试结果图像
        # self.plot_images_labels_predictions(self.x_test,self.y_test,prediction,300)

        #显示混淆矩阵
        print(pd.crosstab(self.y_test,prediction,rownames=['label'],colnames=['predict']))

        #建立真实值与测试值的对应
        df = pd.DataFrame({'label':self.y_test,'prediction':prediction})
        print(df[:2])
        print('\n\n\n')
        print(df)

        #查询真实值是5，但预测值为3数据
        print(df[(df.label == 5)&(df.prediction == 3)])


        return train_history

    def show_train_history(self,train_history,train,validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train history')
        plt.ylabel(train)
        plt.xlabel('epoch')
        plt.legend(['train','validation'],loc = 'upper left')
        plt.show()

if __name__ == '__main__':
    obj = Data()
    trainX,testX = obj.changeX()
    trainY,testY = obj.changeY()
    train_history = obj.CreateModel(trainX,trainY,testX,testY)
    obj.show_train_history(train_history,'acc','val_acc')

    # plot_image(x_train[5])
    # plot_images_labels_predictions(x_train,y_train,[1,5,6,2,3],0,5)
    # plot_images_labels_predictions(x_test,y_test,[1,5,6,2,3],0,5)