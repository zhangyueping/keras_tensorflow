#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/4 17:05
#@Author  :Zhang Yueping
#@FileName: CNN_mnist.py

#@Software: PyCharm

#数据预处理
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import pandas as pd
np.random.seed(10)
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import matplotlib.pyplot as plt


class CNN_mnist(object):
    def __init__(self):
        (self.x_train,self.y_train),(self.x_test,self.y_test) = mnist.load_data()

    def changeX(self):
        x_train = self.x_train.reshape(self.x_train.shape[0],28,28,1).astype('float32')
        x_test = self.x_test.reshape(self.x_test.shape[0],28,28,1).astype('float32')
        x_train_norm = x_train/255
        x_test_norm = x_test/255
        return x_train_norm,x_test_norm

    def changeY(self):
        y_trainHot = np_utils.to_categorical(self.y_train)
        y_testHot = np_utils.to_categorical(self.y_test)
        return y_trainHot,y_testHot

    def model(self,x_train_norm,x_test_norm,y_trainHot,y_testHot):
        model = Sequential()
        #将Conv2D用作模型中的第一层时，需要提供关键字参数input_shape
        #卷积层与池化层
        model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=1))
        model.add(Conv2D(filters=36,kernel_size=(5,5),padding='valid',activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        #全连接层
        model.add(Flatten())
        #隐藏层
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.5))
        #输出层
        model.add(Dense(10,activation='softmax'))

        print(model.summary())
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        try:
            model.load_weights('SaveModel/mnistCnnModel.h5')
            print('模型加载成功，继续训练模型！')
        except:
            print('模型加载失败，开始训练新模型')

        # 训练数据
        train_history = model.fit(x=x_train_norm, y=y_trainHot, batch_size=100, epochs=5, verbose=2,
                                  validation_split=0.2)
        model.save_weights('SaveModel/mnistCnnModel.h5')
        print('saved model to disk')

        # print(train_history)
        #评估准确率
        scores = model.evaluate(x_test_norm,y_testHot)
        print('scores',scores)

        #进行预测
        predict = model.predict_classes(x_test_norm)
        print(predict[:10])

        #显示混淆矩阵
        print(pd.crosstab(self.y_test,predict,rownames=['label'],colnames=['predict']))
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
    obj = CNN_mnist()
    x_train_norm,x_test_norm = obj.changeX()
    y_trainHot,y_testHot = obj.changeY()
    train_history = obj.model(x_train_norm,x_test_norm,y_trainHot,y_testHot)
    obj.show_train_history(train_history,'acc','val_acc')
