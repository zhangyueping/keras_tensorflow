#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/5 10:12
#@Author  :Zhang Yueping
#@FileName: keras_CIFAR10.py

#@Software: PyCharm

from keras.datasets import cifar10
import numpy as np
import pandas as pd
np.random.seed(10)
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import  Conv2D,MaxPooling2D,ZeroPadding2D

class Cifra10(object):

    def __init__(self):
        (self.x_train,self.y_train),(self.x_test,self.y_test) = cifar10.load_data()
        self.label_dict = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
                     5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

    def plot_images_prediction(self,images,labels,prediction,idx,num=10):
        #显示图片
        fig = plt.gcf()
        fig.set_size_inches(12,14)
        if num > 25:
            num = 25
        for i in range(0,num):
            ax = plt.subplot(5,5,1+i)
            ax.imshow(images[idx],cmap = 'binary')

            title = str(i)+','+self.label_dict[labels[i][0]]
            if len(prediction)>0:
                title +='=>'+self.label_dict[prediction[i]]

            ax.set_title(title,fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        plt.show()


    def changeX(self):
        #处理数据
        #x_train 维度 [50000,32,32,3]
        x_train_norm = self.x_train.astype('float32')/255.0
        x_test_norm = self.x_test.astype('float32')/255.0

        print(x_train_norm[0][0][0])
        return x_train_norm,x_test_norm

    def changeY(self):
        print(self.y_train[:5])
        y_trainHot = np_utils.to_categorical(self.y_train)
        y_testHot = np_utils.to_categorical(self.y_test)

        print(y_trainHot[:5])
        return y_trainHot,y_testHot

    def model(self,x_train_norm,y_trainHot,x_test_norm,y_testHot):
        model = Sequential()
        model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32,32,3),
                         activation='relu',padding='same'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())
        model.add(Dropout(0.5))

        model.add(Dense(1024,activation='relu'))
        model.add(Dropout(0.25))

        model.add(Dense(10,activation='softmax'))

        print(model.summary())

        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        try:
            model.load_weights('SaveModel/cifarCnnModel.h5')
            print('模型加载成功，继续训练模型！')
        except:
            print('模型加载失败，开始训练新模型')

        #训练数据
        model.fit(x_train_norm,y_trainHot,batch_size=128,epochs=5,verbose=1,validation_split=0.2)
        model.save_weights('SaveModel/cifarCnnModel.h5')
        print('saved model to disk')
        scores = model.evaluate(x_test_norm,y_testHot,verbose=0)
        print(scores)

        #预测
        prediction = model.predict_classes(x_test_norm)
        print(prediction[:10])

        #显示前10项预测结果
        self.plot_images_prediction(self.x_test,self.y_test,prediction,0,10)

        #查看预测概率
        prediction_probability = model.predict(x_test_norm)
        print('第0张的概率：',predict_probality[0])

        #显示混淆矩阵，确保都是一维数组
        print('混淆矩阵检测',prediction.shape,self.y_test.shape)

        pd.crosstab(self.y_test.reshape(-1),prediction,rownames=['label'],colnames=['predict'])


        return prediction,prediction_probability


    def show_probality(self,prediction,prediction_probality,i):
        print('label:',self.label_dict[self.y_test[i][0]],'predict:',self.label_dict[prediction[i]])

        plt.figure(figsize=(2,2))
        plt.imshow(np.reshape(self.x_test[i],(32,32,3)))
        plt.show()

        for j in range(10):
            print(self.label_dict[j]+'probality:%1.9f')%(prediction_probality[i][j])



if __name__ == '__main__':
    a = Cifra10()
    # print(a.x_train[0],a.x_train.shape,a.y_train.shape,a.x_test.shape,a.y_test[0])
    # a.plot_images_prediction(a.x_train,a.y_train,[],10)
    x_train_norm,x_test_norm = a.changeX()
    y_trainHot,y_testHot = a.changeY()

    predict,predict_probality = a.model(x_train_norm,y_trainHot,x_test_norm,y_testHot)
    #查看第三项预测的概率
    a.show_probality(predict,predict_probality,3)
