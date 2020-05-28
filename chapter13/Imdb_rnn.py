#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/5/10 8:38
#@Author  :Zhang Yueping
#@FileName: Imdb_rnn.py

#@Software: PyCharm

import urllib.request
import os
import tarfile
from keras.preprocessing import sequence  #将用于截长补短
from keras.preprocessing.text import Tokenizer  #用于建立字典
import re
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.layers.recurrent import SimpleRNN
import numpy as np
#下载数据

url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
filepath = r'E:\tensorflow+keras\keras-tensorflow-\chapter13\aclImdb.tar.gz'
if not os.path.isfile(filepath):
    result = urllib.request.urlretrieve(url,filepath)
    print('download:',result)
#解压
if not os.path.exists('aclImdb'):
    tfile = tarfile.open('aclImdb.tar.gz','r:gz')
    result = tfile.extractall(r'E:/tensorflow+keras/keras-tensorflow-/chapter13/')

sentimentDict = {1: '正面的', 0: '负面的'}
class Imdb_MLP(object):

    def rm_tags(self,text):
        re_tag = re.compile(r'<[^>]+>') #删除html的标签
        # return re_tag.sub('',text)
        return re.sub(re_tag,'',text)

    def read_files(self,filetypes):
        #将训练文件（测试文件）数据放在列表中
        path = r'E:/tensorflow+keras/keras-tensorflow-/chapter13/aclImdb/'
        file_list = []

        positive_path = path+filetypes+'/pos/'
        for f in os.listdir(positive_path):
            file_list+=[positive_path+f]

        negative_path = path + filetypes + '/neg/'
        for f in os.listdir(negative_path):
            file_list += [negative_path + f]

        print('read',filetypes,'files',len(file_list))

        all_labels = ([1]*12500 + [0]*12500)

        all_texts = []
        #将所有内容放在列表中
        for fi in file_list:
            with open(fi,encoding='utf-8')as file_input:
                all_texts += [self.rm_tags(' '.join(file_input.readlines()))]

        return all_labels,all_texts

    def tokens(self,train_texts,test_texts=None,texts1=None):
        #建立字典
        token = Tokenizer(num_words=3800)
        token.fit_on_texts(train_texts)
        #查看读取了多少篇文章
        # print(token.document_count)
        #查看word_index属性
        # print(token.word_index)
        if texts1:
            print('texts',texts1)
            texts_seq = token.texts_to_sequences(texts1)
            texts_ = sequence.pad_sequences(texts_seq, maxlen=380)
            print(texts_,len(texts_))
            return texts_[0]
        #将X转换为数字
        x_train_seq = token.texts_to_sequences(train_texts)
        x_test_seq = token.texts_to_sequences(test_texts)

        print(train_texts[0])
        print(x_train_seq[0])

        #将转换后的文章长度统一为100
        x_train = sequence.pad_sequences(x_train_seq,maxlen=380)
        x_test = sequence.pad_sequences(x_test_seq,maxlen=380)

        print('before',len(x_train_seq),'after',len(x_train))
        print('before', x_train_seq[1], 'after', x_train[1])

        return x_train,x_test

    def models(self,x_train,y_train):
        model = Sequential()
        #Embedding要求输入数据是整数编码的，每个字都用一个唯一的整数表示
        model.add(Embedding(output_dim=32,input_dim=3800,input_length=380))
        model.add(Dropout(0.35))
        #RNN层
        model.add(SimpleRNN(units=16))

        model.add(Dense(units=256,activation='relu'))
        model.add(Dropout(0.35))
        model.add(Dense(units=1,activation='sigmoid'))
        print('模型参数：',model.summary())

        #训练模型
        try:
            model = load_model('savemodel/cimdbmodel_rnn.h5')
            print('模型加载成功')
        except:
            print('模型加载失败，开始训练新模型')
            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
            train_history = model.fit(x_train,y_train,batch_size=100,epochs=5,verbose=2,validation_split=0.2)
            model.save('savemodel/cimdbmodel_rnn.h5')
            print('saved model to disk')
        return model

    def predict(self,model,x_test=None,y_test=None,textss=None):
        print(type(textss))

        if np.any(textss != None):
            predict = model.predict_classes(textss)
            return predict
        print('tytytyt', x_test.shape)
        scores = model.evaluate(x_test,y_test,verbose=1)
        print('scores:',scores)

        #预测
        predict = model.predict_classes(x_test)
        # print(predict[:10])
        predict_classes = predict.reshape(-1)
        # print(predict_classes)

        return predict_classes

    def display(self,test_texts,y_test,predict_classes,i):
        # 查看数据测试结果
        sentimentDict = {1: '正面的', 0: '负面的'}
        print(test_texts[i])
        print('真实值：',sentimentDict[y_test[i]],'预测值：',sentimentDict[predict_classes[i]])

    def preview(self,train_texts,input_texts,model):
        texts_seq = self.tokens(train_texts = train_texts, texts1 = [input_texts])
        print('电影评价长度：', texts_seq,texts_seq.shape)
        texts_seq = texts_seq.reshape(1,380)
        predict2 = self.predict(model=model,textss = texts_seq)
        print(sentimentDict[predict2[0][0]])



if __name__ == '__main__':
    obj = Imdb_MLP()
    y_train,train_texts = obj.read_files('train')
    y_test,test_texts = obj.read_files('test')
    # print('文章',train_texts[12501],y_train[12501],'\n')
    x_train, x_test = obj.tokens(train_texts,test_texts)   #转换为标准数字列表
    model = obj.models(x_train,y_train)  #得到模型

    #如果是标准语篇，直接使用
    predict = obj.predict(model,x_test=x_test,y_test=y_test)
    obj.display(test_texts,y_test,predict,30)

    # input_texts ="""A fabulous movie, I enjoyed every moment. So beautifully done that I would watch it again. It's a true musical as they used to be. I cried and laughed, it brought out many emotions. It's a great family film. The artistry and special effects make a great Disney style fantasy come to life. The music and songs were very pleasant in typical Disney fashion."""
    # obj.preview(train_texts,input_texts,model)