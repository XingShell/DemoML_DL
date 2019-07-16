'''产生数据以及数据评价功能'''

import numpy as np
from gensim import corpora,models,similarities
from sklearn.model_selection import train_test_split
from functools import reduce
import jieba
import re
import os
from util import evaluate



class Data_Load(object):
    def __init__(self,genNewVec=False):
        self.num_topics = 30
        self.dim = self.num_topics
        self.num_classes = 2
        self.batch_size = 5000
        self.label =[]
        self.data = []
        self.allnum = 0
        self.positive = 0 # the num of label is 1
        self.isLaborContract = True
        self.isInternshipContract = False
        self.trainDataDir = '../lib/'
        self.testDataDir = self.trainDataDir
        self.saveVecName = '%s/DataVec'%self.trainDataDir  # 固定维度数据保存Name
        self.genNewVec = genNewVec
        self.testsize = 0.25
        self.isSplit = True
        if self.genNewVec:
            self.genDataVec()
        else:
            pass
        if self.isLaborContract:
            self.load_label()
        if self.isInternshipContract:
            self.load_label_internships()
        else:
            pass
        self.readDataVec()
        if self.isSplit:
            x_train, x_test, y_train, y_test = train_test_split(self.data, self.label, test_size=self.testsize)
            self.x_train = np.array(x_train);self.y_train = np.array(y_train);self.x_test = np.array(x_test);self.y_test = np.array(y_test)

            self._num_examples = len(self.y_train)  # 训练样本数
        else:
            self._num_examples = self.allnum  # 训练样本数
        self._epochs_completed = 0  # 完成遍历轮数
        self._index_in_epochs = 0  # 调用next_batch()函数后记住上一次位置


    def readDataVec(self):
        data_tem = []
        with open(self.saveVecName, 'r') as f:
            for line in f:
                data_tem.append(line)
        self.allnum = len(data_tem)
        for i in range(len(data_tem)):
            el = data_tem[i].strip().split(' ')
            self.data.append([])
            for e in el:
                if len(el) < self.dim:
                    self.data[i].extend([0 for i in range(self.dim)])
                else:
                    self.data[i].append(float(e))

    #加载分类劳动合同标签库
    def load_label(self):
        path = '%s/标签库.txt'%self.trainDataDir
        with open(path,'r',encoding='utf-8') as f:
            for line in f:
                self.label.append(np.float32(line.split('\t')[1][0]))
        return self.label

    #加载分类实习合同标签库
    def load_label_internships(self):
        path = '%s/标签库_实习.txt'%self.trainDataDir
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.label.append(float(line.split('\t')[1][0]))
        return self.label

    def showPositiveRate(self):
        for i in range(len(self.label)):
            if self.label[i] == '0':
                pass
            if self.label[i] == '1':
                self.positive += 1
        print(self.positive/self.allnum)

    def allinsplit(self):
        return self.x_train, self.x_test, self.y_train, self.y_test
    def all(self):
        return np.array(self.data[:]), np.array(self.label[:])

    # def str2float(s):
    #     d = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    #     def c(i):
    #         return d[i]
    #     def fn(x, y):
    #         return x * 10 + y
    #     q = s.index('.')                # 遍历并返回 . 的位置
    #     s1 = s[0:q]
    #     s2 = s[q+1:]
    #     return reduce(fn, map(c, s1)) + reduce(fn, map(c, s2)) * (10**(-len(s2)))

    def genDataVec(self):
        dictionary = corpora.Dictionary.load(r'%s/contract.dict'%self.trainDataDir)
        corpus = corpora.MmCorpus(r'%s/contract.mm'%self.trainDataDir)
        tfidf = models.TfidfModel(corpus)
        lsi = models.LsiModel(tfidf[corpus], id2word=dictionary, num_topics=self.num_topics)
        # error??
        lsi.save(r'../lib/model.lsi')
        index = similarities.MatrixSimilarity(lsi[corpus]).save(r'../lib/lsi_model.index')
        with open(self.saveVecName, 'w') as f:
            for contract in corpus:
                for elem in lsi[contract]:
                    f.write(" "+str(elem[1]))
                f.write('\n')


    def shuffle(self):
        permutation = np.random.permutation(self._num_examples)
        if self.isSplit:
            self.x_train = self.x_train[permutation, :]
            self.y_train = self.y_train[permutation]
        else:
            self.data = self.data[permutation, :]
            self.labels = self.labels[permutation]

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epochs
        if self._epochs_completed == 0 and start == 0 and shuffle:
            self.shuffle()
        if self.isSplit:
            if start + batch_size > self._num_examples:
                self._epochs_completed += 1
                rest_num_examples = self._num_examples - start
                rest_part_data = self.x_train[start:self._num_examples]
                rest_part_labels = self.y_train[start:self._num_examples]
                if shuffle:
                    self.shuffle()
                start = 0
                self._index_in_epochs = batch_size - rest_num_examples
                end = self._index_in_epochs
                new_part_data = self.x_train[start:end]
                new_part_labels = self.y_train[start:end]
                return np.concatenate((rest_part_data, new_part_data), axis=0), np.concatenate(
                    (rest_part_labels, new_part_labels), axis=0)

            else:
                self._index_in_epochs += batch_size
                end = self._index_in_epochs
                return self.x_train[start:end], self.y_train[start:end]
        else:
            if start + batch_size > self._num_examples:
                self._epochs_completed += 1
                rest_num_examples = self._num_examples - start
                rest_part_data = self.data[start:self._num_examples]
                rest_part_labels = self.labels[start:self._num_examples]
                if shuffle:
                    self.shuffle()
                start = 0
                self._index_in_epochs = batch_size - rest_num_examples
                end = self._index_in_epochs
                new_part_data = self.data[start:end]
                new_part_labels = self.labels[start:end]
                return np.concatenate((rest_part_data, new_part_data), axis=0), np.concatenate(
                    (rest_part_labels, new_part_labels), axis=0)

            else:
                self._index_in_epochs += batch_size
                end = self._index_in_epochs
                return self.data[start:end], self.labels[start:end]
    def chooceall(self):
        DataisOne = []
        DataisZero = []
        for index, y in enumerate(self.label):
            if y == 1:
                DataisOne.append(self.data[index])
            else:
                DataisZero.append(self.data[index])
        permutation = np.random.permutation(len(DataisZero))
        DataisZero = np.array(DataisZero)
        DataisZero = DataisZero[permutation, :]
        len1 = len(DataisOne)
        # print(len1) # 663
        x = np.concatenate((DataisZero[:int(1.3*len1)], np.array(DataisOne)), axis=0)
        y = np.concatenate((np.zeros(int(1.3*len1)),np.ones(len(DataisOne))),axis=0)
        return x,y










if __name__ == "__main__":
    a = Data_Load(True)
    # a = Data_Load()
    # x,y = a.chooceall()
    # x,y = a.next_batch(100)
    # print(x[0])
    # a.genDataVec()
    # train_x,train_y,test_x,test_y =

