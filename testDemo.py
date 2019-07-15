from gensim import corpora,models,similarities
import jieba
import re
import os
import datetime
import numpy as np
import SVMDemo
import xgboostDemo



label = []
trainDataDir = 'TrainData'
#加载分类劳动合同标签库
def load_label():
    path = '%s/标签库.txt'%trainDataDir
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            label.append(np.float32(line.split('\t')[1][0]))
    return label

#加载分类实习合同标签库
def load_label_internships():
    path = '%s/标签库_实习.txt'%trainDataDir
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            label.append(float(line.split('\t')[1][0]))
    return label
#加载stopword
def loadstopword(path):
    stop_word = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n','')
            stop_word.append(line)
    return stop_word

def jiebacut(path):
    jieba.load_userdict(r'my_dict.txt')
    pattern = re.compile( '[\ue80b└└┌┐│┬\\t\\n\\s,.<>/?:;\'\"[\\]{}()\\|~!@#$%^&*\\-_=+a-zA-Z，。《》、？：；“”‘’｛｝【】（）…￥！—┄－\\d.．＿─]+')
    with open(path, 'r', encoding='utf-8') as f:
        strline = re.sub(pattern, '', f.read())
        words = jieba.lcut(strline)
    stopword = loadstopword(r'stop_words.txt')
    words2 = [word for word in words if word not in stopword]
    return words2

class ClassifyContract(object):
    def __init__(self,filepath,pathdir):
        self.filepath = filepath
        self.pathdir = pathdir
        self.wordappend = []

#classification等于1：劳动合同分类；2：实习合同分类，默认为1

    def classifyBatch(self,classification=1):
        if classification==1:
            label = load_label()
        if classification==2:
            label = load_label_internships()
        dictionary = corpora.Dictionary.load(r'./TrainData/contract.dict')
        lsi = models.LsiModel.load(r'./model.lsi')
        L = self.listAll(self.pathdir)
        if len(L)>500:
            L = L[:500]
        batchVec = []
        for l in L:
            word =  jiebacut(l)
            vec = dictionary.doc2bow(word)
            word_vec = lsi[vec]
            vec = []
            self.wordappend.append(word_vec)
            for a in word_vec:
                vec.append(a[1])
            batchVec.append(vec)
        return batchVec

    def listAll(self, file_dir):
        L = []
        add = 0
        for root, dirs,files in os.walk(file_dir):
            pass
        L = [root+file for file in files]
        return L


if __name__=='__main__':
    i=0
    classification = 1
    path = r'../../test_data/测试_劳动合同/'
    # path = r'../../test_data/测试_非劳动合同/'
    allnum = 0
    positive =0
    label = []
    # starttime = datetime.datetime.now()
    cc = ClassifyContract(None,path)
    tem = cc.classifyBatch()
    # a = xgboostDemo.xgbpredictBatch(tem)
    a = SVMDemo.SVM().loadModelPredict(tem)
    for t in a:
        if t == 1:
            positive += 1
        allnum += 1
    print(positive/allnum)

    endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)



    if classification == 1:
        label = load_label()
    if classification == 2:
        label = load_label_internships()
    exampleVec = cc.wordappend
    index = similarities.MatrixSimilarity.load(r'./lsi_model.index')
    ans = []
    allnum = 0
    positive = 0
    for i in exampleVec:
        sims = index[i]
        sims2 = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)[:1]
        ans.append(label[sims2[0][0]])
    for t in ans:
        if t == 1:
            positive += 1
        allnum += 1
    print(positive/allnum)





