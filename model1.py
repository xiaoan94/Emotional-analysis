#!/usr/bin/env python3
# coding: utf-8


import sys
defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sys
from sklearn.metrics import f1_score
import jieba
from one_xgb import *
import xgboost as xgb
from gensim.models.word2vec import LineSentence, Word2Vec


def segmentWord(cont):
    c = []
    for i in cont:
       a = list(jieba.cut(i))
       b = " ".join(a)
       c.append(b)
    return c

def readData(d_train ,d_valid, d_test):
    print("训练样本 = %d" % len(d_train))
    print("验证样本 = %d" % len(d_valid))
    print("测试样本 = %d" %len(d_test))
    # content_train=segmentWord(d_train['content'])
    # content_valid=segmentWord(d_valid['content'])
    # content_test=segmentWord(d_test['content'])
    result_folder = 'result/'
    content_train = [line.strip() for line in open(result_folder + "train_word.txt", "r")]
    content_valid = [line.strip() for line in open(result_folder + "valid_word.txt", "r")]
    content_test = [line.strip() for line in open(result_folder + "test_word.txt", "r")]

    vectorizer = TfidfVectorizer(analyzer='word',min_df=3,token_pattern=r"(?u)\b\w\w+\b")
    features = vectorizer.fit_transform(content_train)
    print("训练样本特征表长度为 " + str(features.shape))
    # print(vectorizer.get_feature_names()) #特征名展示
    valid_features = vectorizer.transform(content_valid)
    print("验证样本特征表长度为 "+ str(valid_features.shape))
    test_features = vectorizer.transform(content_test)
    print("测试样本特征表长度为 "+ str(test_features.shape))
    data=[d_train,d_valid,d_test,features,valid_features,test_features]
    return data


## 用word2vector计算每个词的词向量, 并对每个词的向量求和取平均，得到每首诗文的特征向量
def one_wordvc(cut_word, new_model, dimension):
    vec = np.zeros(dimension)
    count = 0
    for w in cut_word.split(" "):
        try:
            if type(w) != str:
                w = str(w)
            vec += new_model[w]
            count += 1
        except:   # 有的词语不满足min_count则不会被记录在词表中
            pass

    if count > 0:
        wodvc = np.array([v / count for v in vec])
    else:
        wodvc = np.zeros(dimension)
    print wodvc
    return wodvc

def wordvs_df(word_list, model, dimension):
    features = []
    for i in word_list:
        features.append(one_wordvc(i, model, dimension))
    features = pd.DataFrame(features)
    print features.shape
    return "word2vc", features

def write_file(data, flag):
    f = open('result/' + flag + "_word.txt", "w")
    for i in data:
        f.write(i + "\n")
    f.close()


def readData2(d_train ,d_valid, d_test):
    print("训练样本 = %d" % len(d_train))
    print("验证样本 = %d" % len(d_valid))
    print("测试样本 = %d" %len(d_test))
    # content_train = segmentWord(d_train['content'])
    # content_valid = segmentWord(d_valid['content'])
    # content_test = segmentWord(d_test['content'])
    # write_file(content_train, "train")
    # write_file(content_valid, "valid")
    # write_file(content_valid, "test")
    result_folder = 'result/'
    content_train = [line.strip() for line in open(result_folder + "train_word.txt", "r")]
    content_valid = [line.strip() for line in open(result_folder + "valid_word.txt", "r")]
    content_test = [line.strip() for line in open(result_folder + "test_word.txt", "r")]

    print "word2vc---------------"
    ### min_count可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    ### size是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百,也是神经网络 NN 层单元数，它也对应了训练算法的自由程度
    ### workers参数控制训练的并行数
    dimension = 1000
    # content_train_word = []
    # for i in content_train:
    #     content_train_word.extend(" ".split(i))
    # model = Word2Vec(sentences=content_train_word, size=dimension, min_count=2, workers=multiprocessing.cpu_count())  ## 训练集
    # model.save('model/word_model')  ## 保存模型
    model = Word2Vec.load('model/word_model')
    features = wordvs_df(content_train, model, dimension)
    valid_features = wordvs_df(content_valid, model, dimension)
    test_features = wordvs_df(content_test, model, dimension)
    data=[d_train,d_valid,d_test,features,valid_features,test_features]
    return data


def svm_model(data):
    print("--------------------------------------")
    print("Start training SVM")
    d_train,d_valid,d_test,features,valid_features,test_features=data
    #支持向量机
    #C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0
    columns = d_train.columns.values.tolist()
    f1=0
    i = 0
    for col in columns[3:]:
        print(col)
        lb_train = d_train[col]
        lb_valid = d_valid[col]
        lb_train = lb_train.replace(-1, 2)
        lb_train = lb_train.replace(-2, 3)
        lb_valid = lb_valid.replace(-1, 2)
        lb_valid = lb_valid.replace(-2, 3)
        # svmmodel =SVC(C=1,kernel= "linear",degree=3)#kernel：参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF";ß
        # svmmodel.fit(features , lb_train)
        # preds = svmmodel.predict(valid_features)
        lr_model = LogisticRegression(class_weight="balanced").fit(features, lb_train)  ## 逻辑回归模型
        preds = lr_model.predict(valid_features)
        # xgtest = xgb.DMatrix(test_features)
        # xgb_model = xgb_model_one(features, lb_train, valid_features, lb_valid, i)
        # i += 1
        # xgvalid = xgb.DMatrix(valid_features)
        # preds = xgb_model.predict(xgvalid, ntree_limit=xgb_model.best_iteration)
        # preds = xgb_model.predict(xgvalid)
        ff = f1_score(lb_valid, preds, average='macro')
        print('f1=%s'%ff)
        f1 += ff
        # d_test[col]=svmmodel.predict(test_features)
        d_test[col] = lr_model.predict(test_features)
        # d_test[col] = xgb_model.predict(xgtest)
    # d_test.to_csv('result/predict3.csv')
    print('ave_f1=%s'%(f1/20))
    print('predict file result/predict.csv')


if __name__=='__main__':
    d_train = pd.read_csv('./data/trainingset.csv',encoding="utf_8")# 训练数据集
    d_valid = pd.read_csv('./data/validationset.csv',encoding="utf_8")# 验证数据集
    d_test = pd.read_csv('./data/testset.csv',encoding="utf_8")# 测试数据集
    data = readData(d_train ,d_valid, d_test)
    # data = readData2(d_train, d_valid, d_test)
    svm_model(data)





