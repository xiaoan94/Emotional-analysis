#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
import gensim
import jieba




data_more = pd.read_csv("data_more/ratings.csv", encoding="utf_8")
print(data_more.shape)
print(data_more.columns)



def segmentWord(cont):
    c = []
    for i in cont:
       a = list(jieba.cut(i))
       b = " ".join(a).replace('，', '').replace('。', '').replace('？', '')\
             .replace('！', '').replace('“', '').replace('”', '').replace('：', '')\
             .replace('…', '').replace('（', '').replace('）', '').replace('—', '')\
             .replace('《', '').replace('》', '').replace('、', '').replace('‘', '')\
             .replace('’', '').replace('"','').replace('~','').replace('#','').replace('【','')\
            .replace('】','').replace('.','').replace('^','').replace('_','').replace('(','')\
            .replace(')','').replace('～','').replace("'",'').replace(':','')
       c.append(b)
    return c


data_more['comment'] = data_more['comment'].fillna(-1)
print(data_more[data_more['comment'] == -1].shape[0]*1.0/data_more.shape[0])
data_more = data_more[data_more['comment'] != -1]
print(data_more.shape)


dimension = 150
comment = segmentWord(data_more['comment'])
print("================begin==========word2vec=============")
model = gensim.models.Word2Vec(comment, size=dimension, min_count=3)
model.save('data_more/word2vec_150_3')  ## 保存模型






