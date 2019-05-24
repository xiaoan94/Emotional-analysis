import pandas as pd
import numpy as np
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import jieba
import gensim
# from gensim.models import word2vec
import logging


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
    # print(wodvc)
    return wodvc

def wordvs_df(word_list, model, dimension):
    features = []
    for i in word_list:
        features.append(one_wordvc(i, model, dimension))
    features = pd.DataFrame(features)
    print (features.shape)
    return features

def readData(d_train ,d_valid, d_test):
    print("训练样本 = %d" % len(d_train))
    print("验证样本 = %d" % len(d_valid))
    print("测试样本 = %d" %len(d_test))
    content_train=segmentWord(d_train['content'])
    content_valid=segmentWord(d_valid['content'])
    content_test=segmentWord(d_test['content'])


    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    dimension = 100
    model = gensim.models.Word2Vec(content_train,size = dimension, min_count=3)
    features = wordvs_df(content_train, model, dimension)
    valid_features = wordvs_df(content_valid, model, dimension)
    test_features = wordvs_df(content_test, model, dimension)

    data=[d_train,d_valid,d_test,features,valid_features,test_features]
    return data

def LR(data):
    print("--------------------------------------")
    print("Start training LR")
    d_train,d_valid,d_test,features,valid_features,test_features=data

    columns = d_train.columns.values.tolist()
    f1 = 0
    i = 0
    for col in columns[3:]:
        print(col)
        lb_train=d_train[col]
        lb_valid=d_valid[col]
        # svmmodel =SVC(C=1,kernel= "linear",degree=3)
        # #kernel：参数选择有rbf, linear, poly, Sigmoid, 默认的是"RBF";ß
        # svmmodel.fit(features , lb_train)
        # preds=svmmodel.predict(valid_features)
        lr_model = LogisticRegression(class_weight="balanced").fit(features, lb_train)  ## 逻辑回归模型
        preds = lr_model.predict(valid_features)
        ff=f1_score(lb_valid, preds, average='macro')
        print('f1=%s'%ff)
        i = i+1
        print(i)
        f1+=ff
        d_test[col]=lr_model.predict(test_features)
    d_test.to_csv('result/predict.csv')
    print('ave_f1=%s'%(f1/20))
    print('predict file result/predict.csv')

if __name__=='__main__':
    d_train=pd.read_csv('./data/trainingset.csv',encoding="utf_8")# 训练数据集
    d_valid=pd.read_csv('./data/validationset.csv',encoding="utf_8")# 验证数据集
    d_test=pd.read_csv('./data/testset.csv',encoding="utf_8")# 测试数据集
    data=readData(d_train ,d_valid, d_test)
    LR(data)
