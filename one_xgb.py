#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

import pandas as pd
import numpy as np
import xgboost as xgb
import multiprocessing



def xgb_model_one(X_train, y_train, X_val, y_val, i):
    params = {
        'booster': 'gbtree',
        # 'booster': 'gblinear',
        'objective': 'multi:softmax',  # 是一个多类的问题，因此采用了multisoft多分类器
        # 'objective': 'multi:softprob',  # 返回的是每个数据属于各个类别的概率
        'max_depth': 100,  # 构建树的深度
        'silent': 1,  # 取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时信息
        'num_class': 4,  # 类数，与 multisoftmax 并用
        'gamma': 0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。
        # 'lambda':450, # L2 正则项权重
        'subsample': 0.7,  # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
        'colsample_bytree': 0.7,  # 构建树树时的采样比率 (0:1]
        # 'min_child_weight': 100,  # 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
        # 在线性回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative
        'eta': 0.03,  # 如同学习率
        'seed': 710,  # 随机数的种子
        'nthread': multiprocessing.cpu_count(),  # cpu 线程数,根据自己U的个数适当调整
        'scale_pos_weight': 3,  # 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛
    }

    plst = list(params.items())
    num_rounds = 20  # 迭代次数

    xgtrain = xgb.DMatrix(X_train, y_train)
    xgval = xgb.DMatrix(X_val, y_val)
    # return 训练和验证的错误率
    watchlist = [(xgtrain, 'train'), (xgval, 'val')]

    # training model
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=100)
    model.save_model('model/xgb_pred' + str(i) + '.model')  # 用于存储训练出的模型
    return model



