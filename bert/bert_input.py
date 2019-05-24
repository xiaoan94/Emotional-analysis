#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import subprocess

d_train = pd.read_csv('./data/trainingset.csv', encoding="utf_8")  # 训练数据集
d_valid = pd.read_csv('./data/validationset.csv', encoding="utf_8")  # 验证数据集
d_test = pd.read_csv('./data/testset.csv', encoding="utf_8")  # 测试数据集
columns = d_train.columns.values.tolist()
print(columns)




def get_result1():
    for cols in columns[3:]:
        df = pd.read_table("bert_model_test/" + cols + "_test_results.txt", header=None)
        # print(df.shape)
        # labels = ['-2', '-1', '0', '1']
        labels = ['1', '0', '-1', '-2']
        df.columns = labels
        tag_list = []
        for index, row in df.iterrows():
            ind = list(row).index(max(list(row)))  ## 取概率最大的label
            tag = labels[ind]
            tag_list.append(tag)
        d_test[cols] = tag_list
    d_test.to_csv('result/predict_bert.csv')



def get_result2():
    for cols in columns[3:]:
        df = pd.read_table("bert_model_test/" + cols + "_test_results.txt", header=None)
        # print(df.shape)
        # labels = ['-2', '-1', '0', '1']
        labels = ['1', '0', '-1', '-2']
        df.columns = labels
        tag_list = []
        train_label_cnt = d_train.groupby(by=cols).count()
        one_jizhun = list(train_label_cnt['content'] * 1.0 / d_train.shape[0])
        one_jizhun_dict = {
             '-2': one_jizhun[0]
            , '-1': one_jizhun[1]
            , '0': one_jizhun[2]
            , '1': one_jizhun[3]
        }
        for index, row in df.iterrows():
            # raito = [one[1]/one[0] for one in zip(one_jizhun, list(row))]
            raito = [row[xx]/one_jizhun_dict[xx] for xx in labels]
            ind = raito.index(max(raito))   ## 取概率比值最大的label
            tag = labels[ind]
            tag_list.append(tag)
        d_test[cols] = tag_list
    d_test.to_csv('result/predict_bert_ratio.csv')



for cols in columns[3:]:
    p = subprocess.call(['python', 'my_run_classifier.py'
                         , '--task_name=mytask'
                         , '--do_eval=true'
                         , '--data_dir=./data/'
                         , '--vocab_file=G:/bert_chinese/chinese_L-12_H-768_A-12/vocab.txt'
                         , '--bert_config_file=G:/bert_chinese/chinese_L-12_H-768_A-12/bert_config.json'
                         , '--init_checkpoint=G:/bert_chinese/chinese_L-12_H-768_A-12/bert_model.ckpt'
                         , '--max_seq_length=128'
                         , '--train_batch_size=32'
                         , '--learning_rate=2e-5'
                         , '--num_train_epochs=3.0'
                         , '--output_dir=./bert_model_test/'
                         , '--one_label=%s'%cols])
    p2 = subprocess.call(['python', 'my_run_classifier.py'
                         , '--task_name=mytask'
                         , '--do_predict=true'
                         , '--data_dir=./data/'
                         , '--vocab_file=G:/bert_chinese/chinese_L-12_H-768_A-12/vocab.txt'
                         , '--bert_config_file=G:/bert_chinese/chinese_L-12_H-768_A-12/bert_config.json'
                         , '--init_checkpoint=G:/bert_chinese/chinese_L-12_H-768_A-12/bert_model.ckpt'
                         , '--max_seq_length=128'
                         , '--train_batch_size=32'
                         , '--learning_rate=2e-5'
                         , '--num_train_epochs=3.0'
                         , '--output_dir=./bert_model_test/'
                         , '--one_label=%s'%cols])



get_result1()
get_result2()






