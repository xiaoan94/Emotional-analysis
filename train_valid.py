import gensim
import numpy as np
import pandas as pd
import jieba
import torch
from sklearn.metrics import f1_score

class PreProcessor(object):
    def __init__(self, filename, busi_name="location_traffic_convenience"):
        self.filename = filename
        self.busi_name = busi_name
        # self.embedding_dim = 150

        self.word2vec_model = gensim.models.Word2Vec.load('word2vec_150_3')

    # 读取原始csv文件
    def read_csv_file(self):

        data = pd.read_csv(self.filename, sep=',')
        x = data.content.values
        y = data[self.busi_name].values
        y = list(y)
        y = [2 if i == -1 else i for i in y]
        y = [3 if i == -2 else i for i in y]

        return x, y

    # 去掉停用词
    # def clean_stop_words(self, sentences):
    #     stop_words = None
    #     with open("./stop_words.txt", "r") as f:
    #         stop_words = f.readlines()
    #         stop_words = [word.replace("\n", "") for word in stop_words]
    #
    #     # stop words 替换
    #     for i, line in enumerate(sentences):
    #
    #         for word in stop_words:
    #             if word in line:
    #                 line = line.replace(word, "")
    #         sentences[i] = line
    #
    #     return sentences

    def segmentWord(self,cont):
        corpus = []
        for i in cont:
            a = list(jieba.cut(i))
            b = " ".join(a).replace('，', '').replace('。', '').replace('？', '') \
                .replace('！', '').replace('“', '').replace('”', '').replace('：', '') \
                .replace('…', '').replace('（', '').replace('）', '').replace('—', '') \
                .replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                .replace('’', '').replace('"', '').replace('~', '').replace('#', '').replace('【', '') \
                .replace('】', '').replace('.', '').replace('^', '').replace('_', '').replace('(', '') \
                .replace(')', '').replace('～', '').replace("'", '').replace(':', '')

            corpus.append(b)

        # print(c)
        return corpus

    # 词向量，建立词语到词向量的映射
    def form_embedding(self, corpus):
        em_df = []
        for line in corpus:
            i = 0
            line = line.split(' ')
            for word in line:
                try:
                    line[i] = self.word2vec_model[word]
                    i += 1
                except:
                    line[i] = np.zeros(150)
                    i += 1

            if len(line) > 316:
                line = line[:316]

            zero = np.zeros(150)
            if len(line) < 316:
                num_n = 316 - len(line)
                for num in range(num_n):
                    line = np.vstack((line,zero))
            em_df.append(line)
        em_df = np.array(em_df)
        return em_df


    def process(self):
        # 读取原始文件
        x, y = self.read_csv_file()
        # print(type(x))
        # stop words
        # x = self.clean_stop_words(x)
        # 分词
        x = self.segmentWord(x)
        # print(type(x))
        # 词向量到词语的映射
        x = self.form_embedding(x)
        y = np.array(y)
        # print(x)
        # print(type(x))
        return  x, y

class EncoderRNNWithVector(torch.nn.Module):
    def __init__(self, hidden_size, out_size, n_layers=1, batch_size=1):
        super(EncoderRNNWithVector, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size

        # 这里指定了 BATCH FIRST
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

        # 加了一个线性层，全连接
        self.out = torch.nn.Linear(hidden_size, out_size)

    def forward(self, word_inputs, hidden):
        # -1 是在其他确定的情况下，PyTorch 能够自动推断出来，view 函数就是在数据不变的情况下重新整理数据维度
        # batch, time_seq, input
        inputs = word_inputs.view(self.batch_size, -1, self.hidden_size)

        # hidden 就是上下文输出，output 就是 RNN 输出
        output, hidden = self.gru(inputs, hidden)

        output = self.out(output)

        # 仅仅获取 time seq 维度中的最后一个向量
        # the last of time_seq
        output = output[:, -1, :]

        return output, hidden

    def init_hidden(self):

        hidden = torch.autograd.Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return hidden


def _test_rnn_rand_vec(x,y,x_valid,y_valid):
    _xs = x
    _ys = y

    x_valid = x_valid
    y_valid = y_valid
    #
    # x_test = x_test
    # y_test = y_test

    # 隐层 200，输出 6，隐层用词向量的宽度，输出用标签的值得个数 （one-hot)
    model = EncoderRNNWithVector(200, 4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epoch_n = 10

    train_labels = []
    train_pred = []

    for i in range(len(_xs)): # 以此遍历每句话
        for epoch in range(epoch_n):
            encoder_hidden = model.init_hidden()

            input_data = torch.autograd.Variable(torch.Tensor(_xs[i]))
            output_labels = torch.autograd.Variable(torch.LongTensor([_ys[i]]))

            # print(output_labels)

            encoder_outputs, encoder_hidden = model(input_data, encoder_hidden)

            optimizer.zero_grad()

            loss = criterion(encoder_outputs, output_labels)
            loss.backward()
            optimizer.step()
        encoder_outputs = torch.argmax(encoder_outputs,dim=1)
        encoder_outputs = encoder_outputs.data.numpy()[0]
        output_labels = output_labels.data.numpy()[0]

        train_labels.append(output_labels)
        train_pred.append(encoder_outputs)
        print(i, output_labels, encoder_outputs)
        if i % 20 == 0:
            print('--' * 20)

    ff = f1_score(train_labels, train_pred, average='macro')       #
    print('训练集已训练完,f1值是:{}'.format(ff))


    valid_labels = []
    valid_pred = []
    for j in range(len(x_valid)):
        encoder_hidden = model.init_hidden()

        input_data2 = torch.autograd.Variable(torch.Tensor(x_valid[j]))
        output_labels2 = torch.autograd.Variable(torch.LongTensor([y_valid[j]]))

        encoder_outputs2,encoder_hidden2 = model(input_data2,encoder_hidden)

        encoder_outputs2 = torch.argmax(encoder_outputs2, dim=1)

        encoder_outputs2 = encoder_outputs2.data.numpy()[0]
        output_labels2 = output_labels2.data.numpy()[0]

        valid_labels.append(output_labels2)
        valid_pred.append(encoder_outputs2)

        print(j, output_labels2, encoder_outputs2)
        if j % 20 == 0:
            print('--' * 20)
    print('--'*20)
    ff = f1_score(valid_labels, valid_pred, average='macro')
    print(ff)
    return


h = PreProcessor('./data/trainingset.csv')
s,c = h.process()
h1 = PreProcessor('./data/validationset.csv')
s1,c1 = h1.process()
x = _test_rnn_rand_vec(s,c,s1,c1)
