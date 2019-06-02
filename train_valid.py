import gensim
import numpy as np
import pandas as pd
import jieba


class PreProcessor(object):
    def __init__(self, filename, busi_name="location_traffic_convenience"):
        self.filename = filename
        self.busi_name = busi_name
        self.embedding_dim = 150

        # 读取词向量
        # embedding_file = "./word_embedding/word2vec_wx"
        self.word2vec_model = gensim.models.Word2Vec.load('word2vec_150_3')

    # 读取原始csv文件
    def read_csv_file(self):
        # reload(sys)
        # sys.setdefaultencoding('utf-8')
        # print("after coding: " + str(sys.getdefaultencoding()))

        data = pd.read_csv(self.filename, sep=',')
        x = data.content.values
        y = data[self.busi_name].values

        return x, y

    # todo 错别字处理，语义不明确词语处理，拼音繁体处理等
    def correct_wrong_words(self, corpus):
        return corpus

    # 去掉停用词
    def clean_stop_words(self, sentences):
        stop_words = None
        with open("./stop_words.txt", "r") as f:
            stop_words = f.readlines()
            stop_words = [word.replace("\n", "") for word in stop_words]

        # stop words 替换
        for i, line in enumerate(sentences):

            for word in stop_words:
                if word in line:
                    line = line.replace(word, "")
            sentences[i] = line

        return sentences

    # 分词，将不在词向量中的jieba分词单独挑出来，他们不做分词
    def get_words_after_jieba(self, sentences):
        # jieba分词
        all_exclude_words = dict()
        while (1):
            words_after_jieba = [[w for w in jieba.cut(line) if w.strip()] for line in sentences]
            # 遍历不包含在word2vec中的word
            new_exclude_words = []
            for line in words_after_jieba:
                for word in line:
                    if word not in self.word2vec_model.wv.vocab and word not in all_exclude_words:
                        all_exclude_words[word] = 1
                        new_exclude_words.append(word)
                    elif word not in self.word2vec_model.wv.vocab:
                        all_exclude_words[word] += 1

            # 剩余未包含词小于阈值，返回分词结果，结束。否则添加到jieba del_word中，然后重新分词
            if len(new_exclude_words) < 10:
                print("length of not in w2v words: %d, words are:" % len(new_exclude_words))
                for word in new_exclude_words:
                    print(word)
                print("\nall exclude words are: ")
                for word in all_exclude_words:
                    if all_exclude_words[word] > 5:
                        print("%s: %d," % (word, all_exclude_words[word]))
                return words_after_jieba
            else:
                for word in new_exclude_words:
                    jieba.del_word(word)

        raise Exception("get_words_after_jieba error")

    # 去除不在词向量中的词
    def remove_words_not_in_embedding(self, corpus):
        for i, sentence in enumerate(corpus):
            for word in sentence:
                if word not in self.word2vec_model.wv.vocab:
                    sentence.remove(word)
                    corpus[i] = sentence

        return corpus

    # 词向量，建立词语到词向量的映射
    def form_embedding(self, corpus):
        # 1 读取词向量
        w2v = dict(zip(self.word2vec_model.wv.index2word, self.word2vec_model.wv.syn0))

        # 2 创建词语词典，从而知道文本中有多少词语
        w2index = dict()        # 词语为key，索引为value的字典
        index = 1
        for sentence in corpus:
            for word in sentence:
                if word not in w2index:
                    w2index[word] = index
                    index += 1
        print("\nlength of w2index is %d" % len(w2index))

        # 3 建立词语到词向量的映射
        # embeddings = np.random.randn(len(w2index) + 1, self.embedding_dim)
        embeddings = np.zeros(shape=(len(w2index) + 1, self.embedding_dim), dtype=float)
        embeddings[0] = 0   # 未映射到的词语，全部赋值为0

        n_not_in_w2v = 0
        for word, index in w2index.items():
            if word in self.word2vec_model.wv.vocab:
                embeddings[index] = w2v[word]
            else:
                print("not in w2v: %s" % word)
                n_not_in_w2v += 1
        print("words not in w2v count: %d" % n_not_in_w2v)

        del self.word2vec_model, w2v

        # 4 语料从中文词映射为索引
        x = [[w2index[word] for word in sentence] for sentence in corpus]

        return embeddings, x

    # 预处理，主函数
    def process(self):
        # 读取原始文件
        x, y = self.read_csv_file()

        # 错别字，繁简体，拼音，语义不明确，等的处理
        x = self.correct_wrong_words(x)

        # stop words
        # x = self.clean_stop_words(x)

        # 分词
        x = self.get_words_after_jieba(x)

        # remove不在词向量中的词
        x = self.remove_words_not_in_embedding(x)

        # 词向量到词语的映射
        embeddings, x = self.form_embedding(x)

        # 打印
        print("embeddings[1] is, ", embeddings[1])
        print("corpus after index mapping is, ", x[0])
        print("length of each line of corpus is, ", [len(line) for line in x])

        return embeddings, x, y
h = PreProcessor('./data/trainingset.csv')
h.process()