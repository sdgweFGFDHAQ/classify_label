# encoding=utf-8
import os

from gensim.models import KeyedVectors
from multiprocessing import Manager, Pool
import numpy as np
import pandas as pd
import torch

from global_parameter import StaticParameter as SP
from mini_tool import WordSegment, error_callback

# 标准化的已打标数据集
labeled_data_path = './data/labeled_data.csv'
# 标准化的未打标数据集
unlabeled_data_path = './data/unlabeled_data.csv'


# 读取原始文件,将数据格式标准化
def set_file_standard_data(path, is_label=True):
    """
    :param path:
    :param is_label: 是否清洗无标签数据,默认为 True
    """
    csv_data = pd.read_csv(path,
                           usecols=['id', 'name', 'category1_new', 'category2_new', 'category3_new'],
                           keep_default_na=False)
    pool = Pool(processes=4)
    standard_df = Manager().list()
    # 设置jieba
    segment = WordSegment()
    # 选择输出
    if is_label:
        csv_data = csv_data[csv_data['category1_new'].notnull() & (csv_data['category1_new'] != "")]
        # 用一级标签填充空白(NAN)的二级标签、三级标签
        csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
        csv_data['category3_new'].fillna(csv_data['category2_new'], inplace=True)
        # 得到各级标签映射字典
        category = csv_data[['category1_new', 'category2_new', 'category3_new']]
        category = category.drop_duplicates(keep='first')
        category.reset_index(inplace=True, drop=True)
        category.to_csv('./data/category_dict.csv')
        print("类别个数：", len(category['category3_new']))
        # 得到标准数据
        csv_data_groups = csv_data.groupby('category3_new')
        for category3, csv_data_i in csv_data_groups:
            pool.apply_async(cut_world_async, args=(segment, csv_data_i, standard_df), error_callback=error_callback)
        result_data = pd.concat(standard_df, ignore_index=True)
        result_data.to_csv('./data/labeled_data.csv', columns=['id', 'name', 'category3_new', 'cut_name'])
    else:
        csv_data = csv_data[csv_data['category1_new'].null() | (csv_data['category1_new'] == "")]
        # 得到标准数据
        csv_data_groups = csv_data.groupby('category3_new')
        for category3, csv_data_i in csv_data_groups:
            pool.apply_async(cut_world_async, args=(segment, csv_data_i, standard_df), error_callback=error_callback)
        result_data = pd.concat(standard_df, ignore_index=True)
        result_data.to_csv('./data/unlabeled_data.csv', columns=['id', 'name', 'category3_new', 'cut_name'])
    pool.close()
    pool.join()


def cut_world_async(segmenter, df, result_df):
    df['cut_name'] = df['name'].apply(segmenter.cut_word)
    result_df.append(df)


# 读取全量数据
def get_data(is_label=True):
    """
    :param is_label: 是否读取有标签数据,默认为 True
    :return:
    """
    if is_label:
        path = labeled_data_path
        csv_data = pd.read_csv(path, usecols=['id', 'name', 'category3_new', 'cut_name'])
        return csv_data
    else:
        path = unlabeled_data_path
        csv_data = pd.read_csv(path, usecols=['id', 'name', 'category3_new', 'cut_name'])
        return csv_data


class Preprocess:
    def __init__(self, sen_len):  # 首先定义类的一些属性
        self.embedding = KeyedVectors.load_word2vec_format(SP.PATH_ZZX + '/workplace/fewsamples/models/word2vec.vector')
        self.vector_size = self.embedding.vector_size
        self.sen_len = sen_len
        self.word2idx = {}
        self.idx2word = []
        self.embedding_matrix = []
        self.lab2idx = {}
        self.idx2lab = {}

    # 构建向量矩阵
    def create_tokenizer(self):
        # 把'<PAD>'、'<UNK>'加进embedding
        vocab_list = list()
        vocab_list.append('<PAD>')
        vocab_list = vocab_list + [k for k in self.embedding.key_to_index.keys()]
        vocab_list.append('<UNK>')
        embeddings_matrix = np.zeros((len(vocab_list) + 2, self.vector_size))
        for i in range(len(vocab_list)):
            try:
                word = vocab_list[i]
                embeddings_matrix[i] = self.embedding[word]
            except Exception:
                vector = torch.zeros(1, self.vector_size)
                # torch.nn.init.uniform_(vector)
                embeddings_matrix[i] = vector
            finally:
                self.word2idx[vocab_list[i]] = i
        self.idx2word = vocab_list
        self.embedding_matrix = torch.tensor(embeddings_matrix)
        return self.embedding_matrix

    # 把句子里面的字变成相对应的index
    def get_pad_word2idx(self, sentences):
        text_to_sequence = []
        for sentence in sentences:
            sentence = sentence.split()
            sequence = []
            for word in sentence:
                if word in self.word2idx.keys():
                    sequence.append(self.word2idx[word])
                else:
                    sequence.append(self.word2idx['<UNK>'])
            # 统一句子长度
            sequence = self.pad_sequence(sequence)
            text_to_sequence.append(sequence)
        return torch.tensor(text_to_sequence)

    # 填充长度,将每个句子变成一样的长度
    def pad_sequence(self, sentence):
        if len(sentence) > self.sen_len:
            new_sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            new_sentence = list()
            for _ in range(pad_len):
                new_sentence.append(self.word2idx["<PAD>"])
            # new_sentence.extend(sentence)
            sentence.extend(new_sentence)
            new_sentence = sentence
        assert len(new_sentence) == self.sen_len
        return new_sentence

    # 获取数据“标签列“的向量形式
    def get_lab2idx(self, labels):
        cat_df = pd.read_csv('../category_to_id.csv')
        # 生成类别字典
        self.lab2idx = dict(zip(cat_df['category3_new'], cat_df['cat_id']))
        self.idx2lab = dict(zip(cat_df['cat_id'], cat_df['category3_new']))
        # 输出1D标签索引
        y = list()
        if labels is not None:
            for lab in labels:
                y.append(self.lab2idx[lab])
            return torch.LongTensor(y)
        else:
            return None

    # 店名长度的分布分析
    def length_distribution(self, sentences):
        len_list = []
        for sentence in sentences:
            len_list.append(len(sentence.split()))
        set_len = int(np.median(len_list))
        self.sen_len = set_len if set_len > self.sen_len else self.sen_len
        print('length_distribution:', self.sen_len)
