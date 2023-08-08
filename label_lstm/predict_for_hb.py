import os
import time

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

from preprocess_data import Preprocess
from global_parameter import StaticParameter as SP
from mini_tool import WordSegment

warnings.filterwarnings("ignore", category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DefineDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def get_city_forhb(city_list):
    path_part = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_hb.csv'
    for city in city_list:
        path_city = SP.PATH_ZZX_DATA + city + '.csv'
        if os.path.exists(path_city):
            csv_data = pd.read_csv(path_city,
                                   usecols=['id', 'name', 'category1_new', 'category2_new', 'category3_new'])
            # 用一级标签填充空白(NAN)的二级标签、三级标签
            # csv_data = csv_data[csv_data['category1_new'].notnull() & (csv_data['category1_new'] != "")]
            csv_data['category2_new'].fillna(csv_data['category1_new'], inplace=True)
            csv_data['category3_new'].fillna(csv_data['category2_new'], inplace=True)
            # 得到标准数据
            segment = WordSegment()
            csv_data['cut_name'] = csv_data['name'].apply(segment.cut_word)
            if os.path.exists(path_part) and os.path.getsize(path_part):
                csv_data.to_csv(path_part,
                                columns=['id', 'name', 'category3_new', 'cut_name'], mode='a', header=False)
            else:
                csv_data.to_csv(path_part,
                                columns=['id', 'name', 'category3_new', 'cut_name'], mode='w')


def get_file_forhb():
    cities = ['随州市', '恩施土家族苗族自治州', '武汉市', '宜昌市', '黄冈', '咸宁市', '鄂州市', '荆门市', '襄阳市',
              '神农架林区', '黄石市', '孝感市', '十堰市', '天门市', '荆州市', '仙桃市', '潜江市']
    path_sta = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_hb.csv'
    if os.path.exists(path_sta):
        open(path_sta, "r+").truncate()
    get_city_forhb(cities)


def predict_result_forhb(model):
    gz_df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_hb.csv')
    data_x = gz_df['cut_name'].values
    # data pre_processing
    preprocess = Preprocess(sen_len=7)
    # 加载model paragram
    embedding = preprocess.create_tokenizer()
    # 初始化参数
    data_x = preprocess.get_pad_word2idx(data_x)
    preprocess.get_lab2idx(None)
    pre_x = DefineDataset(data_x, None)
    pre_ip = DataLoader(dataset=pre_x, batch_size=32, shuffle=False, drop_last=False)
    pre_lists = list()
    # 將 model 的模式设定为 eval，固定model的参数
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(pre_ip):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            # 2. 计算输出
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            pre_label = outputs.argmax(axis=1)
            pre_lists.extend(pre_label)
    cate_lists = []
    for ind in pre_lists:
        cate_lists.append(preprocess.idx2lab[ind.item()])
    result = pd.DataFrame(
        {'store_id': gz_df['id'], 'name': gz_df['name'], 'category3_new': gz_df['category3_new'],
         'predict_category': cate_lists})
    result.to_csv(SP.PATH_ZZX_PREDICT_DATA + 'predict_category_hb.csv')


if __name__ == '__main__':
    # pred预测集
    # get_file_forhb()
    lstm_model = torch.load('best_lstm.model')
    predict_result_forhb(lstm_model)
