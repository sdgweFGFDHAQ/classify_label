import os
from multiprocessing import Pool

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

from transformers import AutoModel, AutoTokenizer

from global_parameter import StaticParameter as SP
from mini_tool import WordSegment, error_callback

warnings.filterwarnings("ignore", category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrian_bert_url = SP.PRETRAIN_BERT_URL
load_bert_model = SP.BEST_BERT_MODEL
chunksize = 1000000


class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = self.load_data(dataframe)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)

    def load_data(self, df):
        # 从数据路径加载数据
        # 返回数据列表，每个元素是一个样本
        name_lists = df['name'].tolist()
        return name_lists

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # 将中文文本转换为 tokens
        encoded_dict = self.tokenizer.encode_plus(sample,
                                                  add_special_tokens=True, max_length=12, padding='max_length',
                                                  truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        return input_ids.squeeze(0), attention_mask.squeeze(0)


# 批量标准化
def set_file_standard_data(city_list, part_i):
    for city in city_list:
        path_city = SP.PATH_ZZX_DATA + city + '.csv'
        path_part = SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(part_i) + '.csv'
        if os.path.exists(path_city):
            csv_data = pd.read_csv(path_city, usecols=['id', 'name'], keep_default_na=False)
            # 得到标准数据
            segment = WordSegment()
            csv_data['cut_name'] = csv_data['name'].apply(segment.cut_word)
            # 过滤非中文店名导致的'cut_name'=nan
            csv_data = csv_data[csv_data['cut_name'].notna()]
            if os.path.exists(path_part) and os.path.getsize(path_part):
                csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(part_i) + '.csv',
                                columns=['id', 'name', 'cut_name'],
                                mode='a', header=False)
            else:
                csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(part_i) + '.csv',
                                columns=['id', 'name', 'cut_name'],
                                mode='w')


def rerun_get_CK_file(cities):
    # 初始化，清空文件
    for csv_i in range(SP.SEGMENT_NUMBER):
        path_sta = SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(csv_i) + '.csv'
        if os.path.exists(path_sta):
            open(path_sta, "r+").truncate()
    # 合并城市,分为8部分
    city_num = len(cities)
    index_list = [int(city_num * i / SP.SEGMENT_NUMBER) for i in range(SP.SEGMENT_NUMBER + 1)]
    pool = Pool(processes=4)
    for index in range(len(index_list) - 1):
        cities_i = cities[index_list[index]:index_list[index + 1]]
        pool.apply_async(set_file_standard_data, args=(cities_i, index), error_callback=error_callback)
    pool.close()
    pool.join()


def get_city_forCK(city_list):
    path_part = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_CK.csv'
    for city in city_list:
        path_city = SP.PATH_ZZX_DATA + city + '.csv'
        if os.path.exists(path_city):
            csv_data = pd.read_csv(path_city,
                                   usecols=['id', 'name'])
            # 得到标准数据
            segment = WordSegment()
            csv_data['cut_name'] = csv_data['name'].apply(segment.cut_word)
            if os.path.exists(path_part) and os.path.getsize(path_part):
                csv_data.to_csv(path_part,
                                columns=['id', 'name', 'category1_new', 'cut_name'], mode='a', header=False)
            else:
                csv_data.to_csv(path_part,
                                columns=['id', 'name', 'category1_new', 'cut_name'], mode='w')


def predict_result(df, dataloder, model, idx2lab, part_i):
    # 进度条
    # progress_bar = tqdm(total=len(dataloder), desc='Predicting')
    pre_ind_lists, pre_max_value = list(), list()
    # 將 model 的模式设定为 eval，固定model的参数
    model.eval()
    with torch.no_grad():
        for i, (inputs, masks) in enumerate(dataloder):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            masks = masks.to(device, dtype=torch.long)
            # 2. 计算输出
            outputs = model(inputs, masks)
            outputs = outputs.squeeze(1)
            pre_value, pre_label = torch.max(outputs, dim=1)
            pre_ind_lists.extend(pre_label)
            pre_max_value.extend(pre_value)

            # 更新进度条
            # progress_bar.update(1)
    # 关闭进度条
    # progress_bar.close()
    cate_lists, threshold_cate_lists = [], []
    max_value_lists = []
    for ind in range(len(pre_ind_lists)):
        pre_ind = pre_ind_lists[ind]
        pre_value = pre_max_value[ind]
        cate_lists.append(idx2lab[pre_ind.item()])
        threshold_cate_lists.append(idx2lab[pre_ind.item()] if pre_value > 0.80 else idx2lab[-1])
        max_value_lists.append(pre_value.item())
    result = pd.DataFrame(
        {'id': df['id'], 'name': df['name'], 'state': df['state'],
         'predict_category': cate_lists, 'threshold_category': threshold_cate_lists, 'max_value': max_value_lists})

    path_part = SP.PATH_ZZX_PREDICT_DATA + 'predict_CK_category_' + str(part_i) + '.csv'
    if os.path.exists(path_part) and os.path.getsize(path_part):
        result.to_csv(path_part, mode='a', header=False)
    else:
        result.to_csv(path_part, mode='w')


def predict_csv(pi, model, idx2lab):
    df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(pi) + '.csv', chunksize=chunksize)
    for df_i in df:
        df_i = df_i[(df_i['cut_name'].notna() & df_i['cut_name'].notnull())]
        # 处理特征
        dataset = MyDataset(df_i)
        pre_dataloder = DataLoader(dataset=dataset, batch_size=SP.BATCH_SIZE, shuffle=False, drop_last=False)

        predict_result(df_i, pre_dataloder, model, idx2lab, pi)

    print("predict_CK_category_{} 写入完成".format(pi))


def predict_result_forCK_bert():
    for csv_i in range(SP.SEGMENT_NUMBER):
        path_pre = SP.PATH_ZZX_PREDICT_DATA + 'predict_CK_category_' + str(csv_i) + '.csv'
        if os.path.exists(path_pre):
            open(path_pre, "r+").truncate()

    cat_df = pd.read_csv('category_to_id.csv')
    category_number = cat_df.shape[0]  # 78
    idx2lab = dict(zip(cat_df['cat_id'], cat_df['category1_new']))
    idx2lab[-1] = SP.UNKNOWN_CATEGORY

    # bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    # lstm_model = BertLSTMNet_1(
    #     bert_embedding=bert_layer,
    #     input_dim=768,
    #     hidden_dim=128,
    #     num_classes=category_number,
    #     num_layers=2
    # ).to(device)
    # lstm_model.load_state_dict(torch.load("best_lstm_bert.pth"))
    lstm_model = torch.load(load_bert_model)

    for part_i in range(SP.SEGMENT_NUMBER):
        df = pd.read_csv(SP.PATH_ZZX_DATA + 'store_CK_data_' + str(part_i) + '.csv', chunksize=chunksize)
        for df_i in df:
            df_i = df_i[(df_i['name'].notna() & df_i['name'].notnull())]
            # 处理特征
            dataset = MyDataset(df_i)
            pre_dataloder = DataLoader(dataset=dataset, batch_size=SP.BATCH_SIZE, shuffle=False, drop_last=False)

            predict_result(df_i, pre_dataloder, lstm_model, idx2lab, part_i)

            print("predict_CK_category_{} 写入完成".format(part_i))

