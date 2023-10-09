import os
import time
import warnings
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiprocessing import Pool
import torch
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, BertConfig

from model_bert import BertLSTMNet_2
from deep_learning_bert import training, evaluating
from global_parameter import StaticParameter as SP
from mini_tool import WordSegment, error_callback

warnings.filterwarnings("ignore", category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"
category_number = 0
cities = SP.CITIES


def typicalsamling(group, threshold):
    count = len(group.index)
    percent = 0.2  # 超参数，考虑缓解样本不平衡但不过多增加训练集 选择的比例
    floor = round(percent * threshold)  # 最少样本数
    if count > threshold:
        return group.sample(n=threshold, random_state=23)
    elif count < floor:
        num_copies = floor // count + 1
        df_copied = pd.concat([group] * num_copies, ignore_index=True)
        df_copied = df_copied[:floor]  # 只保留floor条样本
        return df_copied
    else:
        return group.sample(frac=1)


def random_get_trainset(is_labeled=True, labeled_is_all=False):
    standard_df = pd.DataFrame(columns=['id', 'name', 'category1_new', 'cut_name'])
    result_path = 'standard_store_data.csv'
    all_fix = ''
    for i in range(SP.SEGMENT_NUMBER):
        path = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(i) + '.csv'
        df_i = pd.read_csv(path, usecols=['id', 'name', 'category1_new', 'cut_name'], keep_default_na=False)
        if is_labeled:
            df_i = df_i[df_i['category1_new'] != '']
            all_fix = '_labeled'
            if labeled_is_all:
                # 全量数据
                standard_df_i = df_i
                result_path = 'all' + all_fix + '_data.csv'
            else:
                # 部分数据
                # standard_df_i = df_i.groupby('category1_new').sample(frac=0.12, random_state=23)
                standard_df_i = df_i.groupby('category1_new').apply(typicalsamling, 1800)
        else:
            df_i = df_i[df_i['category1_new'] == '']
            standard_df_i = df_i
            all_fix = '_unlabeled'
            result_path = 'all' + all_fix + '_data.csv'
        standard_df = pd.concat([standard_df, standard_df_i])
    standard_df = standard_df.sample(frac=1).reset_index(drop=True)
    logging.info('standard_df数据量：{}'.format(len(standard_df.index)))
    standard_df.to_csv(SP.PATH_ZZX_STANDARD_DATA + result_path, index=False)


def get_dataset(city):
    city_df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + city + '.csv')
    print(len(city_df.index))

    city_df['flag'] = city_df['category1_new'].apply(lambda x: 1 if pd.notnull(x) and x != '' else 0)
    data_x, data_y = city_df['name'].values, city_df['flag'].values

    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)

    # 处理特征
    encoded_dict = tokenizer.batch_encode_plus(city_df['name'].tolist(),
                                               add_special_tokens=True, max_length=12, padding='max_length',
                                               truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    dataset = TensorDataset(input_ids, torch.Tensor(data_y), attention_masks)
    return dataset, bert_layer, tokenizer


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


def accuracy(pred_y, y):
    pred_y = (pred_y > 0.5).int()
    correct = (pred_y == y).float()
    acc = correct.sum() / len(correct)
    return acc


def training(train_loader, model):
    # 多分类损失函数
    criterion = nn.BCEWithLogitsLoss()
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 將 model 的模式设定为 train，这样 optimizer 就可以更新 model 的参数
    model.train()
    train_len = len(train_loader)
    epoch_los, epoch_acc = 0, 0
    for i, (inputs, labels, masks) in enumerate(train_loader):
        # 1. 放到GPU上
        inputs = inputs.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.long)
        masks = masks.to(device, dtype=torch.long)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        outputs = model(inputs, masks)
        outputs = outputs.squeeze(1)  # 去掉最外面的 dimension
        # 4. 计算损失
        # outputs:batch_size*num_classes labels:1D
        loss = criterion(outputs, labels)
        epoch_los += loss.item()
        # 5.预测结果
        accu = accuracy(outputs, labels)
        epoch_acc += accu.item()
        # 6. 反向传播
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
    loss_value = epoch_los / train_len
    acc_value = epoch_acc / train_len * 100
    return loss_value, acc_value


def evaluating(val_loader, model):
    # 多分类损失函数
    criterion = nn.BCEWithLogitsLoss()
    # 將 model 的模式设定为 eval，固定model的参数
    model.eval()
    val_len = len(val_loader)
    with torch.no_grad():
        epoch_los, epoch_acc = 0, 0
        for i, (inputs, labels, masks) in enumerate(val_loader):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            masks = masks.to(device, dtype=torch.long)
            # 2. 计算输出
            outputs = model(inputs, masks)
            outputs = outputs.squeeze(1)
            # 3. 计算损失
            loss = criterion(outputs, labels)
            epoch_los += loss.item()
            # 4. 预测结果
            accu = accuracy(outputs, labels)
            epoch_acc += accu.item()
        loss_value = epoch_los / val_len
        acc_value = epoch_acc / val_len * 100
    print('-----------------------------------')
    return loss_value, acc_value


def search_best_model(train_set, test_set, embedding):
    model = BertLSTMNet_2(
        bert_embedding=embedding,
        input_dim=768,
        hidden_dim=128,
        num_classes=1
    ).to(device)
    train_ip = DataLoader(dataset=train_set, batch_size=512, shuffle=True, drop_last=True)
    test_ip = DataLoader(dataset=test_set, batch_size=512, shuffle=True, drop_last=False)
    # run epochs
    best_accuracy = 0.
    for ep in range(15):
        print('==========train epoch: {}============'.format(ep))
        train_lv, train_av = training(train_ip, model)
        eval_lv, eval_av = evaluating(test_ip, model)

        print('epoch:{} train_lv{:.3f}, train_av{:.3f}'.format(ep, train_lv, train_av))
        print('   eval_lv{:.3f}, eval_lv{:.3f}'.format(eval_lv, eval_lv))

        if eval_av > best_accuracy:
            best_accuracy = eval_av
            # torch.save(model, "best_lstm_bert.model")
            torch.save(model.state_dict(), "best_lstm_bert.pth")


def predict_result(df, dataloder, model, idx2lab, city):
    try:
        pre_lists = list()
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

                pre_lists.extend(outputs)
        cate_value = []
        for ind in pre_lists:
            cate_value.append(1 if ind.item() > 0.5 else 0)
        result = pd.DataFrame({'id': df['id'], 'name': df['name'], 'flag': cate_value, 'flag_value': pre_lists})
        result.to_csv(SP.PATH_ZZX_PREDICT_DATA + '/other/predict_category_' + city + '.csv')
    except Exception as e:
        with open('error_city.txt', 'a') as ef:
            ef.write(str(time.time()))
            ef.write('出错的city: ' + city + '; 异常e:' + str(e))


# 划分合适的训练集测试集，保存训练模型
def rerun_get_model():
    # 训练模型,获取训练集
    # random_get_trainset()
    # 按城市遍历全量数据
    for city in cities:
        dataset, embedding_matrix, prepro = get_dataset(city)
        # 保存最好的模型
        train_set, test_set = train_test_split(dataset, test_size=0.2)
        search_best_model(train_set, test_set, embedding_matrix)


# 预测数据
def rerun_predict_result():
    for csv_i in range(SP.SEGMENT_NUMBER):
        path_pre = SP.PATH_ZZX_PREDICT_DATA + 'predict_category_' + str(csv_i) + '.csv'
        if os.path.exists(path_pre):
            open(path_pre, "r+").truncate()
    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    lstm_model = BertLSTMNet_2(
        bert_embedding=bert_layer,
        input_dim=768,
        hidden_dim=128,
        num_classes=1
    ).to(device)
    lstm_model.load_state_dict(torch.load("best_lstm_bert.pth"))
    # lstm_model = torch.load('best_lstm_bert.model')
    for city in cities:
        df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + city + '.csv')

        # 处理特征
        encoded_dict = tokenizer.batch_encode_plus(df['name'].tolist(),
                                                   add_special_tokens=True, max_length=12, padding='max_length',
                                                   truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        pre_x = TensorDataset(input_ids, attention_masks)

        pre_dataloder = DataLoader(dataset=pre_x, batch_size=512, shuffle=False, drop_last=False)

        idx2lab = {0: 0, 1: 1}
        predict_result(df, pre_dataloder, lstm_model, idx2lab, city)
    print("predict_category_{} 写入完成")


if __name__ == '__main__':
    start0 = time.time()
    # 1 不清洗数据，靠bert，获得正负样本
    # rerun_get_file()
    end0 = time.time()
    print('rerun_get_file time: %s minutes' % ((end0 - start0) / 60))
    # 3 划分训练集测试集，保存训练模型
    start1 = time.time()
    rerun_get_model()
    end1 = time.time()
    print('rerun_get_model time: %s minutes' % ((end1 - start1) / 60))
    # # 4 给无标签数据打标，判别是否传递给分类模型打标
    start2 = time.time()
    rerun_predict_result()
    end2 = time.time()
    print('rerun_predict_result time: %s minutes' % ((end2 - start2) / 60))

# nohup python -u main.py > log.log 2>&1 &
