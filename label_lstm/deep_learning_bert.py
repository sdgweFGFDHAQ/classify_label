import os
import time
import warnings
import logging

import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import torch
from sklearn.model_selection import KFold, train_test_split
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

from model_bert import BertLSTMNet_1
from global_parameter import StaticParameter as SP
from mini_tool import WordSegment, error_callback

warnings.filterwarnings("ignore", category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter('/home/DI/zhouzx/code/classify_label/logs/')

pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"
category_number = 0


# 批量标准化
def get_city(city_list, i):
    for city in city_list:
        set_file_standard_data(city, i)


# 读取原始文件,将数据格式标准化
def set_file_standard_data(city, part_i):
    path_city = SP.PATH_ZZX_DATA + city + '.csv'
    path_part = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv'
    if os.path.exists(path_city):
        csv_data = pd.read_csv(path_city, usecols=['id', 'name', 'category1_new'])
        # 得到标准数据
        use_columns = ['id', 'name', 'category1_new', 'cut_name']
        segment = WordSegment()
        csv_data['cut_name'] = csv_data['name'].apply(segment.cut_word)
        # 过滤非中文店名导致的'cut_name'=nan
        csv_data = csv_data[csv_data['cut_name'].notna()]
        if os.path.exists(path_part) and os.path.getsize(path_part):
            csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv',
                            columns=use_columns, mode='a', header=False)
        else:
            csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv',
                            columns=use_columns, mode='w')


def typicalsamling(group, threshold):
    count = len(group.index)
    percent = 0.5  # 超参数，考虑缓解样本不平衡但不过多增加训练集 选择的比例
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
    use_columns = ['id', 'name', 'category1_new', 'cut_name']
    standard_df = pd.DataFrame(columns=use_columns)
    result_path = 'standard_store_data.csv'
    all_fix = ''
    for i in range(SP.SEGMENT_NUMBER):
        path = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(i) + '.csv'
        df_i = pd.read_csv(path, usecols=use_columns, keep_default_na=False)
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


# 在训练集中加入'其他'类别的数据
def add_other_category(standard_path='standard_store_data.csv', kw_path='standard_data_other_category.csv',
                       save_path='standard_dataset.csv'):
    prefix = SP.PATH_ZZX_STANDARD_DATA
    use_columns = ['id', 'name', 'category1_new']

    # 训练集
    standard_df = pd.read_csv(prefix + standard_path, usecols=use_columns + ['cut_name'], keep_default_na=False)

    # 不在分类范围的'其他'集-根据关键词查找
    kw_df = pd.read_csv(prefix + kw_path, usecols=use_columns, keep_default_na=False)
    kw_df['category1_new'] = '其他'
    segment = WordSegment()
    kw_df['cut_name'] = kw_df['name'].apply(segment.cut_word)
    # 过滤非中文店名导致的'cut_name'=nan
    kw_df = kw_df[kw_df['cut_name'].notna()]

    temp_df = pd.concat([standard_df, kw_df], axis=0)

    # 人工添加'其他'集
    ao_df = pd.read_csv(prefix + 'arti_other_category.csv', usecols=use_columns, keep_default_na=False)
    segment = WordSegment()
    ao_df['cut_name'] = ao_df['name'].apply(segment.cut_word)
    ao_df = ao_df[ao_df['cut_name'].notna()]
    ao_df_copy = pd.concat([ao_df] * 10, ignore_index=True)

    new_df = pd.concat([temp_df, ao_df_copy], axis=0)

    # 合并作为模型训练集
    new_df.to_csv(prefix + save_path, index=False)


def get_dataset(source_csv='standard_dataset.csv'):
    gz_df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + source_csv)
    print(len(gz_df.index))

    category_df = gz_df.drop_duplicates(subset=['category1_new'], keep='first', inplace=False)
    category_df['cat_id'] = category_df['category1_new'].factorize()[0]
    cat_df = category_df[['category1_new', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(
        drop=True)
    cat_df.to_csv('category_to_id.csv')

    data_x, data_y = gz_df['cut_name'].values, gz_df['category1_new'].values
    category_classes = gz_df['category1_new'].unique()

    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)

    # 遍历数据集的每一行
    # 处理特征
    encoded_dict = tokenizer.batch_encode_plus(gz_df['cut_name'].tolist(),
                                               add_special_tokens=True, max_length=12, padding='max_length',
                                               truncation=True, return_tensors='pt')
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']

    # 处理类别
    lab2idx = dict(zip(cat_df['category1_new'], cat_df['cat_id']))
    # idx2lab = dict(zip(cat_df['cat_id'], cat_df['category1_new']))
    label2id_list = [lab2idx[lab] for lab in data_y]

    dataset = TensorDataset(input_ids, torch.Tensor(label2id_list), attention_masks)
    return dataset, bert_layer, tokenizer, len(category_classes)


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
    pred_list = torch.argmax(pred_y, dim=1)
    correct = (pred_list == y).float()
    acc = correct.sum() / len(correct)
    return acc


def training(train_loader, model):
    # 多分类损失函数
    criterion = nn.CrossEntropyLoss()
    # crit = nn.CrossEntropyLoss(reduction='sum')
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
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
    print('\nTrain | Loss:{:.5f} Acc: {:.3f}%'.format(loss_value, acc_value))
    return loss_value, acc_value


def evaluating(val_loader, model):
    # 多分类损失函数
    criterion = nn.CrossEntropyLoss()
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
        print("Valid | Loss:{:.5f} Acc: {:.3f}% ".format(loss_value, acc_value))
    print('-----------------------------------')
    return loss_value, acc_value


# def search_best_dataset(dataset, embedding, category_count):
#     # 使用k折交叉验证
#     kf_5 = KFold(n_splits=5)
#     k, epochs = 0, 3
#     best_accuracy = 0.
#     best_train, best_test = None, None
#     for fold, (train_idx, val_idx) in enumerate(kf_5.split(dataset)):
#         # data_x, data_y = np.array(data_x), np.array(data_y)
#         train_dataset = torch.utils.data.Subset(dataset, train_idx)
#         val_dataset = torch.utils.data.Subset(dataset, val_idx)
#
#         print('==================第{}折================'.format(fold + 1))
#         k += 1
#         model = BertLSTMNet_1(
#             bert_embedding=embedding,
#             input_dim=768,
#             hidden_dim=128,
#             num_classes=category_count,
#             num_layers=2
#         ).to(device)
#         train_ip = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, drop_last=True)
#         test_ip = DataLoader(dataset=val_dataset, batch_size=512, shuffle=True, drop_last=False)
#         accuracy_list = list()
#         # run epochs
#         for ep in range(epochs):
#             training(train_ip, model)
#             _, pre_av = evaluating(test_ip, model)
#             accuracy_list.append(round(pre_av, 3))
#         mean_accuracy = np.mean(accuracy_list)
#         if mean_accuracy > best_accuracy:
#             best_accuracy = mean_accuracy
#             best_train, best_test = train_dataset, val_dataset
#     return best_train, best_test


def search_best_model(train_set, test_set, embedding, category_count):
    model = BertLSTMNet_1(
        bert_embedding=embedding,
        input_dim=768,
        hidden_dim=128,
        num_classes=category_count,
        num_layers=2
    ).to(device)
    train_ip = DataLoader(dataset=train_set, batch_size=512, shuffle=True, drop_last=True)
    test_ip = DataLoader(dataset=test_set, batch_size=512, shuffle=True, drop_last=False)
    # run epochs
    best_accuracy = 0.
    for ep in range(15):
        print('==========train epoch: {}============'.format(ep))
        train_lv, train_av = training(train_ip, model)
        eval_lv, eval_av = evaluating(test_ip, model)
        if eval_av > best_accuracy:
            best_accuracy = eval_av
            torch.save(model, "best_bert_category.model")
            # torch.save(model.state_dict(), "model/best_lstm_bert.pth")
        writer.add_scalars('acc', {'train_acc': train_av, 'test_acc': eval_av}, global_step=ep)
        writer.add_scalars('loss', {'train_loss': train_lv, 'test_loss': eval_lv}, global_step=ep)


def predict_result(df, dataloder, model, idx2lab, part_i):
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
                pre_label = outputs.argmax(axis=1)
                pre_lists.extend(pre_label)
        cate_lists = []
        for ind in pre_lists:
            cate_lists.append(idx2lab[ind.item()])
        result = pd.DataFrame(
            {'store_id': df['id'], 'name': df['name'], 'category1_new': df['category1_new'],
             'predict_category': cate_lists})
        result.to_csv(SP.PATH_ZZX_PREDICT_DATA + 'predict_category_' + str(part_i) + '.csv')
    except Exception as e:
        with open('error_city.txt', 'a') as ef:
            ef.write(str(time.time()))
            ef.write('出错的city: ' + str(part_i) + '; 异常e:' + str(e))


# 用于重新切分店名，生成标准文件
def rerun_get_file():
    cities = SP.CITIES
    # 初始化，清空文件
    for csv_i in range(SP.SEGMENT_NUMBER):
        path_sta = SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(csv_i) + '.csv'
        if os.path.exists(path_sta):
            open(path_sta, "r+").truncate()
    # 合并城市,分为8部分
    city_num = len(cities)
    index_list = [int(city_num * i / SP.SEGMENT_NUMBER) for i in range(SP.SEGMENT_NUMBER + 1)]
    pool = Pool(processes=4)
    for index in range(len(index_list) - 1):
        cities_i = cities[index_list[index]:index_list[index + 1]]
        pool.apply_async(get_city, args=(cities_i, index), error_callback=error_callback)
    pool.close()
    pool.join()


# 划分合适的训练集测试集，保存训练模型
def rerun_get_model():
    global category_number
    # 训练模型,获取训练集
    # random_get_trainset()
    dataset, embedding_matrix, prepro, class_num = get_dataset()
    category_number = class_num
    # 保存最好的模型
    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=23)
    search_best_model(train_set, test_set, embedding_matrix, class_num)
    # K折交叉验证(停用)
    # train_set, test_set = search_best_dataset(dataset, embedding_matrix, class_num)


# 预测数据
def rerun_predict_result():
    for csv_i in range(SP.SEGMENT_NUMBER):
        path_pre = SP.PATH_ZZX_PREDICT_DATA + 'predict_category_' + str(csv_i) + '.csv'
        if os.path.exists(path_pre):
            open(path_pre, "r+").truncate()
    global category_number
    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    # lstm_model = BertLSTMNet_1(
    #     bert_embedding=bert_layer,
    #     input_dim=768,
    #     hidden_dim=128,
    #     num_classes=category_number,
    #     num_layers=2
    # ).to(device)
    # lstm_model.load_state_dict(torch.load("model/best_lstm_bert.pth"))
    lstm_model = torch.load('best_bert_category.model')
    for part_i in range(SP.SEGMENT_NUMBER):
        df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv')
        df = df[(df['cut_name'].notna() & df['cut_name'].notnull())]

        # 处理特征
        encoded_dict = tokenizer.batch_encode_plus(df['cut_name'].tolist(),
                                                   add_special_tokens=True, max_length=12, padding='max_length',
                                                   truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        pre_x = TensorDataset(input_ids, attention_masks)

        pre_dataloder = DataLoader(dataset=pre_x, batch_size=512, shuffle=False, drop_last=False)

        cat_df = pd.read_csv('category_to_id.csv')
        idx2lab = dict(zip(cat_df['cat_id'], cat_df['category1_new']))
        predict_result(df, pre_dataloder, lstm_model, idx2lab, part_i)
        print("predict_category_{} 写入完成".format(part_i))


def draw_trend(history):
    # 绘制损失函数趋势图
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # 绘制准确率趋势图
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

# nohup python -u main.py > log.log 2>&1 &
