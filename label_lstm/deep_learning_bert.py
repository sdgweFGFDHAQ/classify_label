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
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, BertConfig

from model_bert import BertLSTMNet_1, BertLSTMNet_2
from global_parameter import StaticParameter as SP
from mini_tool import WordSegment, error_callback

warnings.filterwarnings("ignore", category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
writer = SummaryWriter('/home/data/temp/zhouzx/classify_label/logs/')

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
        segment = WordSegment()
        csv_data['cut_name'] = csv_data['name'].apply(segment.cut_word)
        # 过滤非中文店名导致的'cut_name'=nan
        csv_data = csv_data[csv_data['cut_name'].notna()]
        if os.path.exists(path_part) and os.path.getsize(path_part):
            csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv',
                            columns=['id', 'name', 'category1_new', 'cut_name'],
                            mode='a', header=False)
        else:
            csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_' + str(part_i) + '.csv',
                            columns=['id', 'name', 'category1_new', 'cut_name'],
                            mode='w')


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


def get_dataset():
    gz_df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_store_data.csv')
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


def search_best_dataset(dataset, embedding, category_count):
    # 使用k折交叉验证
    kf_5 = KFold(n_splits=5)
    k, epochs = 0, 3
    best_accuracy = 0.
    best_train, best_test = None, None
    for fold, (train_idx, val_idx) in enumerate(kf_5.split(dataset)):
        # data_x, data_y = np.array(data_x), np.array(data_y)
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        print('==================第{}折================'.format(fold + 1))
        k += 1
        model = BertLSTMNet_1(
            bert_embedding=embedding,
            input_dim=768,
            hidden_dim=128,
            num_classes=category_count,
            num_layers=2
        ).to(device)
        train_ip = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, drop_last=True)
        test_ip = DataLoader(dataset=val_dataset, batch_size=512, shuffle=True, drop_last=False)
        accuracy_list = list()
        # run epochs
        for ep in range(epochs):
            training(train_ip, model)
            _, pre_av = evaluating(test_ip, model)
            accuracy_list.append(round(pre_av, 3))
        mean_accuracy = np.mean(accuracy_list)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_train, best_test = train_dataset, val_dataset
    return best_train, best_test


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
            # torch.save(model, "best_lstm_bert.model")
            torch.save(model.state_dict(), "best_lstm_bert.pth")
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
    cities = ['江门市', '新乡市', '河源市', '潮州市', '湛江市', '肇庆市', '开封市', '广州市', '安阳市', '茂名市',
              '南阳市', '焦作市',
              '漯河市', '深圳市', '韶关市', '驻马店市', '商丘市', '汕头市', '许昌市', '揭阳市', '郑州市', '汕尾市',
              '惠州市', '平顶山市',
              '清远市', '济源市', '洛阳市', '周口市', '云浮市', '珠海市', '三门峡市', '鹤壁市', '信阳市', '佛山市',
              '梅州市', '濮阳市',
              '徐州市', '宿迁市', '无锡市', '盐城市', '泰州市', '齐齐哈尔市', '常州市', '黑河市', '大庆市', '镇江市',
              '扬州市', '鸡西市',
              '苏州市', '七台河市', '大兴安岭地区', '南通市', '鹤岗市', '南京市', '牡丹江市', '佳木斯市', '绥化市',
              '伊春市', '淮安市',
              '双鸭山市', '连云港市', '哈尔滨市', '随州市', '恩施土家族苗族自治州', '武汉市', '宜昌市', '杭州市',
              '黄冈市', '台州市',
              '温州市', '咸宁市', '鄂州市', '荆门市', '襄阳市', '舟山市', '神农架林区', '宁波市', '丽水市', '黄石市',
              '孝感市', '十堰市',
              '天门市', '荆州市', '仙桃市', '湖州市', '潜江市', '定安县', '本溪市', '辽阳市', '屯昌县', '朝阳市',
              '铁岭市', '锦州市',
              '阜新市', '儋州市', '临高县', '白沙黎族自治县', '鞍山市', '文昌市', '海口市', '陵水黎族自治县',
              '保亭黎族苗族自治县',
              '乐东黎族自治县', '琼海市', '葫芦岛市', '澄迈县', '万宁市', '五指山市', '三亚市', '丹东市', '抚顺市',
              '大连市', '益阳市',
              '昌江黎族自治县', '沈阳市', '三沙市', '北京城区', '营口市', '东方市', '盘锦市', '琼中黎族苗族自治县',
              '景德镇市',
              '黔南布依族苗族自治州', '中卫市', '南昌市', '石嘴山市', '贵阳市', '黔东南苗族侗族自治州', '九江市',
              '吴忠市', '六盘水市',
              '黔西南布依族苗族自治州', '上饶市', '抚州市', '银川市', '新余市', '毕节市', '吉安市', '遵义市', '铜仁市',
              '安顺市', '宜春市',
              '鹰潭市', '固原市', '萍乡市', '赣州市', '滨州市', '潍坊市', '聊城市', '济宁市', '济南市', '青岛市',
              '东营市', '威海市',
              '枣庄市', '烟台市', '菏泽市', '泰安市', '临沂市', '淄博市', '德州市', '日照市', '乌兰察布市', '保山市',
              '呼伦贝尔市',
              '鄂尔多斯市', '普洱市', '玉溪市', '临沧市', '三明市', '漳州市', '呼和浩特市', '曲靖市', '龙岩市',
              '迪庆藏族自治州', '通辽市',
              '楚雄彝族自治州', '宁德市', '泉州市', '阿拉善盟', '大理白族自治州', '南平市', '文山壮族苗族自治州',
              '丽江市', '包头市',
              '西双版纳傣族自治州', '乌海市', '昭通市', '怒江傈僳族自治州', '莆田市', '巴彦淖尔市', '厦门市',
              '德宏傣族景颇族自治州', '昆明市',
              '红河哈尼族彝族自治州', '兴安盟', '福州市', '赤峰市', '锡林郭勒盟', '澳门', '黄山市', '淮北市', '六安市',
              '宣城市', '合肥市',
              '铜陵市', '宿州市', '滁州市', '蚌埠市', '马鞍山市', '亳州市', '芜湖市', '阜阳市', '池州市', '安庆市',
              '淮南市', '沧州市',
              '保定市', '衡水市', '邢台市', '廊坊市', '邯郸市', '承德市', '秦皇岛市', '张家口市', '唐山市', '石家庄市',
              '铜川市',
              '榆林市', '渭南市', '延安市', '汉中市', '宝鸡市', '安康市', '西安市', '咸阳市', '商洛市',
              '玉树藏族自治州', '海东市',
              '巴中市', '辽源市', '延边朝鲜族自治州', '四平市', '遂宁市', '凉山彝族自治州', '海西蒙古族藏族自治州',
              '绵阳市', '海北藏族自治州',
              '泸州市', '白山市', '达州市', '眉山市', '阿坝藏族羌族自治州', '吉林市', '黄南藏族自治州', '内江市',
              '海南藏族自治州', '成都市',
              '广安市', '自贡市', '通化市', '长春市', '白城市', '南充市', '乐山市', '德阳市', '资阳市',
              '甘孜藏族自治州', '攀枝花市',
              '宜宾市', '松原市', '广元市', '雅安市', '果洛藏族自治州', '西宁市', '东莞市', '中山市', '湘潭市',
              '百色市', '玉林市',
              '怀化市', '防城港市', '河池市', '梧州市', '岳阳市', '郴州市', '钦州市', '崇左市', '常德市', '株洲市',
              '北海市', '柳州市',
              '桂林市', '张家界市', '娄底市', '永州市', '湘西土家族苗族自治州', '长沙市', '来宾市', '衡阳市', '邵阳市',
              '南宁市', '兰州市',
              '甘南藏族自治州', '金昌市', '酒泉市', '张掖市', '白银市', '嘉峪关市', '武威市', '天水市', '庆阳市',
              '临夏回族自治州',
              '陇南市', '平凉市', '定西市', '忻州市', '吕梁市', '阳泉市', '太原市', '长治市', '运城市', '临汾市',
              '晋城市', '晋中市',
              '贵港市', '贺州市', '朔州市', '大同市', '上海城区', '日喀则市', '五家渠市', '昌吉回族自治州', '那曲市',
              '阿里地区',
              '胡杨河市', '石河子市', '北屯市', '克拉玛依市', '克孜勒苏柯尔克孜自治州', '乌鲁木齐市', '山南市',
              '阿克苏地区',
              '博尔塔拉蒙古自治州', '吐鲁番市', '哈密市', '阿拉尔市', '双河市', '可克达拉市', '林芝市', '铁门关市',
              '喀什地区', '塔城地区',
              '天津城区', '伊犁哈萨克自治州', '拉萨市', '和田地区', '巴音郭楞蒙古自治州', '阿勒泰地区', '昆玉市',
              '图木舒克市', '昌都市',
              '重庆郊县', '重庆城区', '香港', '阳江市', '金华市', '嘉兴市', '衢州市', '绍兴市']
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
    train_set, test_set = train_test_split(dataset, test_size=0.2)
    search_best_model(train_set, test_set, embedding_matrix, class_num)
    # K折交叉验证(未使用，停用)
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
    lstm_model = BertLSTMNet_1(
        bert_embedding=bert_layer,
        input_dim=768,
        hidden_dim=128,
        num_classes=category_number,
        num_layers=2
    ).to(device)
    lstm_model.load_state_dict(torch.load("best_lstm_bert.pth"))
    # lstm_model = torch.load('best_lstm_bert.model')
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


if __name__ == '__main__':
    start0 = time.time()
    # 1 用于重新切分店名，生成标准文件
    # rerun_get_file()
    end0 = time.time()
    print('rerun_get_file time: %s minutes' % ((end0 - start0) / 60))
    # 2 随机抽取带标签训练集
    random_get_trainset(is_labeled=True, labeled_is_all=False)
    # 3 划分合适的训练集测试集，保存训练模型
    start1 = time.time()
    rerun_get_model()
    end1 = time.time()
    print('rerun_get_model time: %s minutes' % ((end1 - start1) / 60))
    # # 4 用于重新预测打标，生成预测文件
    start2 = time.time()
    rerun_predict_result()
    end2 = time.time()
    print('rerun_predict_result time: %s minutes' % ((end2 - start2) / 60))
    # 绘制收敛次数图像
    # draw_trend(model_fit)

# nohup python -u main.py > log.log 2>&1 &
