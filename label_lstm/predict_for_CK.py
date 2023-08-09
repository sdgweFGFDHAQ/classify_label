import os
import time
from multiprocessing import Pool

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import warnings

from transformers import AutoModel, AutoTokenizer

from global_parameter import StaticParameter as SP
from mini_tool import WordSegment, error_callback

warnings.filterwarnings("ignore", category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"


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


# 读取原始文件,将数据格式标准化
# 批量标准化
def set_file_standard_data(city_list, part_i):
    for city in city_list:
        path_city = SP.PATH_ZZX_DATA + city + '.csv'
        path_part = SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(part_i) + '.csv'
        if os.path.exists(path_city):
            csv_data = pd.read_csv(path_city, usecols=['id', 'name', 'city'])
            # 得到标准数据
            segment = WordSegment()
            csv_data['cut_name'] = csv_data['name'].apply(segment.cut_word)
            # 过滤非中文店名导致的'cut_name'=nan
            csv_data = csv_data[csv_data['cut_name'].notna()]
            if os.path.exists(path_part) and os.path.getsize(path_part):
                csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(part_i) + '.csv',
                                columns=['id', 'name', 'city', 'cut_name'],
                                mode='a', header=False)
            else:
                csv_data.to_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(part_i) + '.csv',
                                columns=['id', 'name', 'city', 'cut_name'],
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
                                   usecols=['id', 'name', 'city'])
            # 得到标准数据
            segment = WordSegment()
            csv_data['cut_name'] = csv_data['name'].apply(segment.cut_word)
            if os.path.exists(path_part) and os.path.getsize(path_part):
                csv_data.to_csv(path_part,
                                columns=['id', 'name', 'category3_new', 'cut_name'], mode='a', header=False)
            else:
                csv_data.to_csv(path_part,
                                columns=['id', 'name', 'category3_new', 'cut_name'], mode='w')


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
            {'store_id': df['id'], 'name': df['name'], 'city': df['city'],
             'predict_category': cate_lists})
        result.to_csv(SP.PATH_ZZX_PREDICT_DATA + 'predict_CK_category_' + str(part_i) + '.csv')
    except Exception as e:
        with open('error_city.txt', 'a') as ef:
            ef.write(str(time.time()))
            ef.write('出错的city: ' + str(part_i) + '; 异常e:' + str(e))


def predict_result_forCK_bert():
    for csv_i in range(SP.SEGMENT_NUMBER):
        path_pre = SP.PATH_ZZX_PREDICT_DATA + 'predict_CK_category_' + str(csv_i) + '.csv'
        if os.path.exists(path_pre):
            open(path_pre, "r+").truncate()

    cat_df = pd.read_csv('category_to_id.csv')
    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)

    # bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    # category_number = cat_df.shape[0]  # 78
    # lstm_model = BertLSTMNet(
    #     bert_embedding=bert_layer,
    #     input_dim=768,
    #     hidden_dim=128,
    #     num_classes=category_number,
    #     num_layers=2
    # ).to(device)
    # lstm_model.load_state_dict(torch.load("best_lstm_bert.pth"))
    lstm_model = torch.load('best_lstm_bert_3.model')

    for part_i in range(SP.SEGMENT_NUMBER):
        df = pd.read_csv(SP.PATH_ZZX_STANDARD_DATA + 'standard_CK_store_' + str(part_i) + '.csv')
        df = df[(df['cut_name'].notna() & df['cut_name'].notnull())]

        # 处理特征
        encoded_dict = tokenizer.batch_encode_plus(df['cut_name'].tolist(),
                                                   add_special_tokens=True, max_length=12, padding='max_length',
                                                   truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        pre_x = TensorDataset(input_ids, attention_masks)

        pre_dataloder = DataLoader(dataset=pre_x, batch_size=512, shuffle=False, drop_last=False)

        idx2lab = dict(zip(cat_df['cat_id'], cat_df['category1_new']))
        predict_result(df, pre_dataloder, lstm_model, idx2lab, part_i)
        print("predict_CK_category_{} 写入完成".format(part_i))


if __name__ == '__main__':
    # cities = ['江门市', '新乡市', '河源市', '潮州市', '湛江市', '肇庆市', '开封市', '广州市', '安阳市', '茂名市',
    #           '南阳市', '焦作市',
    #           '漯河市', '深圳市', '韶关市', '驻马店市', '商丘市', '汕头市', '许昌市', '揭阳市', '郑州市', '汕尾市',
    #           '惠州市', '平顶山市',
    #           '清远市', '济源市', '洛阳市', '周口市', '云浮市', '珠海市', '三门峡市', '鹤壁市', '信阳市', '佛山市',
    #           '梅州市', '濮阳市',
    #           '徐州市', '宿迁市', '无锡市', '盐城市', '泰州市', '齐齐哈尔市', '常州市', '黑河市', '大庆市', '镇江市',
    #           '扬州市', '鸡西市',
    #           '苏州市', '七台河市', '大兴安岭地区', '南通市', '鹤岗市', '南京市', '牡丹江市', '佳木斯市', '绥化市',
    #           '伊春市', '淮安市',
    #           '双鸭山市', '连云港市', '哈尔滨市', '随州市', '恩施土家族苗族自治州', '武汉市', '宜昌市', '杭州市',
    #           '黄冈市', '台州市',
    #           '温州市', '咸宁市', '鄂州市', '荆门市', '襄阳市', '舟山市', '神农架林区', '宁波市', '丽水市', '黄石市',
    #           '孝感市', '十堰市',
    #           '天门市', '荆州市', '仙桃市', '湖州市', '潜江市', '定安县', '本溪市', '辽阳市', '屯昌县', '朝阳市',
    #           '铁岭市', '锦州市',
    #           '阜新市', '儋州市', '临高县', '白沙黎族自治县', '鞍山市', '文昌市', '海口市', '陵水黎族自治县',
    #           '保亭黎族苗族自治县',
    #           '乐东黎族自治县', '琼海市', '葫芦岛市', '澄迈县', '万宁市', '五指山市', '三亚市', '丹东市', '抚顺市',
    #           '大连市', '益阳市',
    #           '昌江黎族自治县', '沈阳市', '三沙市', '北京城区', '营口市', '东方市', '盘锦市', '琼中黎族苗族自治县',
    #           '景德镇市',
    #           '黔南布依族苗族自治州', '中卫市', '南昌市', '石嘴山市', '贵阳市', '黔东南苗族侗族自治州', '九江市',
    #           '吴忠市', '六盘水市',
    #           '黔西南布依族苗族自治州', '上饶市', '抚州市', '银川市', '新余市', '毕节市', '吉安市', '遵义市', '铜仁市',
    #           '安顺市', '宜春市',
    #           '鹰潭市', '固原市', '萍乡市', '赣州市', '滨州市', '潍坊市', '聊城市', '济宁市', '济南市', '青岛市',
    #           '东营市', '威海市',
    #           '枣庄市', '烟台市', '菏泽市', '泰安市', '临沂市', '淄博市', '德州市', '日照市', '乌兰察布市', '保山市',
    #           '呼伦贝尔市',
    #           '鄂尔多斯市', '普洱市', '玉溪市', '临沧市', '三明市', '漳州市', '呼和浩特市', '曲靖市', '龙岩市',
    #           '迪庆藏族自治州', '通辽市',
    #           '楚雄彝族自治州', '宁德市', '泉州市', '阿拉善盟', '大理白族自治州', '南平市', '文山壮族苗族自治州',
    #           '丽江市', '包头市',
    #           '西双版纳傣族自治州', '乌海市', '昭通市', '怒江傈僳族自治州', '莆田市', '巴彦淖尔市', '厦门市',
    #           '德宏傣族景颇族自治州', '昆明市',
    #           '红河哈尼族彝族自治州', '兴安盟', '福州市', '赤峰市', '锡林郭勒盟', '澳门', '黄山市', '淮北市', '六安市',
    #           '宣城市', '合肥市',
    #           '铜陵市', '宿州市', '滁州市', '蚌埠市', '马鞍山市', '亳州市', '芜湖市', '阜阳市', '池州市', '安庆市',
    #           '淮南市', '沧州市',
    #           '保定市', '衡水市', '邢台市', '廊坊市', '邯郸市', '承德市', '秦皇岛市', '张家口市', '唐山市', '石家庄市',
    #           '铜川市',
    #           '榆林市', '渭南市', '延安市', '汉中市', '宝鸡市', '安康市', '西安市', '咸阳市', '商洛市',
    #           '玉树藏族自治州', '海东市',
    #           '巴中市', '辽源市', '延边朝鲜族自治州', '四平市', '遂宁市', '凉山彝族自治州', '海西蒙古族藏族自治州',
    #           '绵阳市', '海北藏族自治州',
    #           '泸州市', '白山市', '达州市', '眉山市', '阿坝藏族羌族自治州', '吉林市', '黄南藏族自治州', '内江市',
    #           '海南藏族自治州', '成都市',
    #           '广安市', '自贡市', '通化市', '长春市', '白城市', '南充市', '乐山市', '德阳市', '资阳市',
    #           '甘孜藏族自治州', '攀枝花市',
    #           '宜宾市', '松原市', '广元市', '雅安市', '果洛藏族自治州', '西宁市', '东莞市', '中山市', '湘潭市',
    #           '百色市', '玉林市',
    #           '怀化市', '防城港市', '河池市', '梧州市', '岳阳市', '郴州市', '钦州市', '崇左市', '常德市', '株洲市',
    #           '北海市', '柳州市',
    #           '桂林市', '张家界市', '娄底市', '永州市', '湘西土家族苗族自治州', '长沙市', '来宾市', '衡阳市', '邵阳市',
    #           '南宁市', '兰州市',
    #           '甘南藏族自治州', '金昌市', '酒泉市', '张掖市', '白银市', '嘉峪关市', '武威市', '天水市', '庆阳市',
    #           '临夏回族自治州',
    #           '陇南市', '平凉市', '定西市', '忻州市', '吕梁市', '阳泉市', '太原市', '长治市', '运城市', '临汾市',
    #           '晋城市', '晋中市',
    #           '贵港市', '贺州市', '朔州市', '大同市', '上海城区', '日喀则市', '五家渠市', '昌吉回族自治州', '那曲市',
    #           '阿里地区',
    #           '胡杨河市', '石河子市', '北屯市', '克拉玛依市', '克孜勒苏柯尔克孜自治州', '乌鲁木齐市', '山南市',
    #           '阿克苏地区',
    #           '博尔塔拉蒙古自治州', '吐鲁番市', '哈密市', '阿拉尔市', '双河市', '可克达拉市', '林芝市', '铁门关市',
    #           '喀什地区', '塔城地区',
    #           '天津城区', '伊犁哈萨克自治州', '拉萨市', '和田地区', '巴音郭楞蒙古自治州', '阿勒泰地区', '昆玉市',
    #           '图木舒克市', '昌都市',
    #           '重庆郊县', '重庆城区', '香港', '阳江市', '金华市', '嘉兴市', '衢州市', '绍兴市']
    cities = []
    # pred预测集
    rerun_get_CK_file(cities)

    predict_result_forCK_bert()
