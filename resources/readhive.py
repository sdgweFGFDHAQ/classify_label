import os

from pyhive import hive
import pandas as pd


# from impala.dbapi import connect
# from impala.util import as_pandas
# import sasl

def download_data():
    out_url = "/home/DI/zhouzx/code/classify_label/data/"
    conn = hive.Connection(host='124.71.220.115', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    # conn = hive.Connection(host='192.168.0.150',port=10015,username='ai',password='ai123456',
    #                      database='standard_db',auth='CUSTOM')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT city from standard_db.di_store_classify_dedupe")
    city_list = cursor.fetchall()
    city_df = pd.DataFrame(city_list, columns=["city"])
    cities = city_df["city"].tolist()
    print(cities)
    print("下载数量：", len(cities))
    for cityname in cities:
        if cityname is None:
            continue
        print("开始执行sql")
        cursor.execute(
            "select id,name,cname,namepath,appcode,tags, channeltype_new,category1_new "
            "from standard_db.di_store_classify_dedupe where city=" + "'" + cityname + "'")
        print("已经获取数据")
        data_list = cursor.fetchall()
        df = pd.DataFrame(data_list, columns=["id", "name", "cname", "namepath", "appcode", "tags", "channeltype_new",
                                              "category1_new"]).set_index("id")
        # df = pd.DataFrame(list)
        # print(df)
        df.to_csv(out_url + cityname + ".csv")
        print("写入完成", cityname)
    print("数据全部更新完成！")
    cursor.close()
    conn.close()


def download_invalid_data():
    out_url = "/home/DI/zhouzx/code/classify_label/standard_data/"
    conn = hive.Connection(host='124.71.220.115', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    # conn = hive.Connection(host='192.168.0.150',port=10015,username='ai',password='ai123456',
    #                      database='standard_db',auth='CUSTOM')
    cursor = conn.cursor()
    cursor.execute("select keyword from standard_db.store_keyword_invalid where col='name'")
    invalid_kw_list = cursor.fetchall()
    invalid_kw_df = pd.DataFrame(invalid_kw_list, columns=["keyword"])
    keywords = invalid_kw_df["keyword"].tolist()
    print(keywords)
    print("被分类为'其他'类别的数据关键词：", len(keywords))

    file_name = out_url + 'standard_data_other_category.csv'
    # 创建csv文件
    with open(file_name, 'w') as file:
        file.write('')
    # 合并数据集
    for kw in keywords:
        if kw is None:
            continue
        print("开始执行sql")
        cursor.execute("select count(1) from standard_db.di_store_classify_dedupe where name like " + "'%" + kw + "%'")
        data_count = cursor.fetchall()
        count = data_count[0][0]

        # 每个关键词只取1000条数据
        data_sql = "select id,name,cname,namepath,appcode,tags, channeltype_new,category1_new " \
                   "from standard_db.di_store_classify_dedupe where name like " + "'%" + kw + "%'"
        if count > 1000:
            cursor.execute(data_sql+" limit 1000")
        else:
            cursor.execute(data_sql)
        data_list = cursor.fetchall()

        df = pd.DataFrame(data_list, columns=["id", "name", "cname", "namepath", "appcode", "tags", "channeltype_new",
                                              "category1_new"]).set_index("id")
        if os.path.getsize(file_name):
            df.to_csv(file_name, mode='a', header=False)
        else:
            df.to_csv(file_name, mode='w', header=True)
        print("写入包含关键词({})的数据({})条".format(kw, len(data_list)))
    print("数据全部下载完成！")
    cursor.close()
    conn.close()


if __name__ == '__main__':
    # 预测数据
    # cities = ['江门市', '新乡市', '河源市', '潮州市', '湛江市', '肇庆市', '开封市', '广州市', '安阳市', '茂名市', '南阳市', '焦作市',
    #           '漯河市', '深圳市', '韶关市', '驻马店市', '商丘市', '汕头市', '许昌市', '揭阳市', '郑州市', '汕尾市', '惠州市', '平顶山市',
    #           '清远市', '济源市', '洛阳市', '周口市', '云浮市', '珠海市', '三门峡市', '鹤壁市', '信阳市', '佛山市', '梅州市', '濮阳市',
    #           '徐州市', '宿迁市', '无锡市', '盐城市', '泰州市', '齐齐哈尔市', '常州市', '黑河市', '大庆市', '镇江市', '扬州市', '鸡西市',
    #           '苏州市', '七台河市', '大兴安岭地区', '南通市', '鹤岗市', '南京市', '牡丹江市', '佳木斯市', '绥化市', '伊春市', '淮安市',
    #           '双鸭山市', '连云港市', '哈尔滨市', '随州市', '恩施土家族苗族自治州', '武汉市', '宜昌市', '杭州市', '黄冈市', '台州市',
    #           '温州市', '咸宁市', '鄂州市', '荆门市', '襄阳市', '舟山市', '神农架林区', '宁波市', '丽水市', '黄石市', '孝感市', '十堰市',
    #           '天门市', '荆州市', '仙桃市', '湖州市', '潜江市', '定安县', '本溪市', '辽阳市', '屯昌县', '朝阳市', '铁岭市', '锦州市',
    #           '阜新市', '儋州市', '临高县', '白沙黎族自治县', '鞍山市', '文昌市', '海口市', '陵水黎族自治县', '保亭黎族苗族自治县',
    #           '乐东黎族自治县', '琼海市', '葫芦岛市', '澄迈县', '万宁市', '五指山市', '三亚市', '丹东市', '抚顺市', '大连市', '益阳市',
    #           '昌江黎族自治县', '沈阳市', '三沙市', '北京城区', '营口市', '东方市', '盘锦市', '琼中黎族苗族自治县', '景德镇市',
    #           '黔南布依族苗族自治州', '中卫市', '南昌市', '石嘴山市', '贵阳市', '黔东南苗族侗族自治州', '九江市', '吴忠市', '六盘水市',
    #           '黔西南布依族苗族自治州', '上饶市', '抚州市', '银川市', '新余市', '毕节市', '吉安市', '遵义市', '铜仁市', '安顺市', '宜春市',
    #           '鹰潭市', '固原市', '萍乡市', '赣州市', '滨州市', '潍坊市', '聊城市', '济宁市', '济南市', '青岛市', '东营市', '威海市',
    #           '枣庄市', '烟台市', '菏泽市', '泰安市', '临沂市', '淄博市', '德州市', '日照市', '乌兰察布市', '保山市', '呼伦贝尔市',
    #           '鄂尔多斯市', '普洱市', '玉溪市', '临沧市', '三明市', '漳州市', '呼和浩特市', '曲靖市', '龙岩市', '迪庆藏族自治州', '通辽市',
    #           '楚雄彝族自治州', '宁德市', '泉州市', '阿拉善盟', '大理白族自治州', '南平市', '文山壮族苗族自治州', '丽江市', '包头市',
    #           '西双版纳傣族自治州', '乌海市', '昭通市', '怒江傈僳族自治州', '莆田市', '巴彦淖尔市', '厦门市', '德宏傣族景颇族自治州', '昆明市',
    #           '红河哈尼族彝族自治州', '兴安盟', '福州市', '赤峰市', '锡林郭勒盟', '澳门', '黄山市', '淮北市', '六安市', '宣城市', '合肥市',
    #           '铜陵市', '宿州市', '滁州市', '蚌埠市', '马鞍山市', '亳州市', '芜湖市', '阜阳市', '池州市', '安庆市', '淮南市', '沧州市',
    #           '保定市', '衡水市', '邢台市', '廊坊市', '邯郸市', '承德市', '秦皇岛市', '张家口市', '唐山市', '石家庄市', '铜川市',
    #           '榆林市', '渭南市', '延安市', '汉中市', '宝鸡市', '安康市', '西安市', '咸阳市', '商洛市', '玉树藏族自治州', '海东市',
    #           '巴中市', '辽源市', '延边朝鲜族自治州', '四平市', '遂宁市', '凉山彝族自治州', '海西蒙古族藏族自治州', '绵阳市', '海北藏族自治州',
    #           '泸州市', '白山市', '达州市', '眉山市', '阿坝藏族羌族自治州', '吉林市', '黄南藏族自治州', '内江市', '海南藏族自治州', '成都市',
    #           '广安市', '自贡市', '通化市', '长春市', '白城市', '南充市', '乐山市', '德阳市', '资阳市', '甘孜藏族自治州', '攀枝花市',
    #           '宜宾市', '松原市', '广元市', '雅安市', '果洛藏族自治州', '西宁市', '东莞市', '中山市', '湘潭市', '百色市', '玉林市',
    #           '怀化市', '防城港市', '河池市', '梧州市', '岳阳市', '郴州市', '钦州市', '崇左市', '常德市', '株洲市', '北海市', '柳州市',
    #           '桂林市', '张家界市', '娄底市', '永州市', '湘西土家族苗族自治州', '长沙市', '来宾市', '衡阳市', '邵阳市', '南宁市', '兰州市',
    #           '甘南藏族自治州', '金昌市', '酒泉市', '张掖市', '白银市', '嘉峪关市', '武威市', '天水市', '庆阳市', '临夏回族自治州',
    #           '陇南市', '平凉市', '定西市', '忻州市', '吕梁市', '阳泉市', '太原市', '长治市', '运城市', '临汾市', '晋城市', '晋中市',
    #           '贵港市', '贺州市', '朔州市', '大同市', '上海城区', '日喀则市', '五家渠市', '昌吉回族自治州', '那曲市', '阿里地区',
    #           '胡杨河市', '石河子市', '北屯市', '克拉玛依市', '克孜勒苏柯尔克孜自治州', '乌鲁木齐市', '山南市', '阿克苏地区',
    #           '博尔塔拉蒙古自治州', '吐鲁番市', '哈密市', '阿拉尔市', '双河市', '可克达拉市', '林芝市', '铁门关市', '喀什地区', '塔城地区',
    #           '天津城区', '伊犁哈萨克自治州', '拉萨市', '和田地区', '巴音郭楞蒙古自治州', '阿勒泰地区', '昆玉市', '图木舒克市', '昌都市',
    #           '重庆郊县', '重庆城区', '香港', '阳江市', '金华市', '嘉兴市', '衢州市', '绍兴市']
    # download_data()
    download_invalid_data()
