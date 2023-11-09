import os
import time

from pyhive import hive
import pandas as pd

from deep_learning_bert import random_get_trainset, add_other_category, rerun_get_model, rerun_get_file


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
    start0 = time.time()
    # 1 下载训练集
    download_data()
    download_invalid_data()
    end0 = time.time()
    print('rerun_get_file time: %s minutes' % ((end0 - start0) / 60))

    start1 = time.time()
    # 2 按城市划分的数据合并为指定数量csv
    rerun_get_file()
    end1 = time.time()
    print('rerun_get_file time: %s minutes' % ((end1 - start1) / 60))

    start2 = time.time()
    # 3 随机抽取带标签训练集
    random_get_trainset(is_labeled=True, labeled_is_all=False)
    add_other_category()
    end2 = time.time()
    print('rerun_get_file time: %s minutes' % ((end2 - start2) / 60))

    start3 = time.time()
    # 4 划分合适的训练集测试集，保存训练模型
    rerun_get_model()
    end3 = time.time()
    print('rerun_get_model time: %s minutes' % ((end3 - start3) / 60))

    start4 = time.time()
    # 5 用于重新预测打标，生成预测文件
    # rerun_predict_result()
    end4 = time.time()
    print('rerun_predict_result time: %s minutes' % ((end4 - start4) / 60))
    # 绘制收敛次数图像
    # draw_trend(model_fit)

# nohup python -u main.py > log.log 2>&1 &
