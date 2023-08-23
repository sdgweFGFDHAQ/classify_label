# -*- encoding: utf-8 -*-
import math
import sys
import time

from absl import app, flags
from clickhouse_sqlalchemy import make_session, get_declarative_base, engines
from sqlalchemy import create_engine, Column, MetaData, types, text
import pandas as pd
import logging

from predict_for_CK import rerun_get_CK_file, predict_result_forCK_bert

# FLAGS = flags.FLAGS
logging.basicConfig(filename="readCK.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)

# flags.DEFINE_string(
#     'data_path', '/home/DI/zhouzx/code/classify_label/data/',
#     'Path')
# flags.DEFINE_string(
#     'predict_path', '/home/DI/zhouzx/code/classify_label/predict_data/',
#     'Path')
# flags.DEFINE_string(
#     'download_table_name', 'ods_di_store_classify',
#     'table')
# flags.DEFINE_string(
#     'upload_table_name', 'ods_di_store_labeling',
#     'table')
# flags.DEFINE_integer(
#     'upload_batch_size', 1000000,
#     'batch_size')

data_path = '/home/DI/zhouzx/code/classify_label/data/'
predict_path = '/home/DI/zhouzx/code/classify_label/predict_data/'
download_table_name = 'ods_di_store_classify'
upload_table_name = 'ods_di_store_labeling'
upload_batch_size = 100000


# 创建ClickhouseClient类
class ClickhouseClient:
    def __init__(self, conf):
        self._connection = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
        self._engine = create_engine(self._connection, pool_size=100, pool_recycle=3600, pool_timeout=20)
        self._session = make_session(self._engine)
        self._metadata = MetaData(bind=self._engine)
        self._base = get_declarative_base(metadata=self._metadata)

    def create_table(self, table_name):
        class Rate(self._base):
            pk = Column(types.Integer, primary_key=True, autoincrement=True)
            id = Column(types.String, primary_key=False)
            name = Column(types.String)
            predict_category = Column(types.String)

            __tablename__ = table_name
            __table_args__ = (
                engines.Memory(),
            )

        if not self._engine.dialect.has_table(self._engine, Rate.__tablename__):
            Rate.__table__.create()
        return Rate

    def insert_data(self, table, data, batch_size=10000):
        session = self._session
        total_rows = len(data)
        num_batches = math.floor(total_rows / batch_size)  # 计算总批次数
        try:
            for i in range(num_batches):
                batch_data = data[i * batch_size:(i + 1) * batch_size]
                session.bulk_insert_mappings(table, batch_data)
            if total_rows % batch_size != 0:
                batch_data = data[num_batches * batch_size:total_rows]
                session.bulk_insert_mappings(table, batch_data)
            session.commit()
            logging.info("Data inserted successfully.")
        except Exception as e:
            session.rollback()
            logging.info(f"Error inserting data: {str(e)}")
        finally:
            session.close()

    def clear_data(self, table):
        session = self._session
        try:
            # 清空数据表
            session.query(table).filter(True).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            logging.info(f"Error inserting data: {str(e)}")
        finally:
            # 关闭会话
            session.close()

    def query_data_with_raw_sql(self, sql):
        try:
            # 使用 text() 函数构建原生 SQL 查询
            query = text(sql)

            # 执行查询并获取结果
            result = self._session.execute(query).fetchall()  # 可以使用.fetchmany(size=50000)优化

            return result
        except Exception as e:
            logging.info(f"Error querying data with raw SQL: {str(e)}")
            return []
        finally:
            self._session.close()


conf = {
    'user': 'default',
    'password': 'xwclickhouse2022',
    'server_host': '139.9.51.13',
    'port': 9090,
    'db': 'ai_db'
}

# 创建clickhouse客户端
clickhouse_client = ClickhouseClient(conf)


# sql查询获取城市集合
def get_cities():
    logging.info("获取城市列表：")
    sql = "SELECT DISTINCT city FROM ods_di_store_classify where category1_new=''"
    result = clickhouse_client.query_data_with_raw_sql(sql)
    cities = []
    for r in result:
        city_name = r[0]
        if city_name is not None and city_name != '':
            cities.append(city_name)
    logging.info(cities)
    logging.info("获取城市列表完成!")
    return cities


# 条件查询
def get_data(cities):
    logging.info("根据城市列表获取指定数据集")
    for cityname in cities:
        try:
            if cityname is None:
                continue
            logging.info("开始执行sql")
            sql = "SELECT id, name FROM ods_di_store_classify WHERE category1_new='' and city='{0}'" \
                .format(cityname)
            data = clickhouse_client.query_data_with_raw_sql(sql)
            data_df = pd.DataFrame(data, columns=['id', 'name'])
            data_df.to_csv(data_path + str(cityname) + '.csv')
        except Exception as e:
            logging.info(str(cityname) + "出错：" + str(e) + "!")
    logging.info("数据集全部下载完成!")


# 分类算法预测类别，建表并上传数据
def upload_predict_data():
    logging.info("预测数据集上传到数据库")
    # 创建表
    table = clickhouse_client.create_table(table_name=upload_table_name)
    # 清空数据表
    clickhouse_client.clear_data(table=table)
    for index in range(8):
        data = pd.read_csv(
            predict_path + 'predict_CK_category_' + str(index) + '.csv',
            usecols=['id', 'name', 'predict_category'],
            index_col=False)
        data_dict = data.to_dict(orient='records')

        # data_list = [list(row.values()) for row in data_dict]
        # 插入数据
        clickhouse_client.insert_data(table=table, data=data_dict, batch_size=upload_batch_size)
    logging.info("写入数据库完成!")


if __name__ == '__main__':
    # sql查询获取城市集合
    # city_list = get_cities()
    # 条件查询划分8个csv文件
    # get_data(city_list)
    start1 = time.time()
    # # 加载模型 预测结果
    # rerun_get_CK_file(city_list)
    predict_result_forCK_bert()
    end1 = time.time()
    logging.info('加载模型 预测结果 time: %s minutes' % ((end1 - start1) / 60))
    # # 分类算法预测类别，建表并上传数据
    upload_predict_data()
# nohup python -u readCK.py > ck_log.log 2>&1 &

# 插入数据
# 将 DataFrame 数据转换为字典列表
# data_dict = tag_analytics_pdf.to_dict(orient='records')
# clickhouse_client.insert_data('store_tags_statistics_local', data_dict)


# # 查询所有数据
# result = clickhouse_client.query_data("store_tags_statistics")
# for row in result:
#     logging.info(row.tag_id, row.tag_name)
#
# # 查询指定列的数据
# result = clickhouse_client.query_data("store_tags_statistics", columns=["tag_id", "tag_name"])
# for row in result:
#     logging.info(row.tag_id)
#
# filters = {"tag_id": 2004}
# result = clickhouse_client.query_data("store_tags_statistics", filters=filters)
# for row in result:
#     logging.info(row.tag_name)
#
# # 编写 ClickHouse 支持的 SQL 查询
# sql = """
#     SELECT *
#     FROM store_tags_statistics
#     WHERE tag_id = 2004 AND num = 70
# """
#
# # 执行查询并打印结果
# result = clickhouse_client.query_data_with_raw_sql(sql)
# logging.info(result)
