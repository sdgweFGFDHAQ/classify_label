# -*- encoding: utf-8 -*-
"""
@file    :   analytics_tags.py
@contact :   linpeixin@wxchina.com
@license :   (c)copyright 2023-2025

@modify time      @author    @version    @description
------------      -------    --------    -----------
2023/7/17 11:02   lpx        1.0         none
"""
import os
import time

from clickhouse_sqlalchemy import make_session, get_declarative_base
from sqlalchemy import create_engine, Column, types, and_, text
import pandas as pd
import logging

from predict_for_CK import rerun_get_CK_file, predict_result_forCK_bert

# sys.path.append("/home/data/temp/zhouzx/classify_label/label_lstm")
# from predict_for_CK import rerun_get_CK_file, predict_result_forCK_bert

logging.basicConfig(filename="readCK.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)


# 创建ClickhouseClient类
class ClickhouseClient:
    def __init__(self, conf):
        self._connection = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
        self._engine = create_engine(self._connection, pool_size=100, pool_recycle=3600, pool_timeout=20)
        self._session = make_session(self._engine)
        self._base = get_declarative_base()

    def create_table(self, table_name):
        class Table(self._base):
            __tablename__ = table_name
            __table_args__ = {'extend_existing': True}
            id = Column(types.String, primary_key=True)
            name = Column(types.String)
            city = Column(types.String)
            predict_category = Column(types.String)

        self._base.metadata.create_all(self._engine)

        return Table

    def insert_data(self, table_name, data, batch_size=100000):
        session = self._session

        try:
            table = self.create_table(table_name)
            batch = []
            count = 0

            for item in data:
                row = table(**item)
                batch.append(row)
                count += 1

                if count >= batch_size:
                    session.add_all(batch)
                    session.commit()
                    print(f"Inserted {count} records.")
                    batch = []
                    count = 0

            if batch:
                session.add_all(batch)
                session.commit()
                print(f"Inserted {count} records.")

            print("Data inserted successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting data: {str(e)}")
        finally:
            session.close()

    def query_data(self, table_name, columns=None, filters=None):
        session = self._session

        try:
            table = self.create_table(table_name)
            query = session.query(table)

            # 如果指定了列名，则只查询指定列，否则查询所有列
            if columns:
                query = query.with_entities(*[getattr(table, col) for col in columns])

            # 如果指定了条件，则加入到查询中
            # 如果传入了 filters 参数，并且为字典类型，则动态生成过滤条件

            if filters and isinstance(filters, dict):
                condition = None
                for column, value in filters.items():
                    col = table.__table__.columns[column]
                    if condition is None:
                        condition = col == value
                    else:
                        condition = and_(condition, col == value)
                if condition is None:
                    pass
                else:
                    query = query.filter(condition)

            result = query.all()
            return result
        except Exception as e:
            print(f"Error querying data: {str(e)}")
        finally:
            session.close()

    def query_data_with_raw_sql(self, sql):
        try:
            # 使用 text() 函数构建原生 SQL 查询
            query = text(sql)

            # 执行查询并获取结果
            result = self._session.execute(query).fetchall()

            return result
        except Exception as e:
            print(f"Error querying data with raw SQL: {str(e)}")
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


# 插入数据
# 将 DataFrame 数据转换为字典列表
# data_dict = tag_analytics_pdf.to_dict(orient='records')
# clickhouse_client.insert_data('store_tags_statistics_local', data_dict)


# # 查询所有数据
# result = clickhouse_client.query_data("store_tags_statistics")
# for row in result:
#     print(row.tag_id, row.tag_name)
#
# # 查询指定列的数据
# result = clickhouse_client.query_data("store_tags_statistics", columns=["tag_id", "tag_name"])
# for row in result:
#     print(row.tag_id)
#
# filters = {"tag_id": 2004}
# result = clickhouse_client.query_data("store_tags_statistics", filters=filters)
# for row in result:
#     print(row.tag_name)
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
# print(result)


# sql查询获取城市集合
def get_cities():
    logging.info("获取城市列表：")
    sql = "SELECT DISTINCT city FROM ods_di_store"
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
            print("开始执行sql")
            columns = ['id', 'name', 'city']
            filters = {'city': cityname}
            data = clickhouse_client.query_data("ods_di_store", columns=columns, filters=filters)
            data_df = pd.DataFrame(data, columns=['id', 'name', 'city'])
            data_df.to_csv('/home/DI/zhouzx/code/classify_label/data/' + str(cityname) + '.csv')
        except Exception as e:
            print(str(cityname) + "出错：" + str(e) + "!")
    logging.info("数据集全部下载完成!")


# 分类算法预测类别，建表并上传数据
def upload_predict_data():
    logging.info("预测数据集上传到数据库")
    for index in range(8):
        data = pd.read_csv(
            '/home/DI/zhouzx/code/classify_label/predict_data/predict_CK_category_' + str(index) + '.csv',
            index_col=False)
        data_dict = data.to_dict(orient='records')
        data_list = [list(row.values()) for row in data_dict]
        clickhouse_client.insert_data(table_name='ods_di_store_labeling', data=data_list)
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
