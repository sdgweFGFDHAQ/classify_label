# 终端门店自动分类打标
标签体系为5.0
## 数据来源
#### 训练集
```
数据库：数据湖仓
数据表：standard_db.di_store_classify_dedupe
```
#### 预测集
```
数据库：clickhouse
数据表：ai.ods_di_store_classify
```

## 项目运行命令
```
# nohup python -u deep_learning_bert.py > deep_learning_bert.log 2>&1 &
# nohup python -u readCK.py > readCK.log 2>&1 &
```
## 训练自动分类算法模型接口：运行deep_learning_bert.py
### main函数调用的各个方法：
```
1.sql查询下载数据集、下载基于无效关键词集的噪声数据集
    download_data()
    download_invalid_data()
2.按城市划分的数据合并为指定数量csv
    rerun_get_file()
3.按类别抽取样本平衡的训练集，并从噪声集抽取被划分为“其他”类别的训练集
    random_get_trainset(is_labeled=True, labeled_is_all=False)
    add_other_category()
4.训练基于bert的分类模型
    rerun_get_model()
5.加载模型 预测结果
    # rerun_predict_result()
6.绘制可视化图表进行分析
    # draw_trend(model_fit)
```

## 门店类别分类(预测)接口：运行readCK.py
### main函数调用的各个方法：
```
1.sql查询获取城市集合
    city_list = get_cities()
2.条件查询下载数据集，每查询100w条就保存为一个csv
    get_data_offset(file_prefix='store_CK_data_')
3.中文文本清洗、分词
    rerun_get_CK_file(city_list)
4.加载模型 预测结果
    predict_result_forCK_bert()
5.分类算法预测类别，建表并上传数据
    upload_predict_data()
```

## 数据参数接口：
#### 数据库连接配置
```python
conf = {
    'user': 'hive',
    'password': 'xwbigdata2022',
    'server_host': '124.71.220.115', # 外网
    # 'server_host': '192.168.0.150', # 内网
    'port': 10015,
    'db': 'standard_db'
} # 数据湖仓

conf = {
    'user': 'default',
    'password': 'xwclickhouse2022',
    'server_host': '139.9.51.13',
    'port': 9090,
    'db': 'ai_db'
} # clickhouse
```
#### 数据表配置
全量更新数据
```python
# 访问数据表
download_table_name = 'ods_di_store'
# 保存预测结果数据表
upload_table_name = 'ods_di_store_labeling'
```
####  暂时写死的静态配置
```python
# 下载数据所需字段
columns = ['id', 'name', 'city']
# 保存数据字段
columns = ['id', 'name', 'city', 'predict_category']

```
