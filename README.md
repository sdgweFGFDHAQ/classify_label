# 终端门店自动分类打标
标签体系为5.0
## 数据表standard_db.di_store_classify_dedupe  
当前为测试用表，标签字段catecory1_new

## 主要接口：运行readCK.py
main函数注解如下
### 1.sql查询获取城市集合
#### city_list = get_cities()
### 2.条件查询下载数据集，根据地区划分为8个csv文件
#### get_data(city_list)
### 3.中文文本清洗、分词
#### rerun_get_CK_file(city_list)
### 4.加载模型 预测结果
#### predict_result_forCK_bert()
### 5.分类算法预测类别，建表并上传数据
#### upload_predict_data()

## 参数接口：
#### 数据库连接配置
```python
conf = {
    'user': 'default',
    'password': 'xwclickhouse2022',
    'server_host': '139.9.51.13',
    'port': 9090,
    'db': 'ai_db'
}
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
