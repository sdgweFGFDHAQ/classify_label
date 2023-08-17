#!/bin/bash

# 使用当前python环境
export python=python

# 系统基础配置
export CUDA_VISIBLE_DEVICES=0

# 文件路径配置
export Root_Dir="/home/DI/zhouzx/code/classify_label/" # 项目工程根目录
export DATA_DIR="${Root_Dir}/data/"                    # 原始数据的保存路径
export STANDARD_DATA_DIR="${Root_Dir}/standard_data/"  # 清洗后标准文件路径
export PREDICT_DATA_DIR="${Root_Dir}/predict_data/"    # 预测结果路径
export RESOURCES_DIR="${Root_Dir}/resources/"          # 分词资源路径
export CODE_DIR="${Root_Dir}/label_lstm/"              # python代码路径

export SAVED_MODEL_PATH="${Root_Dir}/label_lstm/best_lstm_bert.pth" # 保存的分类模型路径

# 参数配置
down_table_name="ods_di_store"
up_table_name="ods_di_store_labeling"
#timestamp=$(date +%s)     # 时间戳
upload_batch_size=1000000 # 上传数据batch

# 运行脚本
${python} readCK.py \
  --download_table_name=${down_table_name} \
  --upload_table_name=${up_table_name} \
  --data_path=${DATA_DIR} \
  --predict_path=${PREDICT_DATA_DIR} \
  --upload_batch_size=${upload_batch_size} \
