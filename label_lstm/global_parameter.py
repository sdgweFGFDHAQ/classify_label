class StaticParameter:
    # 本地调试参数
    # 设置最频繁使用的词个数(在texts_to_matrix取前MAX_NB_WORDS列)
    MAX_WORDS_NUM = 120000
    # 每条cut_name最大的长度
    MAX_LENGTH = 7
    # 设置Embedding层的维度
    EMBEDDING_DIM = 120
    # 全国数据分段跑
    SEGMENT_NUMBER = 8
    # 抽样数量
    DATA_NUMBER = 50000
    # 标准广州数据
    PATH_TRAIN = 'standard_store_'
    # Linux服务器参数
    PATH_ZZX = '/home/data/temp/zhouzx'
    PATH_ZZX_DATA = '/home/DI/zhouzx/code/classify_label/data/'
    PATH_ZZX_STANDARD_DATA = '/home/DI/zhouzx/code/classify_label/standard_data/'
    PATH_ZZX_PREDICT_DATA = '/home/DI/zhouzx/code/classify_label/predict_data/'
