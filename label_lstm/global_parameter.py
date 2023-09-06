class StaticParameter:
    # 本地调试参数
    # 设置最频繁使用的词个数(在texts_to_matrix取前MAX_NB_WORDS列)
    MAX_WORDS_NUM = 120000

    # 全国数据分段跑
    SEGMENT_NUMBER = 11
    # 不能被划分到现有标签体系的数据类别
    UNKNOWN_CATEGORY = ''
    # 预训练bert模型
    PRETRAIN_BERT_URL = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"
    # 加载训练好的模型
    BEST_BERT_MODEL = 'best_lstm_bert_6.model'  # sota,92.4%

    # 每条cut_name最大的长度
    MAX_LENGTH = 7
    # 设置Embedding层的维度
    EMBEDDING_DIM = 120
    # batch_size大小
    BATCH_SIZE = 512

    # Linux服务器参数
    PATH_ZZX = '/home/data/temp/zhouzx'
    PATH_ZZX_DATA = '/home/DI/zhouzx/code/classify_label/data/'
    PATH_ZZX_STANDARD_DATA = '/home/DI/zhouzx/code/classify_label/standard_data/'
    PATH_ZZX_PREDICT_DATA = '/home/DI/zhouzx/code/classify_label/predict_data/'
