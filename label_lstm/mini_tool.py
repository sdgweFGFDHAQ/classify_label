import re
import jieba
from torch.utils.data import Dataset


# 分词工具类
class WordSegment:
    def __init__(self):
        self.set_jieba()
        self.stop_words = set()
        self.load_stop_words('../resources/stopwords.txt')

        # 设置不可分割的词

    def set_jieba(self):
        jieba.load_userdict("../resources/statename.txt")
        jieba.load_userdict("../resources/cityname.txt")
        jieba.load_userdict("../resources/distinctname.txt")
        jieba.load_userdict("../resources/symbol.txt")
        jieba.load_userdict("../resources/namenoise.txt")
        # 自定义词集
        jieba.load_userdict('../resources/indiv_words.txt')

    # 外部加载停用词集 file_path=../resources/stopwords.txt
    def load_stop_words(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.stop_words = set(line.strip() for line in file.readlines())

    # 清洗特殊字符
    def clean_text(self, text):
        # 清洗特殊字符
        text = re.sub(r'\(.*?\)|[^a-zA-Z0-9\u4e00-\u9fa5]|(丨)', ' ', str(text))
        # 形如:"EXO店x铺excelAxB" 去除x
        text = re.sub(r'(?<=[\u4e00-\u9fa5])([xX])(?=[\u4e00-\u9fa5])|(?<=[A-Z])x(?=[A-Z])', ' ', text)
        return text

    # 分词
    def cut_word(self, text):
        text = self.clean_text(text)
        # jieba分词
        l_cut_words = jieba.lcut(text)
        # 去除停用词（地名等无用的词）
        out_word_list = [lc_word for lc_word in l_cut_words if
                         lc_word not in self.stop_words and lc_word != '\t' and not lc_word.isspace()]
        # 如果文本去除后，长度变为0，则回滚去除操作
        if out_word_list and (len(out_word_list) != 0):
            return ' '.join(out_word_list)
        else:
            return ' '.join(l_cut_words)


def error_callback(error):
    print(f"Error info: {error}")


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
