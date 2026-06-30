# 该py脚本中 主要封装的是 原始数据路径(5个), 处理后的数据路径(6个), 模型保存路径(1, 父目录), 类别字典(1个)

class Config:
    def __init__(self):
        # todo 1.项目根目录
        # self.root_path = 'E:/develop/MyWorkSystem/itcast/cd_1/TMF_Project/'
        self.root_path = '../'

        # todo 2.原始数据路径
        self.train_datapath = self.root_path + 'data/clean/train.json'
        self.test_datapath = self.root_path + 'data/clean/test.json'
        self.eval_datapath = self.root_path + 'data/clean/eval.json'
        # 类别文档
        self.class_doc_path = self.root_path + "data/clean/class.txt"

        # todo 3.数据处理保存路径
        # 字符级别fasttext
        self.process_train_datapath_char = self.root_path + "fasttext/final_data/train_process_char.txt"
        self.process_test_datapath_char = self.root_path + "fasttext/final_data/test_process_char.txt"
        self.process_eval_datapath_char = self.root_path + "fasttext/final_data/eval_process_char.txt"

        # 词级别fasttext
        self.process_train_datapath_word = self.root_path + "fasttext/final_data/train_process_word.txt"
        self.process_test_datapath_word = self.root_path + "fasttext/final_data/test_process_word.txt"
        self.process_eval_datapath_word = self.root_path + "fasttext/final_data/eval_process_word.txt"

        # todo 4.模型路径
        self.ft_model_save_path = self.root_path + 'fasttext/save_models'

        # todo 5.处理完的数据（用于训练）
        self.final_data = self.root_path + 'fasttext/final_data'

        # todo 6.类别字典, 格式为: {0: 'business', 1: 'entertainment', 2: 'sports', 3: 'tech'...}
        self.id2class_dict = {i:line.strip() for i, line in enumerate(open(self.class_doc_path))}


# 测试代码
if __name__ == '__main__':
    config = Config()
    print(config.train_datapath)

    # {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science', 5: 'society', 6: 'politics', 7: 'sports',
    # 8: 'game', 9: 'entertainment'}
    print(config.id2class_dict)
