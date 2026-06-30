# config.py
class Config(object):
    def __init__(self):
        # 原始数据路径
        self.test_datapath = "./train.txt"
        self.class_datapath = "./class.txt"


if __name__ == '__main__':
    conf = Config()
    print(conf.test_datapath)