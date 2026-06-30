
"""
EDA:
    探索性数据分析(Exploratory Data Analysis), 是一种分析数据集以总结其主要特征, 一般和图形表示法结合使用.
    大白话: 就是分析数据的, 可以帮我们发现模式, 检测异常, 测试假设, 从而对数据集有一个直观的了解.
"""


# 导包
import pandas as pd
import jieba
from config import Config

# 1. 初始化配置文件.
conf = Config()

# 2. 设置处理的及分析的文件路径, 默认为: train.txt
current_train_path = conf.train_datapath
current_test_path = conf.test_label_path
current_eval_path = conf.eval_label_path

# 3. 读取数据.
train_data = pd.read_json(current_train_path)
test_data = pd.read_json(current_test_path)
eval_data = pd.read_json(current_eval_path)
print(f'train_data: {train_data.head()}')
print(f'test_data: {test_data.head()}')
print(f'eval_data: {eval_data.head()}')


# # 4.进行分词预处理, 对输入文本进行分词, 并限制前30个词.
# def cut_sentence(txt):
#     # 对输入文本进行分词, 并限制前30个词.
#     return ' '.join(list(jieba.cut(txt))[:100])      # ['词1', '词2', '词3'...]

# 4.进行分词预处理, 对输入文本进行分词, 并限制前30个词.
def cut_sentence(txt):
    # 对输入文本进行分词, 并限制前30个词.
    return ' '.join(list(txt)[:200])      # ['词1', '词2', '词3'...]

# 5. 对每行文本进行分词并存储到words列.
train_data['words'] = train_data['content'].apply(cut_sentence)
test_data['words'] = test_data['content'].apply(cut_sentence)
eval_data['words'] = eval_data['content'].apply(cut_sentence)
print(f'train_data: {train_data.head()}')
print(f'test_data: {test_data.head()}')
print(f'eval_data: {eval_data.head()}')


# 6. 保存处理以后的训练数据.
if 'train' in  current_train_path :
    train_data.to_csv(conf.process_train_path, index=False)
    print(f'train数据保存成功, 存储路径为: {conf.process_train_path}!')

if 'test' in  current_test_path :
    test_data.to_csv(conf.process_test_path, index=False)
    print(f'test数据保存成功, 存储路径为: {conf.process_test_path}!')

if 'eval' in  current_eval_path :
    eval_data.to_csv(conf.process_eval_path, index=False)
    print(f'dev数据保存成功, 存储路径为: {conf.process_eval_path}!')


