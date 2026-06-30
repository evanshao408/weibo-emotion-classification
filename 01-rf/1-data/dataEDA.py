"""
EDA:
    探索性数据分析(Exploratory Data Analysis), 是一种分析数据集以总结其主要特征, 一般和图形表示法结合使用.
    大白话: 就是分析数据的, 可以帮我们发现模式, 检测异常, 测试假设, 从而对数据集有一个直观的了解.
"""

# 导包
import pandas as pd  # 用于处理结构化表格数据的
from collections import Counter  # 用于高效统计标签出现的次数
from config import Config  # 获取项目的配置信息
import matplotlib.pyplot as plt  # 用于绘图
import json

# todo 1. 初始化配置与数据路径.
# 1. 创建Config类的实例, 加载项目配置.
conf = Config()
# 2. 从配置中获取训练数据集的路径.
file_path = conf.train_datapath

# todo 2. 读取数据, 并查看基本信息.
# 1. 用pandas读取文件数据
data = pd.read_json(file_path)
# 2. 打印前5行数据.
print(f'前5行数据为: \n{data.head(5)}')
# 3. 打印数据的总行数, 以便了解数据的规模.
print(f'总数据量: {len(data)}, {data.shape[0]}')
print('-' * 80)

# todo 3. 统计标签分布.
# 1. Counter类统计数据中每个标签(label列)的数量.
label_counts = Counter(data['label'])
print(f'标签分布为: \n{label_counts}')
# 2. 循环遍历统计结果, 打印每个标签及其出现的次数.
for label, count in label_counts.items():
    print(f'标签 {label}, 出现了 {count} 次')
#
# # todo 4. 分析文本长度
# # 1. 在数据中新增'text_length'列, 存储每条文本的字符数.
# data['text_length']  = data['text'].str.len()
# # 2. 打印提示信息.
# print(f'\n文本长度前10行: ')
# # 3. 显示文本内容和对应的长度(仅显示'text' 和 'text_length'两列的前10行)
# print(data[['text','text_length']].head(10))
# # 4. 打印文本长度的平均值(保留2位小数)
# print(f'平均长度: {data["text_length"].mean():.2f}')
# # 5. 打印文本长度的标准差(反应长度的离散程度, 保留2位小数)
# print(f'标准差:{data["text_length"].std():.2f}')
# # 6. 打印文本长度的最大值
# print(f'最大值：{data["text_length"].max()}')
# # 7打印文本长度的最小值.
# print(f'最小值：{data["text_length"].min()}')
#
# # todo 5. 绘制text_length与label的直方图.
# plt.hist(data['text_length'], bins=100)
# plt.show()
# plt.hist(data['label'], bins=100)
# plt.show()

