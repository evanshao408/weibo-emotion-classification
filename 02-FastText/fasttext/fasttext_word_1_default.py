'''# 该py文件用于: 字符级别 默认参数训练

# 导包
import fasttext
from config import Config
import datetime
import os

# todo 1. 加载项目配置
config = Config()

# todo 2. 模型训练: 使用 fasttext训练 字符级 文本分类模型
# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
model = fasttext.train_supervised(input=config.process_train_datapath_word)

# # todo 3. 打印模型训练后的关键信息.
# # get_word_vector方法, 获取字符'日'的向量表示(长度: 10维), 查看数字的特征, 例如:
# # [ 0.06251448  0.21214555 -0.5582269  -0.2776528  -0.40804806 -0.2712767, -0.03183828 -0.10957453  0.31537786  0.19355372]
# print(model.get_word_vector('日'))
# print(len(model.get_word_vector('日')))
#
# # labels属性, 获取模型训练到的 所有类别标签
# print(model.labels)
# # 获取模型训练到的 类别数量
# print(len(model.labels))
#
# # get_words方法, 获取模型训练到的 所有单词及对应的词频
# print(model.get_words(include_freq=True))
#
# # 用zip()函数, 配对: 字符和频率, 转为列表, 方便查看每个字符对应的出现次数.
# print(list(zip(*model.get_words(include_freq=True))))

# todo 4. 模型保存. save_model方法保存模型
model_path = config.ft_model_save_path + '/model_word_1_default.bin'
model.save_model(model_path)
print('模型保存成功！')

# todo 5. 模型预测. predict方法预测结果
print(model.predict('日 本 地 震 海 啸'))

# todo 6. words属性, 查看模型学习到的词汇表
print(model.words[:10])

# todo 7. get_subwords方法, 模型子词查看.
print(model.get_subwords('日本地震海啸!'))

# todo 8. get_dimension方法, 查看模型的向量维度.
print(model.get_dimension())

# todo 9. test方法, 模型评估.  (样本数, 精确率, 召回率)
print(model.test(config.process_test_datapath_word))   # (10000, 0.9074, 0.9074)
print(model.test(config.process_dev_datapath_word))    # (10000, 0.8999, 0.8999)'''


# 该py文件用于: 词语级别 默认参数训练
import fasttext
from config import Config
import os

# todo 1. 加载项目配置
config = Config()

# todo 2. 训练词级别模型
model = fasttext.train_supervised(input=config.process_train_datapath_word)

'''# todo 3. 打印模型训练后的关键信息.
print(model.get_word_vector('日'))
print(len(model.get_word_vector('日')))
# labels属性, 获取模型训练到的 所有类别标签
print(model.labels)
# 获取模型训练到的 类别数量
print(len(model.labels))
# get_words方法, 获取模型训练到的 所有单词及对应的词频
print(model.get_words(include_freq=True))
print(list(zip(*model.get_words(include_freq))))'''

# todo 4. 模型保存
os.makedirs(config.ft_model_save_path, exist_ok=True)
model_path = os.path.join(config.ft_model_save_path, 'model_word_1_default.bin')
model.save_model(model_path)
print('模型保存成功！')

# todo 5. 模型预测
print(model.predict('日 本 地 震 海 啸'))

# todo 6. 查看词汇表前10
print(model.words[:10])

# todo 7. 子词查看
print(model.get_subwords('日本地震海啸!'))

# todo 8. 向量维度
print(model.get_dimension())

# todo 9. 模型评估（训练/测试/验证集对应word路径）
print(model.test(config.process_test_datapath_word))  # (4990, 0.7244488977955912, 0.7244488977955912)
print(model.test(config.process_eval_datapath_word))  # (1997, 0.7210816224336505, 0.7210816224336505)