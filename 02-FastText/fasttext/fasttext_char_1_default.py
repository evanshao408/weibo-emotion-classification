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
model = fasttext.train_supervised(input=config.process_train_datapath_char)

''''''# todo 3. 打印模型训练后的关键信息.
# get_word_vector方法, 获取字符'日'的向量表示(长度: 10维), 查看数字的特征, 例如:
# [ 0.06251448  0.21214555 -0.5582269  -0.2776528  -0.40804806 -0.2712767, -0.03183828 -0.10957453  0.31537786  0.19355372]
print(model.get_word_vector('日'))
print(len(model.get_word_vector('日')))

# labels属性, 获取模型训练到的 所有类别标签
print(model.labels)
# 获取模型训练到的 类别数量
print(len(model.labels))

# get_words方法, 获取模型训练到的 所有单词及对应的词频
print(model.get_words(include_freq=True))

# 用zip()函数, 配对: 字符和频率, 转为列表, 方便查看每个字符对应的出现次数.
print(list(zip(*model.get_words(include_freq=True))))''''''

# todo 4. 模型保存. save_model方法保存模型
model_path = config.ft_model_save_path + '/model_char_1_default.bin'
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
print(model.test(config.process_test_datapath_char))   # (10000, 0.8756, 0.8756)
print(model.test(config.process_dev_datapath_char))    # (10000, 0.8719, 0.8719)'''


# 该py文件用于: 字符级别 默认参数训练
# 适配 data_preprocess.py 处理后的字符级数据格式

# 导包
import fasttext
from config import Config
import datetime
import os

# todo 1. 加载项目配置
config = Config()

# todo 2. 模型训练: 使用 fasttext训练 字符级 文本分类模型
# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
# 输入路径对应预处理后的字符级训练集
model = fasttext.train_supervised(
    input=config.process_train_datapath_char,
    # 可根据需要补充默认参数（fasttext默认参数也可显式声明）
    lr=0.1,          # 学习率
    dim=100,         # 向量维度（默认100，可通过get_dimension验证）
    epoch=5,         # 训练轮数
    wordNgrams=1,    # 单字符（字符级无需n-gram）
    verbose=2,       # 训练过程输出详细程度
    minCount=1       # 最小字符出现次数
)

# todo 3. 打印模型训练后的关键信息（可选开启）
'''
# get_word_vector方法, 获取字符'日'的向量表示(长度默认100维)
print("字符'日'的向量表示：", model.get_word_vector('日'))
print("向量维度：", len(model.get_word_vector('日')))

# labels属性, 获取模型训练到的 所有类别标签（__label__+数字形式）
print("模型类别标签：", model.labels)
# 获取模型训练到的 类别数量
print("类别数量：", len(model.labels))

# get_words方法, 获取模型训练到的 所有字符及对应的词频
print("字符及词频（前10个）：", model.get_words(include_freq=True)[:10])

# 用zip()函数, 配对: 字符和频率, 转为列表, 方便查看每个字符对应的出现次数
words, freqs = model.get_words(include_freq=True)
print("字符-频率配对（前10个）：", list(zip(words[:10], freqs[:10])))
'''

# todo 4. 模型保存. save_model方法保存模型
# 确保模型保存目录存在
os.makedirs(config.ft_model_save_path, exist_ok=True)
model_path = os.path.join(config.ft_model_save_path, 'model_char_1_default.bin')
model.save_model(model_path)
print(f'模型保存成功！路径：{model_path}')

# todo 5. 模型预测. predict方法预测结果
# 测试字符级预测（与预处理的字符切分格式一致）
test_text = "日 本 地 震 海 啸"
predict_result = model.predict(test_text)
print(f"测试文本：{test_text}")
print(f"预测结果：{predict_result}")  # 格式：(('__label__数字',), [概率值])

# todo 6. words属性, 查看模型学习到的词汇表（前10个字符）
print("模型学习到的字符（前10个）：", model.words[:10])

# todo 7. get_subwords方法, 模型子词查看（字符级子词特征）
test_subword_text = "日本地震海啸!"
subwords = model.get_subwords(test_subword_text)
print(f"文本'{test_subword_text}'的子词：{subwords}")

# todo 8. get_dimension方法, 查看模型的向量维度（默认100）
print("模型向量维度：", model.get_dimension())

# todo 9. test方法, 模型评估.  (样本数, 精确率, 召回率)
# 评估测试集（字符级）
test_metrics = model.test(config.process_test_datapath_char)
print(f"测试集评估结果（样本数, 精确率, 召回率）：{test_metrics}")   # (4990, 0.6935871743486974, 0.6935871743486974)

# 评估验证/开发集（对应预处理的eval数据集）
dev_metrics = model.test(config.process_eval_datapath_char)
print(f"验证集评估结果（样本数, 精确率, 召回率）：{dev_metrics}")   # (1997, 0.6975463194792189, 0.6975463194792189)