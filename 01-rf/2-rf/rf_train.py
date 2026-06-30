# 导包

import pandas as pd     # 数据读取和处理
import pickle           # 用于模型和向量化器的序列化保存
from sklearn.feature_extraction.text import TfidfVectorizer # 将文本转成数值特征(可以理解为: 词向量)
from sklearn.model_selection import train_test_split        # 训练集和测试集的划分
from sklearn.ensemble import RandomForestClassifier         # 随机森林分类器
# 模型评估指标: 准确率, 精准率, 召回率, F1-score, 分类报告, 混淆矩阵
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from config import Config        # 配置文件
from tqdm import tqdm               # 进度条
# 导入三种集成学习分类器: 随机森林, AdaBoost, GBDT
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# 导入标签编码器
from sklearn.preprocessing import LabelEncoder

# todo 1. Pandas的基础设置.
pd.set_option('display.expand_frame_repr', False)  # 避免宽表格换行
pd.set_option('display.max_columns', None)         # 确保所有列可见
# 实例化配置对象, 用于获取: 文件路径等配置.
conf = Config()

# todo 2.读取训练数据.
# 2.1 读取训练集、测试集和验证集
train_df = pd.read_csv(conf.process_train_path)
test_df = pd.read_csv(conf.process_test_path)      # 需要配置测试集路径
eval_df = pd.read_csv(conf.process_eval_path)        # 需要配置验证集路径

# 2.2 删除各数据集的空值
print(f'训练集删除空值前: {len(train_df)}, 删除后: {len(train_df.dropna(subset=["content"]))}')
train_df = train_df.dropna(subset=['content']).reset_index(drop=True)

print(f'测试集删除空值前: {len(test_df)}, 删除后: {len(test_df.dropna(subset=["content"]))}')
test_df = test_df.dropna(subset=['content']).reset_index(drop=True)

print(f'验证集删除空值前: {len(eval_df)}, 删除后: {len(eval_df.dropna(subset=["content"]))}')
eval_df = eval_df.dropna(subset=['content']).reset_index(drop=True)

# 2.3 提取文本和标签
train_words = train_df['words']
train_labels = train_df['label']

test_words = test_df['words']
test_labels = test_df['label']

dev_words = eval_df['words']
dev_labels = eval_df['label']

# 2.4 将字符串标签转换为数值类型
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)
dev_labels = label_encoder.transform(dev_labels)

# print(f'\n标签映射关系: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}')
# print(f'训练集标签分布: {pd.Series(train_labels).value_counts().to_dict()}')
# print(f'\n训练集示例:\n{train_df.head()}')
# print(f'测试集标签分布: {pd.Series(test_labels).value_counts().to_dict()}')
# print(f'\n测试集示例:\n{test_df.head()}')


# todo 3. 文本特征提取(TF-IDF向量化)
# 3.1 读取停用词文件(停用词: 对分类无意义的词, 如: 的, 是等这些词), 按行分割为列表.
# stop_words = open(conf.stop_words_path, encoding='utf-8').read().split()
# print(f'stop_words: {stop_words[:5]}', stop_words[-5:])

# 3.2 初始化 TF-IDF向量化器, 指定停用词列表(过滤停用词的)
tfidf = TfidfVectorizer(
    token_pattern=r"(?u)\S+",  # 匹配单个字符（关键参数！）
    lowercase=False,            # 不转换小写
    max_features=8000,          # 限制最大特征数
    min_df=2,                   # 最少出现在2个文档中
    max_df=0.95,                # 最多出现在95%的文档中
    ngram_range=(1, 3),         # 使用一元和二元字符组合
    sublinear_tf=True           # 使用对数TF缩放
)

# 3.3 对文本特征进行拟合(学习词汇表) 并转换 TF-IDF特征矩阵
# 仅在训练集上 fit_transform，测试集和验证集只用 transform
train_features = tfidf.fit_transform(train_words)
test_features = tfidf.transform(test_words)
dev_features = tfidf.transform(dev_words)
print(f'训练集特征矩阵: {train_features.shape}')
print(f'测试集特征矩阵: {test_features.shape}')
print(f'验证集特征矩阵: {dev_features.shape}')

# # 3.4 查看生成的词汇表(特征名称)列表
# print(list(tfidf.get_feature_names_out()))      # [...'龚方雄', '龚炜任', '龚蓓'...], 大小是: 34930
# # 3.5 查看词汇表中 词和索引的映射关系(字典形式: 词 -> 列索引)
# print(tfidf.vocabulary_)        # {'中华': 3987, '女子': 12910, '学院': 13359,...}
# # 3.6 再次确认词汇表的大小.
# print(len(tfidf.vocabulary_))   # 34930


# todo 4. 模型的训练和评估
# 4.1 划分训练集和测试集.这个有独自的训练和测试集不用划分
# 4.2 初始化随机森林分类器
model = RandomForestClassifier(
    n_estimators=300,           # 树的数量
    # max_depth=25,               # 限制树的深度，防止过拟合
    # min_samples_split=5,        # 节点分裂所需最小样本数
    # min_samples_leaf=2,         # 叶子节点最少样本数
    class_weight='balanced',    # 自动处理类别不平衡（重要！）
    # max_features='sqrt',        # 每次分裂考虑的特征数
    n_jobs=-1,                  # 使用所有CPU核心
    random_state=42             # 固定随机种子，保证可复现
)

# 4.3 训练模型, 并使用tqdm 显示进度条
for _ in tqdm(range(1)):
    model.fit(train_features, train_labels)

# 4.4 模型预测和评估.
pred_labels = model.predict(test_features)
# 4.5 打印预测结果
print(f'预测结果: {pred_labels}')
print(f'准确率: {accuracy_score(test_labels, pred_labels):.4f}')
# 打印微平均精确率(所有类别合集计算的精确率, 适用于 多类别不平衡的场景)
# 微平均和宏平均: 宏平均: 所有类别的精确率求平均, 微平均: 所有类别的精确率求和, 然后除以类别数量.
print(f'精准率(micro): {precision_score(test_labels, pred_labels, average="micro"):.4f}')
print(f'召回率(micro): {recall_score(test_labels, pred_labels, average="micro"):.4f}')
print(f'F1-score(micro): {f1_score(test_labels, pred_labels, average="micro"):.4f}')


# todo 5. 模型和向量化器保存.
# 5.1 保存训练好的 随机森林模型.
with open(conf.rf_model_save_path, 'wb') as f:
    pickle.dump(model, f)

# 5.2 保存训练好的 TF-IDF向量化器(后续预测时, 需要使用同一个向量化器转换新文本)
with open(conf.tfidf_model_save_path, 'wb') as f:
    pickle.dump(tfidf, f)

# 5.3 保存标签编码器（后续预测时需要将数值转回字符串）
with open(conf.label_encoder_save_path, 'wb') as f:
    pickle.dump(label_encoder, f)

# 5.4 提示保存成功.
print(f'模型、向量化器和标签编码器保存成功!')

