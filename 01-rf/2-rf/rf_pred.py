"""
案例: 演示 随机森林版的 模型预测. 批次预测, 比较适合离线业务(比如每天凌晨跑一下模型, 预测一下昨天的数据的结果)
"""

# 导包
import pandas as pd
import pickle
from config import Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# import warnings
# warnings.filterwarnings('ignore')

# todo 1. 加载配置文件.
# 设置pandas显示选项
pd.set_option('display.max_columns', None)
# 加载配置
conf = Config()

# todo 2. 加载模型和向量化器.
# 1. 加载随机森林模型(rf)
with open(conf.rf_model_save_path, 'rb') as f:
    rf = pickle.load(f)

# 2. 加载向量化器(tfidf)
with open(conf.tfidf_model_save_path, 'rb') as f:
    tfidf = pickle.load(f)

# 3. 加载标签编码器
with open(conf.label_encoder_save_path, 'rb') as f:
    label_encoder = pickle.load(f)

# todo 3. 读取 eval 数据集, 分隔符为: ,(默认的, 可以不处理, 因为是csv文件)
df_data = pd.read_csv(conf.process_eval_path)

# 3.1 删除 words 列为空的行
print(f'删除空值前: {len(df_data)}')
df_data = df_data.dropna(subset=['words']).reset_index(drop=True)
print(f'删除空值后: {len(df_data)}')

# 3.2 重新提取 words 列（关键！删除空值后必须重新提取）
words = df_data['words']

# todo 4. 对 eval 数据集进行向量化.
features = tfidf.transform(words)
# print(f'features.shape: {features.shape}')  # (10000, 34930)

# todo 5. 模型预测.
y_pred = rf.predict(features)
# 5.1 将数值预测标签转换为字符串标签
y_pred_str = label_encoder.inverse_transform(y_pred)

# 5.2 确保真实标签也是字符串类型
y_true_str = df_data['label'].astype(str)

# 计算准确率(accuracy): 正确预测的样本 占 总样本的比例
print(f'准确率: {accuracy_score(y_true_str, y_pred_str):.4f}')
# 计算精确率(precision): 预测为正类的样本中, 实际为正类的样本所占的比例
# macro: 宏平均 对所有类别平等加权, micro: 微观平均
print(f'精确率: {precision_score(y_true_str, y_pred_str, average="macro"):.4f}')
# 计算召回率(recall): 实际为正类的样本中, 预测为正类的样本所占的比例
print(f'召回率: {recall_score(y_true_str, y_pred_str, average="macro"):.4f}')
# 计算F1-score: 精确率和召回率的调和平均数
print(f'F1-score: {f1_score(y_true_str, y_pred_str, average="macro"):.4f}')

# todo 6.保存结果.
df_data['pred_label'] = y_pred
# print(f'df_data: {df_data}')

# 参1: 结果路径, 参2: 分隔符, 参3: 是否保存索引
df_data.to_csv(conf.model_predict_result, sep='\t', index=False)