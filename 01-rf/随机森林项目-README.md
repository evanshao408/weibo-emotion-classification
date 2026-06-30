# SMP2020-EWECT 微博情绪分类 - 随机森林实现

本项目实现了基于随机森林（Random Forest）的微博情绪分类模型，采用 TF-IDF 文本向量化技术，结合 Flask API 和 Streamlit 前端界面，提供完整的情绪分类解决方案。

## 项目背景

本项目基于 **SMP2020-EWECT 微博情绪分类技术评测**（[官网](https://smp2020ewect.github.io/)），该评测由中国中文信息学会社会媒体处理专委会主办，哈尔滨工业大学承办。

### 评测任务

微博情绪分类任务旨在识别微博中蕴含的情绪，输入是一条微博，输出是该微博所蕴含的情绪类别。本次评测将微博情绪分为以下六个类别：

| 类别 | 标签 | 说明 |
|------|------|------|
| 积极 | happy | 正面情绪 |
| 愤怒 | angry | 愤怒情绪 |
| 悲伤 | sad | 悲伤情绪 |
| 恐惧 | fear | 恐惧情绪 |
| 惊奇 | surprise | 惊奇情绪 |
| 无情绪 | neutral | 中性/无明显情绪 |

### 评价指标

评测采用**宏平均F1值**作为评估指标：

```
Macro_F = (Macro_F通用 + Macro_F疫情) / 2
```

## 项目结构

```
├── 1-data/                        # 数据处理模块
│   ├── config.py                  # 数据配置文件（数据集路径）
│   ├── dataEDA.py                 # 探索性数据分析（EDA）
│   └── data/                      # 数据集目录
│       ├── stopwords.txt          # 停用词表
│       ├── usual_class.txt        # 类别定义文件
│       ├── usual_train.json       # 通用微博训练集
│       ├── usual_eval_labeled.json # 通用微博验证集
│       └── usual_test_labeled.json # 通用微博测试集
├── 2-rf/                          # 随机森林模型模块
│   ├── config.py                  # 模型配置文件（路径、超参数）
│   ├── rf_train.py                # 随机森林训练脚本
│   ├── rf_pred.py                 # 批次预测脚本（离线预测）
│   ├── rf_pred_fun.py             # 实时预测函数（API对接）
│   ├── api_flask_server.py        # Flask API服务端
│   ├── api_local_client.py        # API本地客户端测试
│   ├── app_streamlit.py           # Streamlit前端应用
│   ├── app_streamlit_run.py       # Streamlit运行脚本
│   ├── data/                      # 处理后的数据集
│   │   ├── process_usual_train.csv
│   │   ├── process_usual_test.csv
│   │   └── process_usual_eval.csv
│   ├── save_model/                # 保存的模型文件
│   │   ├── rf_model.pkl           # 随机森林模型
│   │   ├── tfidf_model.pkl        # TF-IDF向量化器
│   │   └── label_encoder.pkl      # 标签编码器
│   └── result/                    # 预测结果
│       └── predict_result.txt
└── bert项目-README.md             # BERT知识蒸馏版本说明
```

## 技术架构

### 数据处理流程

```
原始JSON数据 → 分词处理 → TF-IDF向量化 → 特征矩阵
```

**关键技术点**：
- 使用 jieba 进行中文分词
- TF-IDF 向量化提取文本特征
- 最大特征数：8000
- N-gram 范围：(1, 3)
- 处理类别不平衡（class_weight='balanced'）

### 模型架构

```
输入文本 → TF-IDF向量化(8000维) → 随机森林分类器 → 预测标签
```

**关键技术点**：
- 随机森林分类器（300棵决策树）
- 使用所有CPU核心加速训练（n_jobs=-1）
- 类别权重自动平衡处理类别不平衡
- 固定随机种子保证可复现

### 部署架构

```
用户输入 → Streamlit前端 → Flask API → 预测函数 → 返回结果
```

**关键技术点**：
- Flask 提供 RESTful API 服务
- Streamlit 构建交互式前端界面
- 支持实时预测和批次预测两种模式

## 快速开始

### 环境要求

- Python ≥ 3.8
- scikit-learn ≥ 1.0
- pandas
- jieba
- Flask
- Streamlit
- tqdm

### 安装依赖

```bash
pip install scikit-learn pandas jieba flask streamlit tqdm
```

### 数据准备

#### 数据集说明

JSON格式如下：

```json
[
  {"id":1,"content": "微博文本内容", "label": "happy"},
  {"id":2,"content": "微博文本内容", "label": "angry"}
]
```

#### 数据集现状

本项目当前只包含**通用微博数据集**，不包含疫情微博数据集。如需完整评测，需自行添加疫情数据集并修改配置文件。

#### 路径配置

默认配置文件中的路径使用相对路径，需根据实际部署环境修改。修改方法：

```python
# 修改 1-data/config.py 中的路径配置
self.train_datapath = "./data/usual_train.txt"
self.test_label_path = "./data/usual_test_label.txt"
self.dev_label_path = "./data/usual_eval_labeled.txt"
self.class_datapath = "./data/usual_class.txt"

# 修改 2-rf/config.py 中的路径配置
self.train_datapath = "../1-data/data/usual_train.json"
self.test_label_path = "../1-data/data/usual_test_labeled.json"
self.eval_label_path = "../1-data/data/usual_eval_labeled.json"
```

### 训练流程

#### 1. 探索性数据分析（EDA）

```bash
cd 1-data
python dataEDA.py
```

#### 2. 训练随机森林模型

```bash
cd 2-rf
python rf_train.py
```

训练完成后，模型保存在 `2-rf/save_model/` 目录下：
- `rf_model.pkl`：随机森林模型
- `tfidf_model.pkl`：TF-IDF向量化器
- `label_encoder.pkl`：标签编码器

#### 3. 启动Flask API服务

```bash
cd 2-rf
python api_flask_server.py
```

服务默认运行在 `http://192.168.12.21:10088`，可通过修改 `api_flask_server.py` 中的 `host` 和 `port` 参数调整。

#### 4. 启动Streamlit前端

```bash
cd 2-rf
streamlit run app_streamlit.py
```

### 配置说明

主要配置参数在 `2-rf/config.py` 中：

| 参数 | 值 | 说明 |
|------|-----|------|
| n_estimators | 300 | 决策树数量 |
| max_features | 8000 | TF-IDF最大特征数 |
| ngram_range | (1, 3) | N-gram范围 |
| class_weight | balanced | 类别权重平衡 |
| n_jobs | -1 | 使用所有CPU核心 |
| random_state | 42 | 随机种子 |

## 核心文件说明

### 1-data/config.py

数据配置文件，集中管理所有数据文件路径：
- 训练集、测试集、验证集路径
- 类别定义文件路径
- 停用词文件路径

### 1-data/dataEDA.py

探索性数据分析脚本：
- 读取训练数据集
- 统计标签分布
- 分析文本长度特征
- 数据可视化（注释中）

### 2-rf/config.py

模型配置文件，集中管理模型相关配置：
- 原始数据路径和处理后数据路径
- 模型保存路径
- 预测结果保存路径

### 2-rf/rf_train.py

随机森林训练脚本：
- 读取处理后的CSV数据
- TF-IDF向量化（fit_transform）
- 随机森林模型训练
- 模型评估（准确率、精确率、召回率、F1-score）
- 模型保存（pickle序列化）

### 2-rf/rf_pred.py

批次预测脚本：
- 加载训练好的模型和向量化器
- 读取验证集数据
- 批量预测并计算评估指标
- 保存预测结果到文件

### 2-rf/rf_pred_fun.py

实时预测函数：
- 加载模型和向量化器
- 接收单条文本输入
- 分词处理和TF-IDF转换
- 返回预测类别结果

**使用示例**：
```python
from rf_pred_fun import predict_fun

result = predict_fun({'context': '这弄得什么东西真没啥用'})
print(result)
# {'context': '这弄得什么东西真没啥用', 'pred_class': 'angry'}
```

### 2-rf/api_flask_server.py

Flask API服务端：
- 创建Flask应用
- 定义 `/WSent_predict` POST接口
- 接收JSON数据，调用预测函数
- 返回JSON格式预测结果

**API调用示例**：
```bash
curl -X POST http://192.168.12.21:10088/WSent_predict \
  -H "Content-Type: application/json" \
  -d '{"context": "今天天气很好"}'
```

### 2-rf/app_streamlit.py

Streamlit前端应用：
- 创建交互式网页界面
- 用户输入文本框
- 调用Flask API进行预测
- 显示预测结果和耗时

## 预测模式

### 实时预测

适用于在线业务场景，单条数据即时返回结果：
- 使用 `rf_pred_fun.py` 中的 `predict_fun` 函数
- 通过 Flask API 提供服务
- Streamlit 前端交互

### 批次预测

适用于离线业务场景，批量处理数据：
- 使用 `rf_pred.py` 脚本
- 读取CSV文件进行批量预测
- 保存预测结果到文件

## 实验结果

模型训练完成后，会在测试集上进行评估，记录以下指标：
- 准确率（Accuracy）
- 宏平均精确率（Macro-Precision）
- 宏平均召回率（Macro-Recall）
- 宏平均F1-score（Macro-F1）

**训练日志示例**：
```
原始样本总量: 2000
清洗空值后样本量: 1997
准确率: 0.6715
精确率: 0.6758
召回率: 0.6305
F1-score: 0.6295
```

## License

本项目仅供学习和研究使用。