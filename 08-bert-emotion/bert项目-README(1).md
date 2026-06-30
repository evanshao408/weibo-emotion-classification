# SMP2020-EWECT 微博情绪分类 - BERT知识蒸馏实现

本项目实现了基于 BERT 的微博情绪分类模型，并通过知识蒸馏技术将 BERT 教师模型的知识迁移到 BiLSTM 学生模型中，以实现模型压缩和推理加速。

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
├── config_tuned.py              # 配置文件（模型参数、路径、超参数）
├── h1_dataloader_utils_tuned.py # 数据加载工具（JSON解析、数据集构建、Tokenizer编码）
├── h2_bert_classifier_model_tuned.py # BERT分类模型（教师模型）
├── h3_bilstm_classifier_model_tuned.py # BiLSTM分类模型（学生模型）
├── h3_train_predict_tuned.py    # BERT模型训练脚本
├── h4_bert_predict_fun.py       # BERT预测函数（API/APP对接）
├── h4_hard_label_distillation_tuned.py # 硬标签蒸馏
├── h5_soft_label_distillation_tuned.py # 软标签蒸馏（KL散度）
├── h6_soft_hybrid_label_distillation_tuned.py # 软硬标签混合蒸馏（动态权重）
├── model2dev_utils_tuned.py     # 模型评估工具（分类报告、F1、准确率等）
├── bert-base-chinese/           # BERT预训练模型目录
├── data/                        # 数据集目录
│   ├── usual_train.json         # 通用微博训练集
│   ├── usual_eval_labeled.json  # 通用微博验证集
│   └── usual_test_labeled.json  # 通用微博测试集
└── save_models/                 # 保存的模型文件
    ├── bert_emotion_classifier_model_tuned_v2.pt      # BERT教师模型
    └── bert_classifier_bilstm_model_soft.pt           # 蒸馏后的BiLSTM模型
```

## 技术架构

### 教师模型（BERT）

```
输入文本 → BERT Encoder → Masked Mean Pooling → LayerNorm → Dropout → FC(768→256) → GELU → Dropout → FC(256→6) → Logits
```

**关键技术点**：
- 使用 `bert-base-chinese` 预训练模型
- 采用 Masked Mean Pooling 获取句向量
- 添加类别权重和标签平滑（Label Smoothing）
- 使用 R-Drop 正则化增强模型鲁棒性

### 学生模型（BiLSTM）

```
输入文本 → Embedding(768→128) → BiLSTM(128→256×2) → Dropout → FC(512→6) → Logits
```

**关键技术点**：
- 3层双向LSTM
- 嵌入维度：128
- 隐藏层维度：256（双向输出为512）
- Dropout：0.3

### 知识蒸馏策略

#### 硬标签蒸馏（h4）

学生模型学习教师模型的预测标签（argmax结果）：

```
Loss = CrossEntropy(student_logits, teacher_labels)
```

#### 软标签蒸馏（h5）

学生模型学习教师模型的概率分布（带温度T的softmax）：

```
soft_loss = KL_Div(softmax(student_logits/T), softmax(teacher_logits/T)) × T²
hard_loss = CrossEntropy(student_logits, teacher_labels)
Loss = α × soft_loss + (1-α) × hard_loss
```

**参数**：T=2.0，α=0.5

#### 混合蒸馏（h6）

结合真实标签、教师硬标签和教师软标签，动态调整权重：

```
progress = epoch / total_epochs
T = 2.0 × (1 - 0.5 × progress)      # 温度从2.0降到1.0
α = 0.5 × (1 - 0.6 × progress)     # 软标签权重从0.5降到0.2
β = 1.0 - α                         # 硬标签权重从0.5升到0.8

Loss = α × soft_loss + β × 0.6 × teacher_hard_loss + β × 0.4 × real_label_loss
```

## 快速开始

### 环境要求

- Python ≥ 3.8
- PyTorch ≥ 1.10
- Transformers ≥ 4.0
- scikit-learn ≥ 1.0
- tqdm

### 安装依赖

```bash
pip install torch torchvision transformers scikit-learn tqdm
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

默认配置文件 `config_tuned.py` 中的路径使用相对路径（如 `../08-bert-emotion/data/`），需根据实际部署环境修改。修改方法：

```python
# 修改 config_tuned.py 中的路径配置
self.root_path = './'  # 改为当前目录
self.train_datapath = self.root_path + 'data/usual_train.json'
self.test_datapath = self.root_path + 'data/usual_test_labeled.json'
self.dev_datapath = self.root_path + 'data/usual_eval_labeled.json'
self.bert_path = self.root_path + 'bert-base-chinese'
```

### 训练流程

#### 1. 训练BERT教师模型

```bash
python h3_train_predict_tuned.py
```

训练完成后，模型保存在 `save_models/bert_emotion_classifier_model_tuned_v2.pt`。

#### 2. 硬标签蒸馏

```bash
python h4_hard_label_distillation_tuned.py
```

#### 3. 软标签蒸馏

```bash
python h5_soft_label_distillation_tuned.py
```

#### 4. 混合蒸馏

```bash
python h6_soft_hybrid_label_distillation_tuned.py
```

### 配置说明

主要配置参数在 `config_tuned.py` 中：

| 参数 | 值 | 说明 |
|------|-----|------|
| num_epochs | 5 | BERT训练轮数 |
| batch_size | 64 | 批次大小 |
| pad_size | 128 | 序列最大长度 |
| bert_learning_rate | 1e-5 | BERT学习率 |
| head_learning_rate | 5e-5 | 分类头学习率 |
| label_smoothing | 0.05 | 标签平滑系数 |
| rdrop_alpha | 1.0 | R-Drop正则化系数 |
| student_epochs | 10 | 学生模型训练轮数 |
| lstm_learning_rate | 1e-3 | BiLSTM学习率 |
| distill_temperature | 2.0 | 蒸馏温度T |
| distill_alpha | 0.5 | 软标签权重α |

## 核心文件说明

### config_tuned.py

集中管理所有配置参数，包括：
- 模型路径配置
- 训练超参数
- 蒸馏参数
- 类别标签映射

### h1_dataloader_utils_tuned.py

数据加载工具，负责：
- JSON数据解析
- Dataset构建
- Tokenizer编码（BERT分词）
- DataLoader生成

### h2_bert_classifier_model_tuned.py

BERT分类模型定义：
- 加载BERT：手动加载权重 + 自定义预处理
- 池化策略：`masked_mean_pooling`（掩码均值池化）
- 分类器结构：双层全连接 + LayerNorm + Dropout + GELU 激活
- 正则化：Dropout

### h3_bilstm_classifier_model_tuned.py

BiLSTM分类模型定义：
- Embedding层
- 3层双向LSTM
- 全连接分类层

### h3_train_predict_tuned.py

BERT训练脚本：
- 类别权重计算（处理类别不平衡）
- 学习率调度（Warmup + Linear Decay）
- R-Drop正则化
- 模型保存（基于验证集F1）

### h4_bert_predict_fun.py

BERT预测函数，用于API和APP对接：
- 加载已训练的BERT模型
- 接收文本输入，返回预测结果
- 返回完整的预测信息（类别、置信度、各类别概率）

**使用示例**：
```python
from h4_bert_predict_fun import predict_fun

result = predict_fun({'text': '今天天气很好'})
print(result)
# {'text': '今天天气很好', 'pred_class': 'happy', 'pred_index': 2, 
#  'confidence': 0.8523, 'probabilities': {...}, 'cost_time': 12.3}
```

### model2dev_utils_tuned.py

模型评估工具：
- 在验证/测试集上评估模型
- 生成分类报告
- 计算Macro-F1、准确率、精确率、召回率

## 知识蒸馏原理

### 硬标签蒸馏

硬标签蒸馏直接使用教师模型的预测结果（one-hot标签）作为学生模型的训练目标，相当于让学生模型模仿教师模型的最终决策。

### 软标签蒸馏

软标签蒸馏利用教师模型输出的概率分布（softmax结果），这些"软标签"包含了教师模型对不同类别的置信度信息。通过温度参数T，可以调整软标签的平滑程度：

- T较大：分布更平滑，保留更多类别间的关联信息
- T较小：分布更尖锐，接近硬标签

### 混合蒸馏

混合蒸馏结合了真实标签、教师硬标签和教师软标签，通过动态调整权重平衡三者的影响：

- 训练初期：更依赖软标签（α较大），让学生快速学习教师的知识
- 训练后期：更依赖硬标签和真实标签（β较大），保证分类准确性

## 实验结果

模型训练过程中，每100个批次会在验证集上进行评估，记录Macro-F1、准确率、精确率和召回率。最佳模型会根据验证集Macro-F1自动保存。

### 训练日志示例（演示数据）

#### 原模型训练结果

```
全局最高 Macro-F1: 0.7873（本组所有快照峰值）
对应位置：第 3 轮，批次 400/434
配套指标：
   准确度 Accuracy: 0.8092
   Macro-Precision: 0.7774
   Macro-Recall: 0.8014
   验证损失：0.7064
本组超参：
   类别权重：[0.7453112006187439, 1.9479403495788574, 0.9270195960998535, 0.8969992399215698, 0.9625783562660217, 1.4899063110351562]
   总步数：2170，warmup=130，训练 5 epoch
```



#### 硬标签蒸馏结果

```
Macro-F1: 0.6443, 准确度：0.6755, Macro-Precision: 0.6540, Macro-Recall: 0.6427
硬标签蒸馏模型已保存到：../08-bert-emotion/save_models/bert_emotion_classifier_bilstm_model_hard.pt
硬标签蒸馏第10轮训练：100% | 434/434 [01:21<00:00, 5.34it/s]
```

#### 软标签蒸馏结果

```
Macro-F1: 0.7132, 准确度：0.7341, Macro-Precision: 0.7062, Macro-Recall: 0.7257
软标签蒸馏第10轮训练：100% | 434/434 [01:20<00:00, 5.37it/s]
软标签蒸馏完成，最佳Macro-F1: 0.7238
```

#### 混合蒸馏结果

```
neutral      0.88    0.79    0.83    420
sad          0.50    0.56    0.53    346
surprise     0.52    0.54    0.53    170

accuracy     0.70    1997
macro avg    0.67    0.67    0.67    1997
weighted avg 0.71    0.70    0.70    1997

Macro-F1: 0.6677, 准确度：0.6970, Macro-Precision: 0.6655, Macro-Recall: 0.6728
混合蒸馏第10轮训练：100% | 434/434 [01:21<00:00, 5.35it/s]
当前学习率：0.00001000
混合蒸馏训练完成，最佳Macro-F1: 0.6683
```

## License

本项目仅供学习和研究使用。

