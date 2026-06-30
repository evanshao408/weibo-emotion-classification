# 投满分 - 微博情绪分类系统

基于 **SMP2020-EWECT 微博情绪分类技术评测的完整项目，包含从数据清洗、机器学习基线、深度学习优化到AI赋能的全链路解决方案。

## 📚 项目背景

本项目基于 **SMP2020-EWECT 微博情绪分类技术评测（[官网](https://smp2020ewect.github.io/)），该评测由中国中文信息学会社会媒体处理专委会主办，哈尔滨工业大学承办。

### 评测任务

微博情绪分类任务旨在识别微博中蕴含的情绪，输入是一条微博，输出是该微博所蕴含的情绪类别。本次评测将微博情绪分为以下六个类别：

| 类别   | 标签     | 说明            |
| ------ | -------- | --------------- |
| 积极   | happy    | 正面情绪        |
| 愤怒   | angry    | 愤怒情绪        |
| 悲伤   | sad      | 悲伤情绪        |
| 恐惧   | fear     | 恐惧情绪        |
| 惊奇   | surprise | 惊奇情绪        |
| 无情绪 | neutral  | 中性/无明显情绪 |

### 评价指标

评测采用**宏平均F1值**作为评估指标：

```
Macro_F = (Macro_F通用 + Macro_F疫情) / 2
```

## 📁 项目结构

```
news-compass/
├── 00-raw-data/              # 数据清洗组
│   ├── raw/                   # 原始数据
│   ├── clean/                 # 清洗后数据
│   └── clean_data.py          # 数据清洗脚本
├── 01-rf/                    # 随机森林组（Baseline）
│   ├── 1-data/                # 数据处理模块
│   ├── 2-rf/                  # 随机森林模型模块
│   └── 随机森林项目-README.md
├── 02-FastText/              # FastText组
│   ├── data/                  # 数据目录
│   ├── fasttext/              # FastText模型模块
│   └── README.md
├── 03-LLM/                   # LLM组（AI赋能）
│   ├── weibo/                # 评测相关
│   ├── model2pred_Qwen.py     # 千问大模型推理
│   └── api_flask_server.py
├── 08-bert-emotion/          # BERT组
│   ├── bert-base-emotion/     # BERT预训练模型
│   ├── data/                # 数据集
│   ├── save_models/         # 保存的模型
│   ├── config_tuned.py        # 配置文件
│   ├── h1_dataloader_utils_tuned.py
│   ├── h2_bert_classifier_model_tuned.py
│   ├── h3_bilstm_classifier_model_tuned.py
│   ├── h3_train_predict_tuned.py
│   ├── h4_hard_label_distillation_tuned.py
│   ├── h5_soft_label_distillation_tuned.py
│   ├── h6_soft_hybrid_label_distillation_tuned.py
│   └── bert项目-README.md
├── requirements.txt           # 项目依赖
└── README.md              # 本文件
```

## 🎯 分组介绍

### 第一组：00-raw-data（数据清洗组）

**职责**：负责原始数据的清洗和预处理

- 原始数据格式转换
- 数据去重、去噪
- 数据格式标准化

### 第二组：01-rf（随机森林基线组）

**职责**：建立项目基线，验证项目可行性

**技术栈**：

- scikit-learn 随机森林
- TF-IDF 文本向量化
- jieba 中文分词
- Flask + Streamlit 部署

**主要成果：

- 数据探索性分析（EDA）
- 随机森林模型训练
- API服务部署

### 第三组：02-FastText（FastText优化组）

**职责**：使用FastText模型优化性能

**技术栈**：

- FastText 文本分类
- jieba 中文分词
- Flask + Streamlit 部署

**主要成果**：

- 字符级/词语级两种分词方案
- 默认参数/自动调参两种训练策略
- 完整的API服务和前端界面

### 第四组：08-bert-emotion（BERT深度学习组）

**职责**：使用BERT深度学习模型进一步优化，并实现知识蒸馏

**技术栈**：

- PyTorch + Transformers
- BERT预训练模型
- BiLSTM学生模型
- 知识蒸馏技术（硬标签、软标签、混合标签）

**主要成果**：

- BERT教师模型（Macro-F1: ~0.78+
- 知识蒸馏到BiLSTM学生模型
- 三种蒸馏策略实现

### 第五组：03-LLM（AI赋能组）

**职责**：使用大语言模型进行AI赋能

**技术栈**：

- 千问大模型（Qwen）
- LangChain
- Flask API服务

## 🛠️ 技术栈总览

| 技术类别     | 技术选型              |
| ------------ | --------------------- |
| **编程语言** | Python 3.8+           |
| **机器学习** | scikit-learn          |
| **深度学习** | PyTorch, Transformers |
| **文本分类** | FastText, BERT        |
| **中文分词** | jieba                 |
| **API服务**  | Flask                 |
| **前端界面** | Streamlit             |
| **大模型**   | 千问（Qwen）          |

## 🚀 快速开始

### 环境要求

- Python ≥ 3.8

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据集准备

原始数据位于 `00-raw-data/raw/` 目录下，清洗后数据位于 `00-raw-data/clean/` 目录。

各子项目会复制或引用这些数据。

### 快速体验各模型

#### 1. 随机森林Baseline

```bash
cd 01-rf/2-rf
python rf_train.py
python api_flask_server.py
streamlit run app_streamlit.py
```

#### 2. FastText模型

```bash
cd 02-FastText/fasttext
python data_preprocess.py
python fasttext_char_2_auto.py
python api_flask_server.py
```

#### 3. BERT模型

```bash
cd 08-bert-emotion
python h3_train_predict_tuned.py
python h6_soft_hybrid_label_distillation_tuned.py
```

#### 4. LLM大模型

```bash
cd 03-LLM
python api_flask_server.py
```

## 📊 模型性能对比

| 模型                | Macro-F1 | 特点           |
| ------------------- | -------- | -------------- |
| 随机森林 (Baseline) | ~0.70    | 简单快速       |
| FastText            | ~0.74    | 轻量高效       |
| BERT (教师模型)     | ~0.78+   | 效果最佳       |
| BiLSTM (蒸馏模型)   | ~0.72    | 速度快，体积小 |

## 🔌 部署架构

所有子项目均采用统一的部署架构：

```
用户输入 → Streamlit前端 → Flask API → 预测函数 → 返回结果
```

## 💡 项目亮点

1. **全链路覆盖**：从数据清洗、基线、优化到AI赋能
2. **多方案对比**：传统机器学习、深度学习、大模型三种方案
3. **知识蒸馏**：将大模型知识迁移到小模型，兼顾效果与效率
4. **工程化落地**：完整的API服务和前端界面，可直接部署使用



## 📄 License

本项目仅供学习和研究使用。

## 👥 团队成员

- 数据清洗组
- 随机森林组
- FastText组
- BERT组
- LLM组

