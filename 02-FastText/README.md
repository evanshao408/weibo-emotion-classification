# 微博情绪预测系统

基于 FastText 文本分类模型的微博情绪预测系统，支持对微博文本进行情感识别，可自动判别文本背后的情绪倾向。

## 🎯 情绪类别

系统支持识别以下 6 种情绪类别：

| 类别 ID | 情绪名称 | 说明 |
| :--- | :--- | :--- |
| 0 | angry | 愤怒 |
| 1 | fear | 恐惧 |
| 2 | happy | 开心 |
| 3 | neutral | 中性 |
| 4 | sad | 悲伤 |
| 5 | surprise | 惊讶 |

## 📁 项目结构

```
Weibo_Emotion_Classification/
├── data/
│   └── clean/
│       ├── class.txt          # 情绪类别定义
│       ├── eval.json          # 验证集
│       ├── test.json          # 测试集
│       └── train.json         # 训练集
├── fasttext/
│   ├── final_data/            # 预处理后的训练数据
│   │   ├── eval_process_char.txt
│   │   ├── eval_process_word.txt
│   │   ├── test_process_char.txt
│   │   ├── test_process_word.txt
│   │   ├── train_process_char.txt
│   │   └── train_process_word.txt
│   ├── save_models/           # 训练好的模型文件
│   │   ├── model_char_1_default.bin
│   │   ├── model_char_2_auto.bin
│   │   ├── model_word_1_default.bin
│   │   └── model_word_2_auto.bin
│   ├── api_flask_server.py    # Flask API 服务端
│   ├── api_local_client.py    # 本地测试客户端
│   ├── app_streamlit.py       # Streamlit 前端页面
│   ├── app_streamlit_run.py   # Streamlit 运行入口
│   ├── config.py              # 配置文件
│   ├── data_preprocess.py     # 数据预处理脚本
│   ├── fasttext_char_1_default.py   # 字符级模型训练（默认参数）
│   ├── fasttext_char_2_auto.py      # 字符级模型训练（自动调参）
│   ├── fasttext_word_1_default.py   # 词语级模型训练（默认参数）
│   ├── fasttext_word_2_auto.py      # 词语级模型训练（自动调参）
│   └── ft_predict_fun.py      # 预测函数封装
└── README.md
```

## 🛠️ 技术栈

- **Python 3.x**
- **FastText** - 文本分类模型
- **jieba** - 中文分词库
- **Flask** - API 服务框架
- **Streamlit** - 前端页面框架

## 🚀 快速开始

### 1. 环境准备

```bash
pip install fasttext jieba flask streamlit
```

### 2. 数据预处理

将原始 JSON 数据转换为 FastText 所需的格式（支持字符级和词语级两种切分模式）：

```bash
cd fasttext
python data_preprocess.py
```

### 3. 模型训练

系统提供 4 种训练脚本，分别对应字符级/词语级和默认参数/自动调参：

```bash
# 字符级模型（默认参数）
python fasttext_char_1_default.py

# 字符级模型（自动调参）
python fasttext_char_2_auto.py

# 词语级模型（默认参数）
python fasttext_word_1_default.py

# 词语级模型（自动调参）
python fasttext_word_2_auto.py
```

训练完成后，模型文件保存在 `fasttext/save_models/` 目录下。

### 4. 启动 API 服务

```bash
python api_flask_server.py
```

服务默认运行在 `http://192.168.12.116:5000`

### 5. 启动前端页面

```bash
streamlit run app_streamlit.py
```

或使用运行入口脚本：

```bash
python app_streamlit_run.py
```

## 🔌 API 接口

### 情绪预测接口

- **URL**: `/NewCls_handler`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### 请求示例

```json
{
    "text": "今天出门遇见晚霞，晚风温柔，一整天都觉得舒心自在"
}
```

#### 响应示例

```json
{
    "text": "今天出门遇见晚霞，晚风温柔，一整天都觉得舒心自在",
    "pred_class": "happy",
    "cost_time": 0.0023
}
```

## 📊 模型说明

| 模型文件 | 切分模式 | 参数配置 |
| :--- | :--- | :--- |
| `model_char_1_default.bin` | 字符级 | 默认参数 |
| `model_char_2_auto.bin` | 字符级 | 自动调参 |
| `model_word_1_default.bin` | 词语级 | 默认参数 |
| `model_word_2_auto.bin` | 词语级 | 自动调参 |

当前预测函数默认加载 `model_char_2_auto.bin` 模型，如需更换其他模型，修改 `ft_predict_fun.py` 中的模型路径即可。

## � 模型评估结果

### 验证集（eval）

| 指标 | 值 |
| :--- | :--- |
| 总样本数 | 3993 |
| 准确率（Accuracy） | 0.726 |
| 精确率（Precision） | 0.726 |
| 召回率（Recall） | 0.726 |
| F1-Score | 0.726 |

**各类情绪细分指标：**

| 情绪类别 | 精确率（P） | 召回率（R） | F1-Score |
| :--- | :--- | :--- | :--- |
| happy | 0.8038 | 0.8295 | 0.8165 |
| angry | 0.7366 | 0.767 | 0.7515 |
| neutral | 0.7614 | 0.7298 | 0.7453 |
| sad | 0.5654 | 0.5499 | 0.5575 |
| surprise | 0.544 | 0.4839 | 0.5122 |
| fear | 0.5316 | 0.5185 | 0.525 |

### 测试集（test）

| 指标 | 值 |
| :--- | :--- |
| 总样本数 | 7985 |
| 准确率（Accuracy） | 0.7389 |
| 精确率（Precision） | 0.7389 |
| 召回率（Recall） | 0.7389 |
| F1-Score | 0.7389 |

**各类情绪细分指标：**

| 情绪类别 | 精确率（P） | 召回率（R） | F1-Score |
| :--- | :--- | :--- | :--- |
| happy | 0.8232 | 0.8335 | 0.8283 |
| angry | 0.7576 | 0.7757 | 0.7666 |
| neutral | 0.7453 | 0.7626 | 0.7539 |
| sad | 0.5932 | 0.5916 | 0.5924 |
| surprise | 0.5711 | 0.509 | 0.5383 |
| fear | 0.6291 | 0.553 | 0.5753 |

## �💡 使用提示

- 输入单句、长段微博均可识别
- 包含表情、网络词汇不影响识别效果
- 完整短句，情绪判断结果更精准
