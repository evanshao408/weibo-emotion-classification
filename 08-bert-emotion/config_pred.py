# 导包
import torch
import datetime
import json
import os

from transformers import BertTokenizer, BertConfig


# todo 1.定义变量, 记录当前时间(年月日格式)
current_date = datetime.datetime.now().strftime('%Y%m%d')


# todo 2. 定义配置文件类, 集中管理模型和训练所需的参数.
class Config(object):
    def __init__(self):
        # 1. 基础的模型信息
        self.model_name = "bert_tuned"

        # 2. 路径配置 - 使用绝对路径，避免问题
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.train_datapath = os.path.join(self.root_path, 'data/usual_train.json')
        self.test_datapath = os.path.join(self.root_path, 'data/usual_test_labeled.json')
        self.dev_datapath = os.path.join(self.root_path, 'data/usual_eval_labeled.json')

        self.class_list = self._get_class_list()
        self.label2id = {label: idx for idx, label in enumerate(self.class_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.class_list)}

        # 调优版使用独立模型路径, 避免覆盖你当前喜欢的版本
        self.model_save_path = os.path.join(self.root_path, "save_models/bert_emotion_classifier_model_tuned_v2.pt")
        self.finetuned_model_path = self.model_save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. BERT模型相关配置
        self.bert_path = os.path.join(self.root_path, "bert-base-emotion")
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path)

        # 4. 调优版训练参数配置
        self.num_classes = len(self.class_list)
        self.num_epochs = 5
        self.batch_size = 64
        self.pad_size = 128
        self.bert_learning_rate = 1e-5
        self.head_learning_rate = 5e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.06
        self.label_smoothing = 0.05
        self.rdrop_alpha = 1.0

        # 5. 量化模型存放地址
        self.bert_model_quantization_model_path = os.path.join(self.root_path, "save_models/bert_emotion_classifier_quantization_model_tuned_v2.pt")

        # 6. 蒸馏模型存放地址
        self.bert_model_distill_model_path_hard = os.path.join(self.root_path, "save_models/bert_emotion_classifier_bilstm_model_hard.pt")
        self.bert_model_distill_model_path_soft = os.path.join(self.root_path, "save_models/bert_emotion_classifier_bilstm_model_soft.pt")
        self.bert_model_distill_model_path_hybrid = os.path.join(self.root_path, "save_models/bert_emotion_classifier_bilstm_model_hybrid.pt")

        # 7. bert模型蒸馏 -> BiLSTM模型的参数配置.
        self.embed_size = 128
        self.hidden_size_lstm = 256
        self.student_epochs = 10
        self.lstm_learning_rate = 1e-3
        self.dropout = 0.3
        self.num_layers = 3
        # 8. 蒸馏温度和软标签权重
        self.distill_temperature = 2.0
        self.distill_alpha = 0.5

    def _get_class_list(self):
        """
        从训练数据中提取所有唯一的情感标签
        """
        data_path = self.train_datapath
        if not os.path.exists(data_path):
            # 如果找不到数据，返回默认标签列表
            return ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        labels = set()
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'label' in item:
                    labels.add(item['label'])
        return sorted(list(labels))


if __name__ == '__main__':
    config = Config()
    print(f"设备: {config.device}")
    print(f"调优版模型保存路径: {config.model_save_path}")
    print(f"num_epochs={config.num_epochs}, batch_size={config.batch_size}, pad_size={config.pad_size}")
    print(f"bert_lr={config.bert_learning_rate}, head_lr={config.head_learning_rate}, warmup_ratio={config.warmup_ratio}, label_smoothing={config.label_smoothing}, rdrop_alpha={config.rdrop_alpha}")

