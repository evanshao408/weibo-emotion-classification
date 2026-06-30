import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from config_tuned import Config


conf = Config()


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = self._load_pretrained_backbone()
        self.dropout = nn.Dropout(conf.bert_config.hidden_dropout_prob)
        self.norm = nn.LayerNorm(conf.bert_config.hidden_size)
        self.fc1 = nn.Linear(conf.bert_config.hidden_size, 256)
        self.fc2 = nn.Linear(256, conf.num_classes)

    def _load_pretrained_backbone(self):
        bert_model = BertModel(conf.bert_config)
        checkpoint_path = os.path.join(conf.bert_path, 'pytorch_model.bin')
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        bert_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '', 1)
                if new_key == 'embeddings.position_ids':
                    continue
                bert_state_dict[new_key] = value

        bert_model.load_state_dict(bert_state_dict, strict=True)
        return bert_model

    @staticmethod
    def masked_mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self.masked_mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        x = self.fc1(pooled)
        x = F.gelu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

