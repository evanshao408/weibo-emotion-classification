# 该.py文件的作用是 -> 获取到 数据集加载器(DataLoader), 使用调优版配置.

import json
import os

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from config_tuned import Config


conf = Config()


def load_raw_data(file_path):
    """
    从指定JSON文件中加载数据, 处理为: (文本内容, 标签索引) 的元组列表.
    """
    result = []
    json_path = file_path
    if not os.path.exists(json_path):
        json_path = os.path.join('data', os.path.basename(file_path))

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in tqdm(data, desc=f'加载原始数据: {json_path}'):
        text = item.get('content', '').strip()
        label_name = item.get('label')
        if not text or label_name is None:
            continue

        label_id = conf.label2id[label_name]
        result.append((text, label_id))

    return result


class TextDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        text, label = self.data_list[index]
        return text, label


def collate_fn(batch):
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    text_tokens = conf.tokenizer(
        texts,
        add_special_tokens=True,
        padding='max_length',
        max_length=conf.pad_size,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = text_tokens['input_ids']
    attention_mask = text_tokens['attention_mask']
    labels = torch.tensor(labels)
    return input_ids, attention_mask, labels


def build_dataloader():
    train_data_list = load_raw_data(conf.train_datapath)
    dev_data_list = load_raw_data(conf.dev_datapath)
    test_data_list = load_raw_data(conf.test_datapath)

    train_dataset = TextDataset(train_data_list)
    dev_dataset = TextDataset(dev_data_list)
    test_dataset = TextDataset(test_data_list)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    print(len(train_dataloader), len(dev_dataloader), len(test_dataloader))
