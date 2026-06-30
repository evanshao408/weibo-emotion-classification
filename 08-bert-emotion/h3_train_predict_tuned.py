# 该.py文件的作用 -> 调优版训练和预测.

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm

from config_tuned import Config
from model2dev_utils_tuned import model2dev
from h1_dataloader_utils_tuned import build_dataloader
from h2_bert_classifier_model_tuned import BertClassifier

import warnings

warnings.filterwarnings("ignore")


conf = Config()


def _kl_divergence_with_logits(p_logits, q_logits):
    p_log_prob = torch.log_softmax(p_logits, dim=-1)
    q_prob = torch.softmax(q_logits, dim=-1)
    return torch.nn.functional.kl_div(p_log_prob, q_prob, reduction='batchmean')


def model2train():
    train_dataloader, dev_dataloader, test_dataloader = build_dataloader()
    model = BertClassifier().to(conf.device)

    train_labels = [label for _, label in train_dataloader.dataset.data_list]
    label_counts = torch.bincount(torch.tensor(train_labels), minlength=conf.num_classes).float()
    class_weights = torch.sqrt(label_counts.sum() / (conf.num_classes * label_counts))
    class_weights = class_weights.to(conf.device)

    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=conf.label_smoothing)
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    bert_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('bert.'):
            bert_params.append(param)
        else:
            head_params.append(param)

    optimizer = AdamW(
        [
            {"params": bert_params, "lr": conf.bert_learning_rate, "weight_decay": conf.weight_decay},
            {"params": head_params, "lr": conf.head_learning_rate, "weight_decay": conf.weight_decay},
        ],
        betas=(0.9, 0.999)
    )

    total_training_steps = len(train_dataloader) * conf.num_epochs
    warmup_steps = int(total_training_steps * conf.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    best_macro_f1 = 0.0
    print(f"调优版类别权重: {class_weights.detach().cpu().tolist()}")
    print(f"总训练步数: {total_training_steps}, warmup步数: {warmup_steps}")

    for epoch in range(conf.num_epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for _, batch in enumerate(tqdm(train_dataloader, desc=f'调优版第{epoch + 1}轮训练')):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(conf.device)
            attention_mask = attention_mask.to(conf.device)
            labels = labels.to(conf.device)

            if conf.rdrop_alpha and conf.rdrop_alpha > 0:
                logits1 = model(input_ids, attention_mask)
                logits2 = model(input_ids, attention_mask)
                ce = (criterion(logits1, labels) + criterion(logits2, labels)) / 2
                kl = (_kl_divergence_with_logits(logits1, logits2) + _kl_divergence_with_logits(logits2, logits1)) / 2
                loss = ce + conf.rdrop_alpha * kl
            else:
                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_count % 100 == 0 or batch_count == len(train_dataloader):
                report, macro_f1, accuracy, precision, recall = model2dev(model, dev_dataloader, conf.device)
                print(f"\n当前轮次: {epoch + 1}/{conf.num_epochs}, 当前批次: {batch_count}/{len(train_dataloader)}")
                print(f"分类报告: \n{report}")
                print(f"Macro-F1: {macro_f1:.4f}, 准确度: {accuracy:.4f}, Macro-Precision: {precision:.4f}, Macro-Recall: {recall:.4f}")
                print(f"平均损失是:{total_loss / batch_count:.4f}")
                lr_list = scheduler.get_last_lr()
                if len(lr_list) >= 2:
                    print(f"当前学习率: bert={lr_list[0]:.8f}, head={lr_list[1]:.8f}")
                else:
                    print(f"当前学习率: {lr_list[0]:.8f}")

                if macro_f1 > best_macro_f1:
                    torch.save(model.state_dict(), conf.model_save_path)
                    print(f"模型已保存到: {conf.model_save_path}")
                    best_macro_f1 = macro_f1

                model.train()

    print(f"\n调优版训练完成, 最佳Macro-F1: {best_macro_f1:.4f}")


if __name__ == '__main__':
    model2train()
