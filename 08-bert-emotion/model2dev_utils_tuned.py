import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

from config_tuned import Config


conf = Config()


def model2dev(model, data_loader, device):
    """
    在验证或测试集上评估 BERT 分类模型的性能。
    返回: (分类报告, 宏平均F1分数, 准确度, 宏平均精确度, 宏平均召回率)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            y_pred = model(input_ids, attention_mask)
            y_pred_labels = torch.argmax(y_pred, dim=-1)

            all_preds.extend(y_pred_labels.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(conf.num_classes)),
        target_names=conf.class_list,
        zero_division=0
    )
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return report, macro_f1, accuracy, precision, recall
