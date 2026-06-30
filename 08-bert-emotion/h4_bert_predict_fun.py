# 该.py文件的作用是 -> 预测分类的函数版, 对接后续的 API 和 APP版.

# 导包.
import time
import warnings

import torch

from config import Config
from h2_bert_classifier_model import BertClassifier

# 压制警告.
warnings.filterwarnings("ignore")


# todo 1.加载全局配置.
conf = Config()

# todo 2. 准备BERT预测模型, 直接加载已经训练好的模型参数.
model = BertClassifier(load_finetuned=True).to(conf.device)
model.eval()  # 设置模型为评估模式


# todo 3. 定义预测函数, 接收文本数据, 返回情感分类结果.
def predict_fun(data_dict):
    """
    接收包含文本的字典, 通过BERT模型预测文本情感类别, 返回带预测结果的字典.
    :param data_dict: 输入字典, 格式为: {'text': '待预测文本内容'}
    :return: {'text': '待预测文本内容', 'pred_class': '情感标签', 'pred_index': 预测索引, 'confidence': 置信度}
    """
    start_time = time.time()

    # 1. 提取输入文本, 获取待预测的字符串
    text = data_dict['text'].strip()
    if not text:
        raise ValueError("输入文本不能为空")

    # 2. 文本编码, 将原始文本 -> BERT模型可识别的token id
    output = conf.tokenizer(
        [text],
        add_special_tokens=True,
        padding='max_length',
        max_length=conf.pad_size,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # 3. 提取模型所需要的特征
    input_ids = output['input_ids'].to(conf.device)
    attention_mask = output['attention_mask'].to(conf.device)

    # 4. 模型预测动作. 禁用梯度计算以提高效率并减少内存占用
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
        pred_label = conf.id2label[pred_index]
        confidence = probs[0][pred_index].item()

    # 5. 返回更完整的预测结果
    result = dict(data_dict)
    result['pred_index'] = pred_index
    result['pred_class'] = pred_label
    result['confidence'] = round(confidence, 6)
    result['probabilities'] = {
        label: round(prob, 6)
        for label, prob in zip(conf.class_list, probs[0].cpu().tolist())
    }
    result['cost_time'] = round((time.time() - start_time) * 1000, 3)
    return result


# todo 4. 测试代码.
if __name__ == '__main__':
    # 1. 创建测试数据集
    data_dict = {'text': ''}
    # 2. 调用预测接口, 并打印结果
    print(predict_fun(data_dict))
