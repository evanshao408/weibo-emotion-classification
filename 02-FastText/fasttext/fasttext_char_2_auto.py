'''# 该py文件用于: 字符级别 默认参数训练

# 导包
import fasttext
from config import Config
import datetime
import os

# todo 1. 加载项目配置
config = Config()

# todo 2. 模型训练: 使用 fasttext训练 字符级 文本分类模型
# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
# model = fasttext.train_supervised(
#     input=config.process_train_datapath_char,
#     autotuneValidationFile=config.process_eval_datapath_char,
#     autotuneDuration = 120,
#     thread = 1,
#     verbose = 1
#
#     )
model = fasttext.train_supervised(
    input=config.process_train_datapath_char,
    autotuneValidationFile=config.process_eval_datapath_char,
    autotuneDuration = 600,
    autotuneMetric="f1",
    thread = 4,
    verbose = 1
    )

# # todo 3. 打印模型训练后的关键信息.
# # get_word_vector方法, 获取字符'日'的向量表示(长度: 10维), 查看数字的特征, 例如:
# # [ 0.06251448  0.21214555 -0.5582269  -0.2776528  -0.40804806 -0.2712767, -0.03183828 -0.10957453  0.31537786  0.19355372]
# print(model.get_word_vector('日'))
# print(len(model.get_word_vector('日')))
#
# # labels属性, 获取模型训练到的 所有类别标签
# print(model.labels)
# # 获取模型训练到的 类别数量
# print(len(model.labels))
#
# # get_words方法, 获取模型训练到的 所有单词及对应的词频
# print(model.get_words(include_freq=True))
#
# # 用zip()函数, 配对: 字符和频率, 转为列表, 方便查看每个字符对应的出现次数.
# print(list(zip(*model.get_words(include_freq=True))))

# todo 4. 模型保存. save_model方法保存模型
model_path = config.ft_model_save_path + '/model_char_2_auto.bin'
model.save_model(model_path)
print('模型保存成功！')

# todo 5. 模型预测. predict方法预测结果
# print(model.predict('网 又 没 了 , 又 在 挂 流 量 , 要 崩 溃 。'))

# todo 6. words属性, 查看模型学习到的词汇表
print(model.words[:10])

# todo 7. get_subwords方法, 模型子词查看.
# print(model.get_subwords('日本地震海啸!'))

# todo 8. get_dimension方法, 查看模型的向量维度.
print(model.get_dimension())

# todo 9. test方法, 模型评估.  (样本数, 精确率, 召回率)
print(model.test(config.process_test_datapath_char))   # (7985, 0.7386349405134628, 0.7386349405134628)
print(model.test(config.process_eval_datapath_char))    # (3993, 0.7265214124718257, 0.7265214124718257)
'''
# 该py文件用于: 字符级别 自动调整参数训练
# 导包
import fasttext
from config import Config
import datetime
import os

# todo 1. 加载项目配置
config = Config()

# todo 2. 模型训练: 使用 fasttext训练 字符级 文本分类模型
# fasttext.train_supervised(): fasttext的核心训练函数, 用于训练: 有监督学习分类模型
model = fasttext.train_supervised(
    input=config.process_train_datapath_char,
    autotuneValidationFile=config.process_eval_datapath_char,
    autotuneDuration=300,
    autotuneMetric="f1",
    thread=1,
    verbose=1
)

# todo 3. 打印模型训练后的关键信息.
#print(model.get_word_vector('天'))
# labels属性, 获取模型训练到的 所有类别标签
print("全部分类标签：", model.labels)
# 获取模型训练到的 类别数量
print("情感类别总数：", len(model.labels))
# get_words方法, 获取模型训练到的 所有单词及对应的词频
words, freqs = model.get_words(include_freq=True)
print("字符与对应词频：", list(zip(words, freqs)))
print("*"*30)

# todo 4. 模型保存. save_model方法保存模型
save_full_path = os.path.join(config.ft_model_save_path, 'model_char_2_auto.bin')
os.makedirs(config.ft_model_save_path, exist_ok=True)
model.save_model(save_full_path)
print(f"模型已保存至：{save_full_path}")
print("*"*30)

# todo 5. 模型预测. predict方法预测结果
test_text = '网 又 没 了 , 又 在 挂 流 量 , 要 崩 溃 。'
pred_res = model.predict(test_text)
print(f"单条文本预测结果：文本={test_text}，预测标签={pred_res[0][0]}，置信度={pred_res[1][0]:.4f}")
print("*"*30)

# todo 6. words属性, 查看模型学习到的词汇表
print("前10个训练字符：", model.words[:10])
print("*"*30)

# todo 7. get_subwords方法, 模型子词查看.
sub_res = model.get_subwords('日本地震海啸!')
print("模型拆分子词：", sub_res[0])
print("*"*30)

# todo 8. get_dimension方法, 查看模型的向量维度.
print(f"词向量维度：{model.get_dimension()}")
print("*"*50)

# ===================== 新增：完整评估函数 计算Accuracy/Precision/Recall/F1 =====================
def calc_full_metrics(ft_model, data_file):
    # fasttext原生接口获取全局样本、宏精确率、宏召回率
    total_num, global_p, global_r = ft_model.test(data_file)
    # 计算全局F1
    global_f1 = 2 * global_p * global_r / (global_p + global_r) if (global_p + global_r) != 0 else 0.0
    # 初始化每类统计TP/FP/FN
    stat_dict = {lab: {"tp":0, "fp":0, "fn":0} for lab in ft_model.labels}
    total_correct = 0
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ", 1)
            # 跳过残缺脏行，避免解包报错
            if len(parts) != 2:
                print(f"跳过异常脏行：{line}")
                continue
            true_label, text = parts
            pred_label, _ = ft_model.predict(text)
            pred_label = pred_label[0]
            if pred_label == true_label:
                total_correct += 1
                stat_dict[true_label]["tp"] += 1
            else:
                stat_dict[true_label]["fn"] += 1
                stat_dict[pred_label]["fp"] += 1
    # 计算全局准确率Accuracy
    acc = total_correct / total_num if total_num > 0 else 0.0
    # 计算每个类别的P/R/F1
    class_metric = {}
    for label, s in stat_dict.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_metric[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4)
        }
    return {
        "total_sample": total_num,
        "accuracy": round(acc, 4),
        "global_precision": round(global_p, 4),
        "global_recall": round(global_r, 4),
        "global_f1": round(global_f1, 4),
        "each_class": class_metric
    }

# todo 9. 完整输出验证集、测试集四大评估指标
print("==================== 验证集(eval) 完整评估指标 ====================")
dev_metrics = calc_full_metrics(model, config.process_eval_datapath_char)
print(f"总样本数量：{dev_metrics['total_sample']}")
print(f"全局准确率 Accuracy：{dev_metrics['accuracy']}")
print(f"全局精确率 Precision：{dev_metrics['global_precision']}")
print(f"全局召回率 Recall：{dev_metrics['global_recall']}")
print(f"全局F1-Score：{dev_metrics['global_f1']}")
print("每一类情感细分指标：")
for cls, m in dev_metrics["each_class"].items():
    print(f"  {cls} -> P:{m['precision']}, R:{m['recall']}, F1:{m['f1']}")
print("-" * 60)

print("==================== 测试集(test) 完整评估指标 ====================")
test_metrics = calc_full_metrics(model, config.process_test_datapath_char)
print(f"总样本数量：{test_metrics['total_sample']}")
print(f"全局准确率 Accuracy：{test_metrics['accuracy']}")
print(f"全局精确率 Precision：{test_metrics['global_precision']}")
print(f"全局召回率 Recall：{test_metrics['global_recall']}")
print(f"全局F1-Score：{test_metrics['global_f1']}")
print("每一类情感细分指标：")
for cls, m in test_metrics["each_class"].items():
    print(f"  {cls} -> P:{m['precision']}, R:{m['recall']}, F1:{m['f1']}")