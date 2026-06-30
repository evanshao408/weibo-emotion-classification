"""
案例: 演示 随机森林版的 模型预测. 主要应用于实时计算, 也就是传进来一条数据以后, 马上预测出对应的类别
"""

# 导包
import jieba
import pickle
from config import Config


# todo 1. 加载配置文件.
# 加载配置
conf = Config()

# todo 2. 加载模型和向量化器.
# 1. 加载随机森林模型(rf)
with open(conf.rf_model_save_path, 'rb') as f:
    rf = pickle.load(f)

# 2. 加载向量化器(tfidf)
with open(conf.tfidf_model_save_path, 'rb') as f:
    tfidf = pickle.load(f)

# 3. 加载标签编码器
with open(conf.label_encoder_save_path, 'rb') as f:
    label_encoder = pickle.load(f)


# todo 3. 定义预测函数.
def predict_fun(data):      # data: 就是待预测的数据,字典格式,  例如:  {'text': '2011年全国各地高考各科考试时间汇总'}
    # 1. 分词
    words = " ".join(list(data['context'])[:200])
    # 2. 向量化.
    features = tfidf.transform([words])
    # 3. 预测.
    y_pred_id = rf.predict(features)[0]
    # 4. 将数值标签转换为原始字符串标签.
    y_pred = label_encoder.inverse_transform([y_pred_id])[0]

    # 5. 给原始数据新增1列(预测的类别), 并返回.
    data['pred_class'] = y_pred

    # 6. 返回结果, 例如: {'context': '这弄得什么东西真没啥用', 'pred_class': 'angry'}
    return data


# todo 4. 测试
if __name__ == '__main__':
    data = {'context': '这弄得什么东西真没啥用'}
    result = predict_fun(data)
    print(result)

