"""
案例: 演示 随机森林版的 模型预测.
"""
import time

# 导包
import fasttext
import jieba
import pickle

from PIL.ImImagePlugin import split

from config import Config

# todo 1. 加载配置文件.
# 加载配置
config = Config()


# todo 2. 加载模型.
model = fasttext.load_model(config.ft_model_save_path + '/model_char_2_auto.bin')

# todo 3. 定义预测函数.
def predict_fun(data):  # data就是待预测的数据, 例如: {'text': '2011年全国各地高考各科考试时间汇总'}
    """
    :param data:用户传递过来的数据，字典格式{"text":"中华女子学院：本科层次仅1专业招男生"}
    :return:返回data {"text":"中华女子学院：本科层次仅1专业招男生","pred_class":"education"}
    """
    start_time = time.perf_counter()
    # 切分并用空格拼接
    split_words = " ".join(list(data['text']))
    # 预测对应的标签
    res = model.predict(split_words)
    # print(res)      # (('__label__education',), array([0.99991202]))
    # print(type(res))      # <class 'tuple'>
    # label_name = res[0][0][9:]
    label_name = res[0][0].replace("__label__", "")
    # print(label_name)
    data['pred_class'] = label_name
    end_time = time.perf_counter()
    # print(f"预测耗时：{end_time - start_time}")
    data['cost_time'] = end_time - start_time
    return data

# todo 4. 测试
if __name__ == '__main__':
    result = predict_fun(data = {'text': '2011年全国各地高考各科考试时间汇总'})
    print(f"预测的标签是：{result}")
