import time
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv('.env_Qwen')
# 加载系统提示词
SYSTEM_PROMPT = open("./大模型生成提示词", encoding='utf-8').read()


def predict_fun(data):
    client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=os.getenv("base_url"))
    # 开始时间
    start_time = time.time()
    #
    response = client.chat.completions.create(
        model="qwen3.6-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data['text']},
        ],
        stream=False
    )

    result = response.choices[0].message.content
    elapsed_time = (time.time() - start_time) * 1000
    data['pred_class'] = result
    data['cost_time'] = elapsed_time
    return data

def predict_batch(data):
    """
    批量预测: 传入文本列表, 返回包含预测结果和耗时的列表
    内部调用 predict_fun
    :param data:
    :return:
    """
    returns = []
    total = len(data)
    for i, item in enumerate(data):
        print(f"正在处理第{i+1}/{total}条数据")
        result = predict_fun(item)
        results.append(result)
    return results

if __name__ == '__main__':
    data = {'text': '手机买了12天就降了300，摩托你的手机情怀呢？'}
    result = predict_fun(data)
    print(f"*预测类别：{result['pred_class']}")
    print(f"*请求耗时：{result['cost_time']:.2f}ms")