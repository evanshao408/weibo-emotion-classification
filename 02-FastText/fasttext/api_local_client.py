import requests
import time

start_time = time.time()
print("开始时间：",start_time)

url = "http://127.0.0.1:5000/NewCls_handler"
data = {"text":"孙颖莎中国大满贯夺冠"}
res = requests.post(url,json=data)
print('最终的预测结果为：', res.json()["pred_class"])
end_time = time.time()
cost_time = end_time - start_time
print('结束时间：',end_time)
print('单条样本耗时：',cost_time * 1000, 'ms')