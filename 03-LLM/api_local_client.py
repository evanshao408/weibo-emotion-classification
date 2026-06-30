# 该py文件的作用是: 创建一个Flask应用, 并且启动应用(可以理解为: 文字版的客户端)

# 导包
import requests

# 定义接口地址
url = "http://127.0.0.1:5000/predict"

# 测试调用
try:
    # 测试调用API接口
    text = input('请输入微博文本: ')
    # 发送请求
    re = requests.post(url=url, json={"text": text})
    # 打印结果
    print(re.json())

except Exception as e:
    print(f'出问题了, {e}')