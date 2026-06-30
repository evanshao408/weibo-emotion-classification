# 该py文件的主要任务: 通过Flask组件, 构建 路由 + 预测函数的 应用.


# 导包
from flask import Flask, request, jsonify
# request 是 Flask 提供的全局对象，包含了客户端发来的所有信息。
# jsonify: Flask 的工具函数，用于将 Python 字典或列表转换成标准的 JSON 格式发送给前端。
from rf_pred_fun import predict_fun

# todo 1. 创建App应用(对象)
app = Flask(__name__)

# todo 2. 创建预测接口(路由 + 预测函数)
# @app.route：这是一个装饰器，它将下面的函数与特定的网络地址关联起来。
# /WSent_predict：这是接口的 URL 路径。客户端需要访问 http://服务器IP:端口/predict才能触发这个函数。
# methods=['POST']：限制了访问方法只能是 POST。这意味着该接口用于接收客户端发送过来的数据，而不是像网页那样直接展示给用户看。
@app.route('/WSent_predict', methods=['POST'])
def predict():
    # 获取用户请求中的数据, 
    # request.get_json() 会把接收到的 JSON 数据自动转换成 Python 字典。
    data = request.get_json()
    print(f'data: {data}, {type(data)}')        # 字典
    # 调用预测函数
    result = predict_fun(data)
    # 返回结果, 
    # jsonify：Flask 的工具函数，用于将 Python 字典或列表转换成标准的 JSON 格式发送给前端。
    return jsonify(result)


# todo 3. 启动App应用
if __name__ == '__main__':
    # port指定端口号，默认5000
    # debug=True：开启调试模式。代码修改后会自动重启服务器，并且在报错时会显示详细的错误信息，方便开发, 生产环境需关闭
    # 127.0.0.1只能从本机访问,同一局域网内的其他电脑不能访问
    # 0.0.0.0可以从任何网络接口访问, 包括本机, 局域网内其他设备,公网
    # 192.168.109.56指定通过该地址访问
    # app.run(host='127.0.0.1', port=10088, debug=True)
    # app.run(host='0.0.0.0', port=10088, debug=True)
    app.run(host='192.168.12.21', port=10088, debug=True)