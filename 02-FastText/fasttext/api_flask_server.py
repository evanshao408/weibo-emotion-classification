from flask import Flask, request, jsonify
from ft_predict_fun import predict_fun

# 1、创建一个服务器
app = Flask(__name__)

# 2、如何找到服务器
@app.route('/NewCls_handler', methods=['POST'])
def NewCls_handler():
    # 将数据转换成字典类型
    data = request.get_json()
    # print(f'data: {data},{type(data)}')
    result = predict_fun(data)
    # 返回数据转为json格式
    return jsonify(result)


if __name__ == '__main__':
    app.run('192.168.12.116', 5000, debug=True)