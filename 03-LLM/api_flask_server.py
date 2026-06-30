# 该py文件的主要任务: 通过Flask组件, 构建 路由 + 预测函数的 应用.


# 导包
from flask import Flask, request, jsonify
from model2pred_Qwen import predict_fun

# todo 1. 创建App应用(对象)
app = Flask(__name__)


# todo 2. 创建预测接口(路由 + 预测函数)
@app.route('/predict', methods=['POST'])
def predict():
    # 获取用户请求中的数据
    data = request.get_json()
    # 调用预测函数
    re = predict_fun(data)
    # 返回结果
    return jsonify(re)


# todo 3. 启动App应用
if __name__ == '__main__':
    app.run("127.0.0.1", 5000, debug=True)
