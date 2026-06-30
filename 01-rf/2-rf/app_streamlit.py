# 该py文件封装的是 给用户使用的前端页面.

# 导包
import streamlit as st
import requests
import time

# todo 1.streamlit创建画面
# 1. 设置标题
st.title('微博评论感情分类项目')
st.write('这是微博评论感情分类项目')

# 2. 获取用户输入的文本
text = st.text_input('请输入要查询分类的评论:')

# todo 2. 后台发送请求.
# 1. 准备URL
# url = 'http://127.0.0.1:10088/WSent_predict'
url = 'http://192.168.12.21:10088/WSent_predict'
if st.button('基于随机森林模型获取分类!'):
    # 2. 获取任务开始时间
    start_time = time.time()
    try:
        # 3. 发送请求, 获取数据.
        r = requests.post(url, json={'context': text})
        print(f'r: {r.json()}')
        # 4. 计算耗时.
        total_time = (time.time() - start_time) * 1000
        st.write('耗时: ', total_time, 'ms')
        # 5. 显示预测结果到网页.
        st.write('预测结果: ', r.json()['pred_class'])
    # except Exception as e:
    #     st.write(f'出问题了, {e}')
    except:
        print(f'出问题了, 请于管理员联系 110-123456!')
        # 把出错信息写到日志文件中, 方便后续排查.