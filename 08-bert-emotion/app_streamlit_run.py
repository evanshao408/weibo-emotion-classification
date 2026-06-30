# 该.py文件用于启动综合预测页面

import os
import sys

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, 'app_streamlit.py')

print("=" * 60)
print("🎯 投满分 · 微博情绪分类系统 Streamlit 服务")
print("=" * 60)
print("\n服务访问地址:")
print("  - 本地访问: http://127.0.0.1:8501")
print("  - 局域网访问:  ")
print("=" * 60)
print("\n正在启动...\n")

# 启动Streamlit - 监听所有网络接口
os.system(f'streamlit run "{app_path}" --server.address 0.0.0.0 --server.port 8501')

