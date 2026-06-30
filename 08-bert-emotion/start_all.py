# -*- coding: utf-8 -*-
"""
投满分 · 微博情绪分类系统 - 一键启动脚本
同时启动 Flask API 服务和 Streamlit 前端
"""

import os
import sys
import subprocess
import threading
import time

def print_banner():
    print("=" * 60)
    print("   投满分 · 微博情绪分类系统 - 一键启动")
    print("=" * 60)
    print()
    print("服务访问地址:")
    print("  - Streamlit 前端: http://192.168.12.63:8501")
    print("  - Flask API:     http://192.168.12.63:80")
    print()
    print("=" * 60)
    print()

def start_flask_api():
    """启动 Flask API 服务"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_path = os.path.join(current_dir, 'api_flask_server.py')
    print("[Flask API] 正在启动...")
    subprocess.run([sys.executable, api_path])

def start_streamlit():
    """启动 Streamlit 前端"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_path = os.path.join(current_dir, 'app_streamlit_run.py')
    print("[Streamlit] 正在启动...")
    subprocess.run([sys.executable, run_path])

if __name__ == '__main__':
    print_banner()
    
    # 在独立线程中启动 Flask API
    flask_thread = threading.Thread(target=start_flask_api, daemon=True)
    flask_thread.start()
    
    # 等待几秒钟让 API 先启动
    time.sleep(3)
    
    # 启动 Streamlit
    start_streamlit()
