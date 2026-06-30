@echo off
chcp 65001 >nul
echo ============================================================
echo    投满分 · 微博情绪分类系统 - 一键启动脚本
echo ============================================================
echo.
echo [提示] 请确保已在 conda 环境中运行
echo.
echo 服务访问地址:
echo   - Streamlit 前端: http://192.168.12.63:8501
echo   - Flask API:     http://192.168.12.63:80
echo.
echo ============================================================
echo.
echo [1/2] 正在启动 Flask API 服务...
start "Flask API Server" cmd /k "cd /d "%~dp0" && python api_flask_server.py"

timeout /t 3 /nobreak >nul

echo [2/2] 正在启动 Streamlit 前端...
start "Streamlit Frontend" cmd /k "cd /d "%~dp0" && python app_streamlit_run.py"

echo.
echo ✅ 所有服务已启动！
echo.
echo 请在浏览器中访问: http://192.168.12.63:8501
echo.
pause
