@echo off
rem ============================================================
rem  PR 柱子识别工具 GUI —— 一键打包为单个 .exe
rem  前提：本机已安装 Python 3.9+，且已装好依赖
rem      pip install numpy opencv-python pyinstaller
rem  产物：dist\PR_Analyzer.exe （可拷贝到任何 Windows 电脑直接运行）
rem  注：生成一次即可；打包很慢请耐心等待进度条结束
rem ============================================================
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem 依次尝试可用的 Python
set "PY="
if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY set "PY=C:\Program Files\Python39\python.exe"

echo 使用 Python: %PY%
"%PY%" -c "import numpy, cv2, PyInstaller" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [错误] 缺少 numpy / opencv-python / pyinstaller，请先执行:
    echo   "%PY%" -m pip install --user numpy opencv-python pyinstaller
    pause
    exit /b 1
)

echo 开始打包，约需数分钟，请耐心等待...
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name PR_Analyzer ^
    --exclude-module matplotlib ^
    --exclude-module PIL ^
    pr2_gui.py
if errorlevel 1 (
    echo.
    echo [错误] 打包失败，请查看上方错误信息。
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  打包完成: %~dp0dist\PR_Analyzer.exe
echo  运行方式: 把 tif 图像放在一个文件夹里，打开程序选择该文件夹即可
echo ============================================================
pause
