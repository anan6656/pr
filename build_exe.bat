@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ============================================================
rem  PR Analyzer 一键打包脚本（内网/离线环境版）
rem  使用：把本文件与 pr2_gui.py 放在同一文件夹，双击本文件。
rem  说明：
rem   · 本机 Python 需已装好，脚本自动查找（venv → py → miniconda/anaconda
rem     → 常见目录 → PATH，跳过微软商店假 python）。
rem   · 脚本【不访问外网】：
rem       ① 若依赖已装 → 直接打包；
rem       ② 若缺依赖  → 用本机 pip（走公司内网源配置）安装；
rem          仍失败会打印手动命令，不会联网重试；
rem       ③ 纯离线   → 把 numpy / opencv-python / pyinstaller 的 .whl
rem          放进本文件夹下的 wheels 子文件夹，脚本会自动用它们安装。
rem   产物：dist\PR_Analyzer.exe（单文件绿色 exe）
rem ============================================================

echo.
echo ============================================================
echo   PR Analyzer 一键打包开始（内网模式）
echo   当前文件夹: %~dp0
echo ============================================================
echo.

rem ---------- 第 1 步：寻找 Python ----------
set "PY="

if exist "%~dp0venv\Scripts\python.exe" set "PY=%~dp0venv\Scripts\python.exe"

if not defined PY (
    py -3 --version >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)"') do set "PY=%%i"
    )
)

if not defined PY if exist "D:\code\miniconda\python.exe" set "PY=D:\code\miniconda\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\miniconda3\python.exe" set "PY=%LOCALAPPDATA%\miniconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "C:\ProgramData\miniconda3\python.exe" set "PY=C:\ProgramData\miniconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\python.exe" set "PY=%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "C:\ProgramData\Anaconda3\python.exe" set "PY=C:\ProgramData\Anaconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
if not defined PY if exist "C:\Python313\python.exe" set "PY=C:\Python313\python.exe"
if not defined PY if exist "C:\Python312\python.exe" set "PY=C:\Python312\python.exe"
if not defined PY if exist "C:\Python311\python.exe" set "PY=C:\Python311\python.exe"
if not defined PY if exist "C:\Python310\python.exe" set "PY=C:\Python310\python.exe"
if not defined PY if exist "C:\Python39\python.exe" set "PY=C:\Python39\python.exe"
if not defined PY if exist "C:\Program Files\Python313\python.exe" set "PY=C:\Program Files\Python313\python.exe"
if not defined PY if exist "C:\Program Files\Python312\python.exe" set "PY=C:\Program Files\Python312\python.exe"
if not defined PY if exist "C:\Program Files\Python311\python.exe" set "PY=C:\Program Files\Python311\python.exe"
if not defined PY if exist "C:\Program Files\Python310\python.exe" set "PY=C:\Program Files\Python310\python.exe"
if not defined PY if exist "C:\Program Files\Python39\python.exe" set "PY=C:\Program Files\Python39\python.exe"

rem 最后从 PATH 里找，但跳过微软商店的假 python（WindowsApps 里那个）
if not defined PY (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        if not defined PY (
            echo %%i | findstr /i "WindowsApps" >nul || set "PY=%%i"
        )
    )
)

if not defined PY (
    echo [错误] 没有找到 Python。
    echo        请让 IT 用公司软件源安装 Python 3.9+（或直接安装 conda），
    echo        装好后重新双击本文件即可。
    echo.
    pause
    exit /b 1
)

"%PY%" --version >nul 2>nul
if errorlevel 1 (
    echo [错误] 找到的 Python 无法运行：%PY%
    pause
    exit /b 1
)

echo [1/3] 使用 Python: %PY%
"%PY%" --version
echo.

rem ---------- 第 2 步：检查 / 安装依赖（内网，不访问外网） ----------
"%PY%" -c "import numpy, cv2, PyInstaller" >nul 2>nul
if not errorlevel 1 goto deps_ok

echo [2/3] 缺少依赖（numpy / opencv-python / pyinstaller），开始用 pip 安装 ...
echo.
echo       方案 A：同目录有 wheels 文件夹（纯离线）
if exist "%~dp0wheels" (
    echo         → 使用 %~dp0wheels 下的离线安装包
    "%PY%" -m pip install --disable-pip-version-check -q --no-index ^
        --find-links "%~dp0wheels" numpy opencv-python pyinstaller
    if errorlevel 1 (
        echo.
        echo       [错误] wheels 离线安装失败。
        echo       请确认 wheels 文件夹里包含与当前 Python 版本匹配的：
        echo           numpy / opencv-python / pyinstaller  对应的 .whl 文件
        echo       当前 Python: 
        "%PY%" --version
        echo.
        echo       可让 IT 从外网下载后拷贝进 wheels 文件夹。
        pause
        exit /b 1
    )
) else (
    echo         → 走本机 pip 的内网源（需公司已配置可用源）
    echo.
    "%PY%" -m pip install --disable-pip-version-check numpy opencv-python pyinstaller
    if errorlevel 1 (
        echo.
        echo       [错误] pip 安装失败（未连接外网导致属于正常现象）。
        echo       请选择以下任一方式解决：
        echo        1. 让 IT 配置公司内网 pip 源后，重新双击本文件；
        echo        2. 或手动安装后重跑：
        echo           "%PY%" -m pip install numpy opencv-python pyinstaller
        echo        3. 或在公司可联网的电脑下载 .whl 放入本文件夹的 wheels
        echo           子目录，再重新双击本文件（自动离线安装）。
        echo.
        pause
        exit /b 1
    )
)

"%PY%" -c "import numpy, cv2, PyInstaller" >nul 2>nul
if errorlevel 1 (
    echo [错误] 依赖仍不完整，请按上面的提示处理。
    pause
    exit /b 1
)

:deps_ok
echo [2/3] 依赖已齐全。
echo.

rem ---------- 第 3 步：PyInstaller 打包 ----------
echo [3/3] 开始打包（单文件 exe），约需 3~10 分钟，请勿关闭窗口 ...
echo.

set "EXE_NAME=PR_Analyzer"
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed ^
    --name %EXE_NAME% ^
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
echo   打包完成！
echo   产物: %~dp0dist\%EXE_NAME%.exe
echo   （单个绿色文件，可拷贝到任何 Windows 电脑直接双击运行）
echo   验证: 命令行运行  dist\%EXE_NAME%.exe --selftest
echo         生成 selftest_ok.txt，里面为 OK 即打包正常。
echo ============================================================
echo.

if exist "%~dp0dist" start "" explorer "%~dp0dist"
pause
endlocal
