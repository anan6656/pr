# -*- coding: utf-8 -*-
"""pr_6_main -- 主入口：离线打包（PyInstaller）与 GUI 启动。

从 pr6.1.py 拆分而来。职责：
1. ``python pr_6_main.py --build-exe`` 以当前解释器离线打包 GUI 程序；
   hiddenimports 显式列出全部 pr_* 模块（拆分后不再单文件）。
   v6.3: 源码先复制到本地临时目录再构建，规避 UNC 网络盘上
   PyInstaller 写数万临时文件导致的缓慢/失败；完成后拷回 dist。
2. ``python pr_6_main.py`` / ``python pr_6_main.py --selftest`` 启动 GUI 或自检。
"""

import os
import sys

from pr_1_config import APP_TITLE, APP_VERSION, EXE_NAME

APP_TITLE = APP_TITLE
EXE_NAME = EXE_NAME


def build_standalone_exe():
    """以当前解释器离线打包 GUI；不下载、不安装任何依赖。"""
    import importlib.util
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    required = {"numpy": "NumPy>=2", "cv2": "OpenCV", "tkinter": "Tkinter",
                "matplotlib": "Matplotlib", "PyInstaller": "PyInstaller"}
    missing = [label for module, label in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("无法离线打包，当前 Python 环境缺少：" + "、".join(missing))
        print("请先从离线安装介质安装依赖，或直接使用已打包的 EXE。程序不会联网安装。")
        return 1
    # numpy>=1.24 即可打包运行（cv2/matplotlib 兼容）；不强制 2.x。
    try:
        source = Path(__file__).resolve()
    except (OSError, PermissionError):
        # UNC 网络路径上 resolve 可能失败（WinError 59/5），退回普通绝对路径。
        source = Path(os.path.abspath(__file__))
    project_dir = source.parent
    output = project_dir / "dist"
    temp_dir = Path(tempfile.mkdtemp(prefix="pr_build_"))
    try:
        # v6.3: 在本地临时目录构建，规避 UNC 共享上 PyInstaller 的慢/失败。
        for py in sorted(project_dir.glob("pr_*.py")):
            shutil.copy2(py, temp_dir / py.name)
        local_source = temp_dir / source.name
        build_dir = temp_dir / "build"
        build_dir.mkdir(exist_ok=True)
        version_file = build_dir / "version_info.txt"
        # PyInstaller 要求 filevers/prodvers 恰好 4 段（如 6.3 → (6,3,0,0)）。
        parts = [int(v) for v in APP_VERSION.split(".")]
        version_tuple = tuple((parts + [0, 0, 0, 0])[:4])
        version_file.write_text(
            "VSVersionInfo(ffi=FixedFileInfo(filevers=" + repr(version_tuple) +
            ", prodvers=" + repr(version_tuple) + ", mask=0x3f, flags=0x0, OS=0x40004, "
            "fileType=0x1, subtype=0x0, date=(0, 0)), kids=[StringFileInfo([StringTable('040904B0', ["
            f"StringStruct('FileDescription', '{APP_TITLE}'), StringStruct('FileVersion', '{APP_VERSION}'), "
            f"StringStruct('ProductName', 'PR profile analyzer'), StringStruct('ProductVersion', '{APP_VERSION}')"
            "])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])", encoding="utf-8")
        local_dist = temp_dir / "dist"
        command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
                   "--noupx", "--name", EXE_NAME, "--version-file", str(version_file),
                   "--distpath", str(local_dist), "--workpath", str(build_dir),
                   "--specpath", str(build_dir)]
        for module in ("scipy", "pandas", "IPython", "pytest", "torch", "PyQt5", "PyQt6", "PySide6"):
            command.extend(["--exclude-module", module])
        # 拆分后为多模块：显式把 pr_* 收进 onefile。
        for module in ("pr_1_config", "pr_2_image_recognition", "pr_3_contour_calc",
                       "pr_4_contour_compare", "pr_5_gui"):
            command.extend(["--hidden-import", module])
        print("仅使用当前 PY 文件离线打包（本地临时目录），输出：" + str(output), flush=True)
        result = subprocess.run(command + [str(local_source)], cwd=str(temp_dir), check=False)
        if result.returncode == 0:
            exe_file = local_dist / (EXE_NAME + ".exe")
            if not exe_file.is_file():
                print("打包未生成预期 EXE：" + str(exe_file))
                return 1
            output.mkdir(parents=True, exist_ok=True)
            final_exe = output / exe_file.name
            shutil.copy2(exe_file, final_exe)
            print("打包完成：" + str(final_exe))
        return result.returncode
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    if "--build-exe" in sys.argv[1:] and not getattr(sys, "frozen", False):
        raise SystemExit(build_standalone_exe())

    from pr_5_gui import gui_main
    gui_main()


if __name__ == "__main__":
    main()
