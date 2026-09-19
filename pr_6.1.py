# -*- coding: utf-8 -*-
"""PR Analyzer v6.0.0 — binary 粗定位、受约束动态规划及亚像素灰度边缘测量。
运行: python pr_6.py
离线打包: python pr_6.py --build-exe
打包环境须预装 Python、NumPy>=2、OpenCV、Tkinter、Matplotlib、PyInstaller。
只需要本文件，不依赖外部 spec、说明文档或模型文件；运行时无需联网。
算法及统计说明已内置于帮助菜单。
"""

import os
import sys

APP_VERSION = "6.0.0"
APP_TITLE = f"PR Analyzer v{APP_VERSION}"
EXE_NAME = f"PR_Analyzer_v{APP_VERSION}"


def application_dir():
    return os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))


def build_standalone_exe():
    """以当前解释器离线打包此单文件；不下载、不安装任何依赖。"""
    import importlib.util
    import shutil
    import subprocess
    from pathlib import Path

    required = {"numpy": "NumPy>=2", "cv2": "OpenCV", "tkinter": "Tkinter",
                "matplotlib": "Matplotlib", "PyInstaller": "PyInstaller"}
    missing = [label for module, label in required.items() if importlib.util.find_spec(module) is None]
    if missing:
        print("无法离线打包，当前 Python 环境缺少：" + "、".join(missing))
        print("请先从离线安装介质安装依赖，或直接使用已打包的 EXE。程序不会联网安装。")
        return 1
    import numpy
    if int(numpy.__version__.split(".")[0]) < 2:
        print("无法打包：需要 NumPy 2.0 或以上版本。")
        return 1
    source = Path(__file__).resolve()
    output = source.parent / "dist"
    build_dir = source.parent / "build"
    build_dir.mkdir(exist_ok=True)
    version_file = build_dir / "version_info.txt"
    version_tuple = tuple(int(v) for v in APP_VERSION.split(".")) + (0,)
    version_file.write_text(
        "VSVersionInfo(ffi=FixedFileInfo(filevers=" + repr(version_tuple) +
        ", prodvers=" + repr(version_tuple) + ", mask=0x3f, flags=0x0, OS=0x40004, "
        "fileType=0x1, subtype=0x0, date=(0, 0)), kids=[StringFileInfo([StringTable('040904B0', ["
        f"StringStruct('FileDescription', '{APP_TITLE}'), StringStruct('FileVersion', '{APP_VERSION}'), "
        f"StringStruct('ProductName', 'PR Analyzer'), StringStruct('ProductVersion', '{APP_VERSION}')"
        "])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])", encoding="utf-8")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
               "--noupx", "--optimize", "1", "--name", EXE_NAME, "--version-file", str(version_file),
               "--distpath", str(output), "--workpath", str(source.parent / "build"),
               "--specpath", str(source.parent / "build")]
    for module in ("scipy", "pandas", "IPython", "pytest", "torch", "PyQt5", "PyQt6", "PySide6"):
        command.extend(["--exclude-module", module])
    print("仅使用当前 PY 文件离线打包，输出：" + str(output), flush=True)
    result = subprocess.run(command + [str(source)], cwd=str(source.parent), check=False)
    if result.returncode == 0:
        # 打包完成后删除临时 spec / build，只留下可携带的 EXE。
        shutil.rmtree(build_dir, ignore_errors=True)
        print("打包完成：" + str(output / (EXE_NAME + ".exe")))
    return result.returncode


if __name__ == "__main__" and "--build-exe" in sys.argv[1:] and not getattr(sys, "frozen", False):
    raise SystemExit(build_standalone_exe())

# ---------------------------------------------------------------------
# 限制科学计算库线程数：避免 GUI / 打包环境中过度抢占 CPU。
# 注意：这不是 NumPy X86_V2 CPU 指令集兼容补丁。
# ---------------------------------------------------------------------
for _env_name in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_env_name, "1")

import glob
import math
import queue
import shutil
import threading
import traceback
import contextlib
import csv
import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Optional

import numpy as np
import cv2

# matplotlib 仅用于直线量测灰度剖面
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    # 让曲线标题能正常显示中文（缺少中文字体时 matplotlib 会画成方框）
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei UI", "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


# =====================================================================
# 全局参数
# =====================================================================

INPUT_DIR = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
INPUT_GLOBS = ("*.tif", "*.tiff")
OUTPUT_SUBDIR = True
DEBUG_OUTPUT = True
# 仅控制 ROI 过程图；自动识别始终输出到 test 文件夹。
PILLAR_POLARITY = "bright"  # bright / dark / auto；SEM 默认亮柱，禁止自动翻转为 trench
REFINE_RADIUS_PX = 32

# ---- binary 失败时的几何提议后备参数 ----
GEOMETRIC_ROW_START_RATIO = 0.22
GEOMETRIC_ROW_END_RATIO = 0.37
GEOMETRIC_MIN_RUN_PX = 5
GEOMETRIC_PROFILE_SMOOTH = 11
GEOMETRIC_EDGE_SMOOTH = 5
GEOMETRIC_TRACE_STEP_RATIO = 0.12
GEOMETRIC_TRACE_DISTANCE_PENALTY = 0.18
GEOMETRIC_MIN_EDGE_RESPONSE = 1.5

# ---- 用户可理解的自动识别先验 ----
# 低层阈值、平滑核和追踪权重由程序自动管理。用户只需要给出一个
# 大致的 Line CD（像素）和允许误差；填 0 表示完全自动估计。这个范围
# 作为“软先验”参与候选评分，同时用于过滤明显不可能的窄条/宽块，
# 不会把图像边缘被裁掉的残柱直接删掉。
AUTO_CD_ESTIMATE_PX = 0.0
AUTO_CD_TOLERANCE_PX = 12.0

# ---- 柱子有效性 ----
MIN_COLUMN_HEIGHT = 20
MIN_AVG_WIDTH = 15
MIN_ASPECT_RATIO = 1.2
MAX_TRACE_GAP = 8

# ---- 比例尺 ----
SCALE_PIXELS = 100.0
SCALE_NM = 50.0

IMAGE_SCALE_FILE = "pr_image_scales.json"

# ---- 绘图 ----
ALPHA = 0.15
LINE_THICKNESS = 1
AXIS_MARGIN_L = 42
AXIS_MARGIN_B = 24
PANEL_W = 100

# OpenCV 是 BGR
PR_COLORS = [
    (0, 0, 255),      # 红
    (0, 180, 0),      # 绿
    (255, 0, 0),      # 蓝
    (255, 0, 255),    # 品红
    (255, 180, 0),    # 青蓝
    (0, 165, 255),    # 橙
]


# =====================================================================
# 基础工具
# =====================================================================

def ensure_odd(value, minimum=1, maximum=None):
    k = max(minimum, int(round(value)))
    if maximum is not None:
        k = min(k, int(maximum))
    if k % 2 == 0:
        k += 1
        if maximum is not None and k > maximum:
            k -= 2
    return max(1, k)


def smooth_1d(data, kernel_size):
    """真正的一维高斯平滑。内部统一 reshape 为 1×N，核使用 (K,1)。"""
    arr = np.asarray(data, dtype=np.float32).reshape(1, -1)
    if arr.shape[1] < 2:
        return arr.flatten()
    k = ensure_odd(kernel_size, 1, max(1, arr.shape[1] if arr.shape[1] % 2 else arr.shape[1] - 1))
    if k <= 1:
        return arr.flatten()
    return cv2.GaussianBlur(arr, (k, 1), 0).flatten()


def normalize_01(data, percentile=99.0):
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0:
        return arr
    hi = float(np.percentile(arr, percentile))
    if not math.isfinite(hi) or hi <= 1e-12:
        hi = float(arr.max())
    if hi <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / hi, 0.0, 1.0).astype(np.float32)


def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """兼容 Windows 中文路径。"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size:
            img = cv2.imdecode(buf, flags)
            if img is not None:
                return img
    except Exception:
        pass
    return cv2.imread(path, flags)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, encoded = cv2.imencode(ext, img)
        if ok:
            encoded.tofile(path)
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img)


def to_gray8(img_raw):
    if img_raw is None:
        raise ValueError("空图像")

    if img_raw.ndim == 3:
        if img_raw.shape[2] == 4:
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2GRAY)
        elif img_raw.shape[2] == 3:
            gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_raw[..., 0]
    else:
        gray = img_raw

    if gray.dtype == np.uint8:
        return gray.copy()

    arr = gray.astype(np.float32)
    lo, hi = np.percentile(arr, (1, 99))
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())

    if hi <= lo:
        return np.zeros(gray.shape, dtype=np.uint8)

    out = (arr - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def calculate_image_quality(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    hist = np.histogram(gray.ravel(), bins=256, range=(0, 256))[0]
    spread = int(np.count_nonzero(hist))

    return {
        "edge_strength": float(np.mean(grad)),
        "contrast": float(np.std(gray)),
        "histogram_spread": spread,
    }


def _binary_fraction(binary):
    return float(np.count_nonzero(binary)) / float(binary.size or 1)


# =====================================================================
# 预处理：亮度 + 灰度斜率
# =====================================================================

def preprocess_image(image_path, on_stage=None):
    """保边降噪先于 CLAHE；测量保留原始灰度，增强图只用于粗定位。"""
    raw = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"无法读取图片：{image_path}")
    gray = to_gray8(raw)
    if on_stage is not None:
        on_stage("step1_gray.png", gray)
    filtered = cv2.bilateralFilter(cv2.medianBlur(gray, 3), 5, 20, 7)
    if on_stage is not None:
        on_stage("step2_filtered.png", filtered)
    enhanced = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(filtered)
    if on_stage is not None:
        on_stage("step3_enhanced.png", enhanced)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    gx = normalize_01(np.abs(cv2.Scharr(filtered, cv2.CV_32F, 1, 0)))
    gy = normalize_01(np.abs(cv2.Scharr(filtered, cv2.CV_32F, 0, 1)))
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    aux = {"gray": gray, "filtered": filtered, "enhanced": enhanced, "blur": blur,
           "grad_x": gx, "grad_y": gy, "quality": calculate_image_quality(filtered),
           "threshold_mode": "otsu_candidates", "binary_fraction": _binary_fraction(binary)}
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), binary, aux


def _run_segments(mask, min_len=1):
    """返回布尔数组中连续 True 区间，区间为闭区间。"""
    out = []
    start = None
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= int(min_len):
                out.append((int(start), int(i - 1)))
            start = None
    if start is not None and len(mask) - start >= int(min_len):
        out.append((int(start), int(len(mask) - 1)))
    return out


def _main_row_activity(gray, prefer_upper=True):
    """估计柱体主体的 y 范围。

    这里不依赖固定灰度阈值。柱体出现时，整行的 robust contrast（90%-10%
    分位差）和横向边缘能量会同时升高；底座、标尺和文字通常只形成很短的
    次级区间，因此选最长主体段，并用更高的末端阈值截掉底部融合区。
    """
    g = np.asarray(gray, dtype=np.uint8)
    h, w = g.shape[:2]
    if h < 4 or w < 4:
        return 0, max(0, h - 1), np.zeros(h, dtype=np.float32)

    # 先去掉每一行的一阶照明趋势。SEM 图常有从左到右的渐变，若直接
    # 用 q90-q10，会把整幅背景的斜坡误当成柱体；线性去趋势后只保留
    # 局部柱/沟结构，对曝光不均和轻微 vignette 更稳。
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    x2 = float(np.dot(x, x)) + 1e-6
    row_mean = np.mean(g.astype(np.float32), axis=1)
    slope = np.sum((g.astype(np.float32) - row_mean[:, None]) * x[None, :], axis=1) / x2
    residual = g.astype(np.float32) - (row_mean[:, None] + slope[:, None] * x[None, :])
    q10 = np.percentile(residual, 10, axis=1).astype(np.float32)
    q90 = np.percentile(residual, 90, axis=1).astype(np.float32)
    contrast = smooth_1d(q90 - q10, max(7, GEOMETRIC_PROFILE_SMOOTH))

    blur = cv2.GaussianBlur(g, (5, 5), 0)
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    edge = smooth_1d(np.percentile(gx, 85, axis=1), max(7, GEOMETRIC_PROFILE_SMOOTH))

    c_hi = max(float(np.percentile(contrast, 95)), 1e-6)
    e_hi = max(float(np.percentile(edge, 95)), 1e-6)
    activity = smooth_1d(
        0.72 * np.clip(contrast / c_hi, 0.0, 2.0)
        + 0.28 * np.clip(edge / e_hi, 0.0, 2.0),
        max(7, GEOMETRIC_PROFILE_SMOOTH),
    )

    peak = float(np.max(activity)) if activity.size else 0.0
    if peak <= 1e-8:
        return 0, max(0, h - 1), activity

    # 行活动本身也做一次 Otsu 分割。固定 0.22 在高噪声/均匀底图上会
    # 把背景噪声接成整幅主体；Otsu 只作为更高的自适应门槛，弱对比图仍
    # 保留 GEOMETRIC_ROW_START_RATIO 这个下限。
    activity_scaled = np.clip(np.round(activity / peak * 255.0), 0, 255).astype(np.uint8)
    otsu_activity, _ = cv2.threshold(
        activity_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_ratio = float(otsu_activity) / 255.0
    start_ratio = max(
        float(np.clip(GEOMETRIC_ROW_START_RATIO, 0.08, 0.70)),
        float(np.clip(0.75 * otsu_ratio, 0.12, 0.65)),
    )

    min_len = max(12, int(round(h * 0.04)))
    runs = _run_segments(
        activity >= peak * start_ratio,
        min_len=min_len,
    )
    if not runs:
        return 0, max(0, h - 1), activity

    # 优先选靠近图像上半区的长主体，避免底部刻度/文字成为主段。
    # PR profile 主体位于图像上半部：候选限制在 0.55h 以内开始；
    # 上半部没有足够长的活动段时才回退到全图最长段。
    half_lim = int(h * 0.55)
    upper = [r for r in runs if r[0] < half_lim] if prefer_upper else runs
    if upper:
        min_main = max(8, int(round(h * 0.05)))
        good_upper = [r for r in upper if (r[1] - r[0] + 1) >= min_main] or upper
        main = max(good_upper, key=lambda p: (p[1] - p[0] + 1, -p[0]))
    else:
        # 全部活动段都在下半部（如 A55 底部颗粒纹理区活动度远超
        # 弱对比柱区）：检查上半部是否存在活动度达峰值一定比例的
        # 候选主体段，有则优先采用（PR profile 主体在图像上半部）。
        upper_thr = peak * 0.30
        upper_runs = _run_segments(
            activity[:half_lim] >= upper_thr,
            min_len=max(12, int(round(h * 0.06))),
        )
        if upper_runs:
            main = max(upper_runs, key=lambda p: (p[1] - p[0] + 1, -p[0]))
        else:
            candidates = [r for r in runs if r[0] < int(h * 0.70)] or runs
            main = max(candidates, key=lambda p: (p[1] - p[0] + 1, -p[0]))
    top = int(main[0])

    # 主体底部在柱脚与基座融合处会快速失去横向周期结构。
    # end_thr 不得超过活动度的稳健水平（中位数的 1.15 倍）：如
    # A55 底部颗粒纹理把全局峰值抬得很高（1.9 vs 柱区 0.85），
    # 固定比例阈值（0.37×peak）会把柱体下半段（活动度 0.65~0.69）
    # 误判为融合区截掉，导致柱脚丢失。
    end_thr = peak * float(np.clip(GEOMETRIC_ROW_END_RATIO, 0.12, 0.90))
    robust_floor = float(np.median(activity[activity > 0])) * 1.15 \
        if np.any(activity > 0) else 0.0
    end_thr = min(end_thr, max(robust_floor, peak * 0.12))
    in_main = np.arange(top, int(main[1]) + 1, dtype=np.int32)
    good_end = in_main[activity[in_main] >= end_thr]
    bottom = int(good_end[-1]) if good_end.size else int(main[1])
    if bottom <= top:
        bottom = int(main[1])

    return top, bottom, activity.astype(np.float32)


def _geometric_x_profile(gray, y_top, y_bottom):
    """由逐行 robust 归一化亮度的中位数建立柱实体投影。"""
    g = np.asarray(gray, dtype=np.uint8)
    h, w = g.shape[:2]
    y0 = max(0, int(y_top) + max(2, int(round((y_bottom - y_top + 1) * 0.035))))
    y1 = min(h - 1, int(y_bottom) - max(2, int(round((y_bottom - y_top + 1) * 0.035))))
    if y1 <= y0:
        y0, y1 = int(y_top), int(y_bottom)

    rows = g[y0:y1 + 1].astype(np.float32)
    # 与 y 活动估计一致，去除每一行的一阶照明趋势；否则左右渐变会让
    # 同一根柱的两侧平台落到不同的归一化阈值，造成假分柱。
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    x2 = float(np.dot(x, x)) + 1e-6
    row_mean = np.mean(rows, axis=1)
    slope = np.sum((rows - row_mean[:, None]) * x[None, :], axis=1) / x2
    rows = rows - (row_mean[:, None] + slope[:, None] * x[None, :])
    q10 = np.percentile(rows, 10, axis=1)
    q90 = np.percentile(rows, 90, axis=1)
    den = np.maximum(q90 - q10, 2.0)
    normalized = np.clip((rows - q10[:, None]) / den[:, None], -1.0, 2.0)
    profile = np.median(normalized, axis=0).astype(np.float32)
    profile = smooth_1d(profile, max(5, GEOMETRIC_PROFILE_SMOOTH))

    if profile.size:
        # Otsu 在“背景低平台 / 柱体高平台”之间选分界，避免固定比例
        # 因图像亮度、对比度或柱宽占比变化而失效。
        scaled = np.clip(np.round(profile * 255.0), 0, 255).astype(np.uint8)
        otsu_t, _ = cv2.threshold(
            scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = float(np.clip(float(otsu_t) / 255.0, 0.12, 0.72))
    else:
        threshold = 0.35

    runs = _run_segments(profile >= threshold, min_len=GEOMETRIC_MIN_RUN_PX)
    if not runs:
        return profile, [], threshold

    # 高亮纹理可能让一个实体柱在投影上出现几个很近的短缺口。
    # 先用很保守的邻近合并消除它们，真正的柱间距通常远大于柱内缺口。
    widths = np.array([e - s + 1 for s, e in runs], dtype=np.float32)
    typical = float(np.median(widths)) if widths.size else 5.0
    merge_gap = max(4, int(round(0.20 * typical)))
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])

    # 极短段仍可能来自文字/噪声；保留贴边残柱，但过滤远小于主体宽度的段。
    widths = np.array([e - s + 1 for s, e in merged], dtype=np.float32)
    ref_width = float(np.median(widths[widths >= max(5.0, np.percentile(widths, 25))])) \
        if widths.size else 0.0
    if ref_width > 0:
        min_keep = max(float(GEOMETRIC_MIN_RUN_PX), ref_width * 0.12)
        merged = [
            (s, e) for s, e in merged
            if (e - s + 1) >= min_keep or s <= 1 or e >= w - 2
        ]

    return profile, merged, threshold


def _smooth_row_and_gradient(gray):
    """返回逐行横向平滑灰度和 signed dI/dx。"""
    g = np.asarray(gray, dtype=np.uint8)
    k = ensure_odd(GEOMETRIC_EDGE_SMOOTH, 3, 15)
    smoothed = np.empty_like(g, dtype=np.float32)
    for y in range(g.shape[0]):
        smoothed[y] = cv2.GaussianBlur(g[y:y + 1], (k, 1), 0).reshape(-1)
    return smoothed, np.gradient(smoothed, axis=1).astype(np.float32)


def _choose_tracked_edge(gradient_row, previous, previous2, lo, hi, side):
    """在连续窗口内选最可信的左右侧壁位置。

    纯 argmax 容易跳到柱内纹理；这里用有符号梯度 + 轨迹连续性联合评分。
    ``side=+1`` 寻找左壁的正梯度，``side=-1`` 寻找右壁的负梯度。
    """
    lo, hi = int(min(lo, hi)), int(max(lo, hi))
    if hi < lo:
        return int(previous)
    x = np.arange(lo, hi + 1, dtype=np.int32)
    response = side * np.asarray(gradient_row[lo:hi + 1], dtype=np.float32)
    predicted = float(previous)
    if previous2 is not None:
        # 允许跟随真实侧壁缓慢倾斜，但限制二阶预测，避免低信噪比行
        # 把轨迹逐步推到 cell 边界。
        delta = float(np.clip(previous - previous2, -2.0, 2.0))
        predicted = float(previous) + delta
        predicted = float(np.clip(predicted, lo, hi))
    distance = np.abs(x.astype(np.float32) - predicted)
    # 以 robust 梯度尺度归一化，使不同 y 的权重可比。
    scale = float(np.percentile(np.abs(response), 75)) if response.size else 1.0
    scale = max(scale, 1.0)
    score = response / scale - GEOMETRIC_TRACE_DISTANCE_PENALTY * distance
    best = int(x[int(np.argmax(score))])

    # 没有足够的边缘证据时，宁可沿用平滑轨迹，也不要跳到纹理边缘。
    local_response = float(response[best - lo]) if lo <= best <= hi else 0.0
    if local_response < GEOMETRIC_MIN_EDGE_RESPONSE and previous is not None:
        return int(round(predicted))
    return best


def _track_geometric_column(gray_smooth, gradient, segment, cell, seed_y,
                            y_top, y_bottom):
    """从强边缘行向上下双向追踪一根柱的左右边界。"""
    h, w = gray_smooth.shape[:2]
    seg_l, seg_r = [int(v) for v in segment]
    cell_l, cell_r = [int(v) for v in cell]
    center = int(round((seg_l + seg_r) * 0.5))
    visible_left = seg_l > 1
    visible_right = seg_r < w - 2

    # 种子行在柱区侧壁附近寻找最强有符号梯度；边缘残柱的不可见侧固定在图边。
    seed_row = gradient[int(np.clip(seed_y, 0, h - 1))]
    width_hint = max(8, seg_r - seg_l + 1)
    seed_window = max(12, int(round(width_hint * 0.32)))
    if visible_left:
        l0 = max(cell_l, seg_l - seed_window)
        l1 = min(center, seg_l + seed_window)
        left = _choose_tracked_edge(seed_row, seg_l, None, l0, l1, +1)
    else:
        left = 0
    if visible_right:
        r0 = max(center, seg_r - seed_window)
        r1 = min(cell_r, seg_r + seed_window)
        right = _choose_tracked_edge(seed_row, seg_r, None, r0, r1, -1)
    else:
        right = w - 1

    rows = {int(seed_y): (int(left), int(right))}
    step = max(4, int(round(width_hint * float(GEOMETRIC_TRACE_STEP_RATIO))))

    for direction in (-1, 1):
        prev_l, prev_r = left, right
        prev2_l = prev2_r = None
        start = int(seed_y) + direction
        stop = int(y_top) - 1 if direction < 0 else int(y_bottom) + 1
        for y in range(start, stop, direction):
            if visible_left:
                l_lo = max(cell_l, prev_l - step)
                l_hi = min(center, prev_l + step)
                cur_l = _choose_tracked_edge(
                    gradient[y], prev_l, prev2_l, l_lo, l_hi, +1
                )
            else:
                cur_l = 0

            if visible_right:
                r_lo = max(center, prev_r - step)
                r_hi = min(cell_r, prev_r + step)
                cur_r = _choose_tracked_edge(
                    gradient[y], prev_r, prev2_r, r_lo, r_hi, -1
                )
            else:
                cur_r = w - 1

            if cur_r < cur_l:
                cur_l, cur_r = min(cur_l, cur_r), max(cur_l, cur_r)
            rows[int(y)] = (int(cur_l), int(cur_r))
            prev2_l, prev2_r = prev_l, prev_r
            prev_l, prev_r = cur_l, cur_r

    ordered = []
    for y in range(int(y_top), int(y_bottom) + 1):
        if y in rows:
            ordered.append((int(y), int(rows[y][0]), int(rows[y][1])))

    if len(ordered) >= 5:
        # 轻微高斯平滑仅压制 1~2 px 的纹理抖动，不改变边缘残柱的固定侧。
        arr_l = smooth_1d(np.array([r[1] for r in ordered], dtype=np.float32), 5)
        arr_r = smooth_1d(np.array([r[2] for r in ordered], dtype=np.float32), 5)
        fixed_l = not visible_left
        fixed_r = not visible_right
        smoothed = []
        for i, (y, l, r) in enumerate(ordered):
            nl = 0 if fixed_l else int(round(arr_l[i]))
            nr = w - 1 if fixed_r else int(round(arr_r[i]))
            if nr < nl:
                nl, nr = min(nl, nr), max(nl, nr)
            smoothed.append((y, nl, nr))
        ordered = smoothed

    return ordered


def _auto_cd_range():
    """返回用户给出的 Line CD 软先验 ``(center, tolerance)``。

    ``0`` 是有意保留的默认值：程序会从识别候选自身估计典型宽度，
    用户不需要先猜一个可能完全不适用的数值。
    """
    try:
        center = float(AUTO_CD_ESTIMATE_PX)
        tolerance = float(AUTO_CD_TOLERANCE_PX)
    except Exception:
        return None
    if not math.isfinite(center) or center <= 0:
        return None
    if not math.isfinite(tolerance) or tolerance <= 0:
        tolerance = max(2.0, center * 0.15)
    return float(center), float(max(1.0, tolerance))


def _auto_cd_tolerance(default_center=None):
    """读取允许误差，即使 Line CD 估计值设为 0 也保持其有效性。"""
    try:
        tolerance = float(AUTO_CD_TOLERANCE_PX)
    except Exception:
        tolerance = float(default_center or 0.0) * 0.18
    if not math.isfinite(tolerance) or tolerance <= 0:
        tolerance = float(default_center or 0.0) * 0.18
    return float(max(1.0, tolerance)) if tolerance > 0 else 1.0


def _column_width_px(column):
    """用逐行轮廓的中位边界距离衡量物理 Line CD。

    ``ranges`` 保存的是左右边界坐标而不是像素格子；因此物理宽度与
    手动量测、统计 CD 统一采用 ``x_end - x_start``。内部几何过滤仍可
    使用含端点的像素格数量，但不会再混入用户可见的 Line CD 先验。
    """
    ranges = column.get("ranges") or []
    widths = np.asarray(
        [float(e) - float(s) for _, s, e in ranges],
        dtype=np.float32,
    )
    widths = widths[np.isfinite(widths) & (widths > 0)]
    if widths.size:
        return float(np.median(widths))
    return float(max(0, int(column.get("right", 0)) -
                    int(column.get("left", 0))))


def _apply_cd_hint(columns, image_width):
    """按用户宽度先验软过滤明显不可能的候选。

    边缘残柱天然比完整柱窄，因此边界柱只在极端窄小/过宽时才剔除。
    若过滤后候选太少，会保留原候选，确保错误先验不会让检测完全失效。
    """
    hint = _auto_cd_range()
    if hint is None or not columns:
        return columns

    center, tolerance = hint
    # 两倍容差是保护带；真正评分仍使用一倍容差。
    lo = max(3.0, center - 2.0 * tolerance)
    hi = center + 2.0 * tolerance
    kept = []
    for col in columns:
        width = _column_width_px(col)
        edge_clipped = (
            int(col.get("left", 0)) <= 1 or
            int(col.get("right", image_width - 1)) >= image_width - 2
        )
        if edge_clipped:
            # 残柱至少应保留主体的一小部分；上限放宽，允许轻微粘连。
            ok = width >= max(3.0, center * 0.12) and width <= hi * 1.55
        else:
            ok = lo <= width <= hi
            # 宽度稍微超出保护带时保留，交给置信度排序决定。
            if not ok and width >= center * 0.55 and width <= hi * 1.45:
                ok = True
        if ok:
            kept.append(col)

    # 保护至少一半候选，避免用户输入了不匹配的估计值后“一根不剩”。
    if len(kept) < max(1, int(math.ceil(len(columns) * 0.5))):
        return columns
    return kept


def _detection_confidence(columns, gray=None):
    """给候选分柱结果一个可解释的排序分数（越高越可信）。"""
    if not columns:
        return -1e9

    g = to_gray8(gray) if gray is not None else None
    h, w = g.shape[:2] if g is not None else (1, 1)
    widths = np.asarray([_column_width_px(c) for c in columns], dtype=np.float32)
    widths = widths[np.isfinite(widths) & (widths > 0)]
    if widths.size == 0:
        return -1e9

    # 完整柱比边缘残柱更适合估计“典型 CD”；优先使用非贴边候选。
    body = [
        _column_width_px(c) for c in columns
        if int(c.get("left", 0)) > 1 and int(c.get("right", w - 1)) < w - 2
    ]
    fit_widths = np.asarray(body if len(body) >= 2 else widths, dtype=np.float32)
    med = float(np.median(fit_widths))
    mad = float(np.median(np.abs(fit_widths - med)))
    # MAD 在“多数宽度相同 + 一个粘连/过宽候选”的情况下可能恰好为 0，
    # 例如 [83, 107, 83]；仅看 MAD 会错误地把该轮当成完全一致。加入
    # 标准差项，仍保留 MAD 对少量纹理抖动的鲁棒性。
    mad_term = mad / max(2.0, med * 0.22)
    spread_term = float(np.std(fit_widths)) / max(2.0, med * 0.18)
    consistency = float(np.exp(-max(mad_term, spread_term)))

    hint = _auto_cd_range()
    if hint is None:
        width_fit = 1.0
    else:
        center, tolerance = hint
        # 对完整柱使用中心宽度；边缘残柱不把分数拉得过低。
        deviations = np.abs(fit_widths - center)
        width_fit = float(np.mean(np.exp(-deviations / max(1.0, tolerance))))

    heights = np.asarray([
        int(c.get("y_bottom", 0)) - int(c.get("y_top", 0)) + 1
        for c in columns
    ], dtype=np.float32)
    heights = heights[np.isfinite(heights) & (heights > 0)]
    # 不要用 ``h*0.18`` 作为分母：ROI 中一个短截候选和覆盖完整主体的
    # 候选都会很快饱和到 1，导致 early-stop/候选排序无法偏好完整柱体。
    # 用图像高度归一化，保留“覆盖范围越完整，分数越高”的单调信息。
    height_fit = float(np.clip(np.median(heights) / max(1.0, float(h)), 0.0, 1.0))

    centers = np.asarray([
        float(c.get("center_x", (c.get("left", 0) + c.get("right", 0)) * 0.5))
        for c in columns
    ], dtype=np.float32)
    centers.sort()
    if centers.size >= 2:
        gaps = np.diff(centers)
        gap_fit = float(np.clip(
            np.median(gaps) / max(3.0, med * 0.80), 0.0, 1.0
        ))
    else:
        gap_fit = 0.15

    # 贴边候选通常只是残柱，或是反相后把左右背景当成了“柱”。适度
    # 扣分可避免单根真实柱被两个巨大边缘背景块替代，同时仍保留
    # pr3 这类确实包含左右残柱的结果。
    edge_count = sum(
        1 for c in columns
        if int(c.get("left", 0)) <= 1
        or int(c.get("right", w - 1)) >= w - 2
    )
    body_count = max(0, len(columns) - edge_count)
    edge_penalty = 0.25 * float(edge_count)
    if body_count == 0 and len(columns) > 0:
        edge_penalty += 1.5

    # 侧壁梯度证据：低对比度候选仍可得分，但纯纹理窄条通常明显偏低。
    # 用柱体内部 vs 沟心灰度对比替代纯梯度：真实柱体（亮于沟）
    # 得正证据，框含背景/颗粒纹理的过宽候选（柱内即基底）对比
    # 接近零，被显著压分（如 A689079 equalized 候选框宽 67px，
    # 真实柱仅 48~57px）。
    contrast_fit = 0.0
    if g is not None and g.size and len(columns) >= 2:
        cols_sorted = sorted(
            columns, key=lambda c: float(c.get("center_x", 0)))
        cv_vals = []
        for c in cols_sorted:
            l = int(np.clip(c.get("left", 0), 0, w - 1))
            r = int(np.clip(c.get("right", w - 1), 0, w - 1))
            wt = max(2, int(round(0.25 * (r - l + 1))))
            cx0 = int(np.clip(0.5 * (l + r) - wt, 0, w - 1))
            cx1 = int(np.clip(0.5 * (l + r) + wt + 1, 1, w))
            y0 = int(np.clip(c.get("y_top", 0), 0, h - 1))
            y1 = int(np.clip(c.get("y_bottom", h - 1), 0, h - 1))
            if y1 > y0:
                cv_vals.append(float(np.median(g[y0:y1 + 1, cx0:cx1])))
        tv_vals = []
        for a, b in zip(cols_sorted[:-1], cols_sorted[1:]):
            ra = int(np.clip(a.get("right", 0), 0, w - 1))
            lb = int(np.clip(b.get("left", 0), 0, w - 1))
            mid = int(np.clip(0.5 * (ra + lb), 0, w - 1))
            y0 = int(np.clip(max(a.get("y_top", 0), b.get("y_top", 0)), 0, h - 1))
            y1 = int(np.clip(min(a.get("y_bottom", h - 1), b.get("y_bottom", h - 1)), 0, h - 1))
            if y1 > y0:
                tv_vals.append(float(np.median(g[y0:y1 + 1, max(0, mid - 1):mid + 2])))
        if cv_vals and tv_vals:
            diff = float(np.median(cv_vals)) - float(np.median(tv_vals))
            # 15 灰度级的柱/沟对比即可拿满 0.35 分
            contrast_fit = float(np.clip(diff / 15.0, -1.0, 1.0)) * 0.35

    edge_fit = 0.0
    if g is not None and g.size:
        gx = np.abs(np.gradient(g.astype(np.float32), axis=1))
        vals = []
        for c in columns:
            for y, s, e in (c.get("ranges") or [])[::max(1, len(c.get("ranges") or []) // 24 or 1)]:
                yy = int(np.clip(y, 0, h - 1))
                ss = int(np.clip(s, 0, w - 1))
                ee = int(np.clip(e, 0, w - 1))
                vals.append((float(gx[yy, ss]) + float(gx[yy, ee])) * 0.5)
        if vals:
            edge_fit = float(np.clip(np.median(vals) / 8.0, 0.0, 1.0))

    # 数量奖励必须由主体覆盖度加权，否则 ROI 中 CLAHE 产生的多个短阴影
    # 会击败少量完整柱。使用原灰度主体高度，避免依赖 ROI 框本身的大小。
    if g is not None:
        main_top, main_bottom, _ = _main_row_activity(g, prefer_upper=False)
        main_height = max(1, main_bottom - main_top + 1)
        coverage = np.asarray([
            max(0, min(c["y_bottom"], main_bottom) - max(c["y_top"], main_top) + 1) / main_height
            for c in columns
        ])
        count_fit = min(3.0, 0.55 * float(np.sum(coverage ** 2)))
    else:
        count_fit = min(3.0, 0.55 * float(len(columns)))
    return float(
        count_fit
        + 1.45 * width_fit
        + 1.20 * consistency
        + 0.85 * height_fit
        + 0.55 * gap_fit
        + 0.45 * edge_fit
        + contrast_fit
        - edge_penalty
    )


def _reported_cd_range(columns, image_width=None):
    """把用户先验或自动估计转换成可写入输出的像素范围。"""
    hint = _auto_cd_range()
    if hint is not None:
        center, tolerance = hint
        source = "user"
    else:
        all_cols = list(columns or [])
        # 左右贴边柱可能只是被裁掉的一部分，自动估计优先采用至少
        # 两根完整柱的中位宽度；只有没有足够完整柱时才退回全部候选。
        if image_width is None:
            # 兼容旧调用：用候选极值近似图像边界；有真实图像宽度时由
            # 调用方传入，能更准确地识别贴边残柱。
            min_left = min(
                (int(c.get("left", 0)) for c in all_cols),
                default=0,
            )
            max_right = max(
                (int(c.get("right", 0)) for c in all_cols),
                default=0,
            )
        else:
            min_left = 0
            max_right = max(0, int(image_width) - 1)
        body_cols = [
            c for c in all_cols
            if int(c.get("left", 0)) > min_left + 1
            and int(c.get("right", 0)) < max_right - 1
        ]
        source_cols = body_cols if len(body_cols) >= 2 else all_cols
        widths = np.asarray([
            _column_width_px(c) for c in source_cols
            if _column_width_px(c) > 0
        ], dtype=np.float32)
        if widths.size == 0:
            return None
        # 优先用中位数，避免左右贴边残柱把自动估计拉窄。
        center = float(np.median(widths))
        # 估计值为 0 时，仍使用用户输入的 ±像素作为自动范围；
        # 这样两个简化参数始终有明确作用，且结果更容易复核。
        tolerance = _auto_cd_tolerance(center)
        source = "auto"
    return {
        "min": float(max(1.0, center - tolerance)),
        "max": float(center + tolerance),
        "center": float(center),
        "tolerance": float(tolerance),
        "source": source,
    }


def _observed_cd_range(columns, image_width=None):
    """统计最终轮廓实际观测到的柱宽范围（不等同于用户先验）。"""
    cols = list(columns or [])
    if not cols:
        return None

    if image_width is None:
        image_width = max(
            (int(c.get("right", 0)) for c in cols),
            default=0,
        ) + 1
    body = [
        c for c in cols
        if int(c.get("left", 0)) > 1
        and int(c.get("right", 0)) < int(image_width) - 2
    ]
    source = body if len(body) >= 2 else cols
    widths = np.asarray(
        [_column_width_px(c) for c in source], dtype=np.float64
    )
    widths = widths[np.isfinite(widths) & (widths > 0)]
    if not widths.size:
        return None

    center = float(np.median(widths))
    lo = float(np.min(widths))
    hi = float(np.max(widths))
    return {
        "min": lo,
        "max": hi,
        "center": center,
        "tolerance": max(0.0, max(center - lo, hi - center)),
        "source": "observed",
    }


def _detect_columns_geometric(gray, roi_mode=False, y_hint=None):
    """几何优先柱识别，返回 ``(columns, x_profile)``；失败返回 ``(None,None)``。

    y_hint: 原图灰度的主体 y 范围 (top, bottom)。增强类候选（如
    enhanced/equalized）会把基底颗粒纹理的对比度也放大，导致行
    活动度把基底连进主体（如 A55 y[60:479] 贯穿全图）；用原图的
    y 范围作软约束限制搜索带，避免柱体追踪穿到基底里。
    """
    try:
        g = to_gray8(gray)
        h, w = g.shape[:2]
        y_top, y_bottom, row_activity = _main_row_activity(g, prefer_upper=not roi_mode)
        if y_hint is not None:
            try:
                ht, hb = int(y_hint[0]), int(y_hint[1])
                pad = max(6, int(round(h * 0.05)))
                # 双向夹紧：候选自身截太浅（如 enhanced 对比度不均
                # 导致底部提前截断）时扩展到原图柱脚；追踪穿过基底
                # （如颗粒纹理被 CLAHE 放大）时收缩回柱脚下方。
                y_top = max(min(y_top, ht), ht - pad)
                y_bottom = min(max(y_bottom, hb), hb + pad)
            except Exception:
                pass
        if y_bottom - y_top + 1 < max(MIN_COLUMN_HEIGHT, 12):
            return None, None

        profile, segments, _ = _geometric_x_profile(g, y_top, y_bottom)
        if len(segments) == 0:
            return None, None

        # 各柱 cell 由相邻实体段的中点分隔，边缘柱 cell 延伸到图像边。
        cells = []
        for i, (left, right) in enumerate(segments):
            cell_l = 0 if i == 0 else (segments[i - 1][1] + left) // 2
            cell_r = w - 1 if i == len(segments) - 1 else (right + segments[i + 1][0]) // 2
            cells.append((int(cell_l), int(cell_r)))

        gray_smooth, gradient = _smooth_row_and_gradient(g)
        seed_lo = min(y_bottom, y_top + max(5, int(round((y_bottom - y_top + 1) * 0.12))))
        seed_hi = max(seed_lo, y_bottom - max(5, int(round((y_bottom - y_top + 1) * 0.15))))
        body_activity = row_activity[seed_lo:seed_hi + 1]
        seed_y = int(seed_lo + np.argmax(body_activity)) if body_activity.size else int(seed_lo)

        columns = []
        for segment, cell in zip(segments, cells):
            ranges = _track_geometric_column(
                gray_smooth, gradient, segment, cell, seed_y, y_top, y_bottom
            )
            if len(ranges) < max(8, MIN_COLUMN_HEIGHT):
                continue

            # 底部延伸：追踪因局部对比度不足在柱脚前提前停止时
            # （如 A55 enhanced 候选在柱脚处中断），用最后一行的
            # 宽度延伸到 y_bottom，避免柱体下半段丢失。
            if ranges and ranges[-1][0] < y_bottom:
                last_y, ls, le = ranges[-1]
                for yy in range(last_y + 1, y_bottom + 1):
                    ranges.append((int(yy), int(ls), int(le)))

            widths = np.array([e - s + 1 for _, s, e in ranges], dtype=np.float32)
            finite = widths[np.isfinite(widths)]
            if finite.size == 0 or float(np.mean(finite)) < max(3.0, MIN_AVG_WIDTH * 0.45):
                continue
            y_min = int(ranges[0][0])
            y_max = int(ranges[-1][0])
            if y_max - y_min + 1 < max(12, MIN_COLUMN_HEIGHT * 0.75):
                continue

            columns.append({
                "left": int(min(s for _, s, _ in ranges)),
                "right": int(max(e for _, _, e in ranges)),
                "center_x": int(round((segment[0] + segment[1]) * 0.5)),
                "y_top": y_min,
                "y_bottom": y_max,
                "y_proj": row_activity,
                "ranges": ranges,
                "contour": build_contour(ranges),
            })

        if not columns:
            return None, None

        for i, col in enumerate(columns):
            col["label"] = i + 1
            col["color"] = PR_COLORS[i % len(PR_COLORS)]
        return columns, profile.astype(np.float32)
    except Exception:
        # 几何路径是增强项，任何异常都交给旧流程处理，保证 GUI 可用。
        return None, None


# =====================================================================
# 单柱 y 范围 + 逐行侧壁追踪
# =====================================================================


def build_contour(ranges):
    if not ranges:
        return np.empty((0, 1, 2), dtype=np.int32)

    ys = np.array([r[0] for r in ranges], dtype=np.int32)
    xs = np.array([r[1] for r in ranges], dtype=np.int32)
    xe = np.array([r[2] for r in ranges], dtype=np.int32)

    left = np.column_stack((xs, ys))
    right = np.column_stack((xe, ys))[::-1]
    return np.vstack([left, right]).astype(np.int32).reshape((-1, 1, 2))


def _pillar_evidence(columns, gray, sign):
    """逐柱对照同一行内外灰度，单根柱和 ROI 也执行极性检查。"""
    accepted = []
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32)
    noise_floor = float(np.median(np.abs(residual))) * 1.4826
    for col in columns or []:
        contrasts = []
        rows = col.get("ranges", [])
        for y, left, right in rows[::max(1, len(rows) // 32)]:
            y, left, right = int(y), int(round(left)), int(round(right))
            width = right - left
            if width < 3:
                continue
            inset = max(1, width // 4)
            inside = gray[y, left + inset:right - inset + 1]
            gap = max(2, min(12, width // 4))
            outside = np.concatenate((gray[y, max(0, left - gap):max(0, left - 1)],
                                      gray[y, min(gray.shape[1], right + 2):right + gap + 1]))
            if inside.size and outside.size:
                contrasts.append(sign * (float(np.median(inside)) - float(np.median(outside))))
        contrast = float(np.median(contrasts)) if contrasts else 0.0
        support = float(np.mean(np.asarray(contrasts) > 1.0)) if contrasts else 0.0
        # 相反极性不能通过更高的柱数/宽度评分抵消。
        if contrast > max(1.5, 1.8 * noise_floor) and support >= 0.65:
            item = dict(col)
            item["polarity_contrast"] = contrast
            item["polarity_support"] = support
            accepted.append(item)
    return accepted


def _binary_seed_columns(mask, gray, roi_mode=False):
    """binary 先确定柱身、顶部、底部；逐行轮廓只在邻柱间的单元内生长。"""
    h, w = mask.shape
    if h < MIN_COLUMN_HEIGHT or w < 5:
        return []
    top, bottom, _ = _main_row_activity(gray, prefer_upper=not roi_mode)
    if bottom - top < MIN_COLUMN_HEIGHT - 1:
        return []
    trim = max(2, (bottom - top) // 6)
    band = mask[top + trim:max(top + trim + 1, bottom - trim + 1)]
    projection = smooth_1d(np.mean(band > 0, axis=0), 5)
    segments = _run_segments(projection >= 0.5, min_len=5)
    cols = []
    for i, (left, right) in enumerate(segments):
        cell_l = 0 if i == 0 else (segments[i - 1][1] + left) // 2 + 1
        cell_r = w - 1 if i == len(segments) - 1 else (right + segments[i + 1][0]) // 2
        center = (left + right) * 0.5
        rows = []
        for y in range(top, bottom + 1):
            runs = _run_segments(mask[y, cell_l:cell_r + 1] > 0, min_len=3)
            runs = [(a + cell_l, b + cell_l) for a, b in runs]
            runs = [(a, b) for a, b in runs if a <= center <= b]
            if not runs:
                continue
            a, b = max(runs, key=lambda r: r[1] - r[0])
            # 衬底整行相连时不把相邻 cell 的中点作为真实侧壁。
            if (a == cell_l and cell_l > 0) or (b == cell_r and cell_r < w - 1):
                continue
            rows.append((y, a, b))
        if len(rows) < MIN_COLUMN_HEIGHT:
            continue
        # 不跨越大的无证据缺口插值。
        chunks = np.split(np.asarray(rows), np.where(np.diff([r[0] for r in rows]) > MAX_TRACE_GAP)[0] + 1)
        arr = max(chunks, key=len)
        if len(arr) < MIN_COLUMN_HEIGHT:
            continue
        ys = np.arange(arr[0, 0], arr[-1, 0] + 1)
        rows = [(int(y), float(a), float(b)) for y, a, b in zip(
            ys, np.interp(ys, arr[:, 0], arr[:, 1]), np.interp(ys, arr[:, 0], arr[:, 2]))]
        width = float(np.median([b - a for _, a, b in rows]))
        edge = left <= 1 or right >= w - 2
        if width < (3 if edge else MIN_AVG_WIDTH) or len(rows) / max(width, 1) < MIN_ASPECT_RATIO:
            continue
        col = {"ranges": rows, "left": min(a for _, a, _ in rows),
               "right": max(b for _, _, b in rows), "center_x": int(round(center)),
               "y_top": rows[0][0], "y_bottom": rows[-1][0], "width": width,
               "contour": build_contour(rows), "cell": (cell_l, cell_r)}
        cols.append(col)
    return cols


def _constrained_edge_path(response, seed, radius, lower=0, upper=None):
    """窄带动态规划：梯度证据 + binary 距离 + 行间连续性，抛物线亚像素定位。"""
    response = np.asarray(response, dtype=np.float32)
    seed = np.asarray(seed, dtype=np.float32)
    n, width = response.shape
    lo = np.broadcast_to(np.maximum(0, lower), (n,)).astype(float)
    hi = np.broadcast_to(np.minimum(width - 1, width - 1 if upper is None else upper), (n,)).astype(float)
    seed = np.clip(seed, lo, hi)
    offsets = np.arange(-radius, radius + 1)
    xs = np.clip(np.rint(seed).astype(int)[:, None] + offsets,
                 np.ceil(lo).astype(int)[:, None], np.floor(hi).astype(int)[:, None])
    values = np.maximum(response[np.arange(n)[:, None], xs], 0)
    scale = max(1.0, float(np.percentile(values.max(axis=1), 75)))
    unary = -np.clip(values / scale, 0, 3) + 0.15 * (offsets / max(radius, 1)) ** 2
    costs = unary[0].copy()
    back = np.zeros(xs.shape, dtype=np.int16)
    for y in range(1, n):
        delta = xs[y][:, None] - xs[y - 1][None, :]
        transition = 0.07 * np.minimum(delta * delta, 36) + 0.10 * np.abs(delta)
        total = costs[None, :] + transition
        back[y] = np.argmin(total, axis=1)
        costs = unary[y] + total[np.arange(len(offsets)), back[y]]
    state = int(np.argmin(costs))
    path = np.zeros(n, dtype=float)
    strength = np.zeros(n, dtype=float)
    for y in range(n - 1, -1, -1):
        x = int(xs[y, state])
        strength[y] = max(0.0, float(response[y, x]))
        dx = 0.0
        if 0 < x < width - 1 and strength[y] > 0:
            a, b, c = map(float, response[y, x - 1:x + 2])
            den = a - 2 * b + c
            if den < -1e-6 and b >= a and b >= c:
                dx = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        path[y] = np.clip(x + dx, max(lo[y], seed[y] - radius), min(hi[y], seed[y] + radius))
        state = int(back[y, state])
    valid = strength > max(0.8, 0.10 * scale)
    # 无梯度不伪造测量点，保留 binary 位置并在输出中标记。
    path[~valid] = seed[~valid]
    return path, strength, valid


def _refine_binary_columns(columns, filtered, sign):
    h, w = filtered.shape
    gx = cv2.Scharr(filtered, cv2.CV_32F, 1, 0) / 32.0 * sign
    gy = cv2.Scharr(filtered, cv2.CV_32F, 0, 1) / 32.0 * sign
    result = []
    typical_width = float(np.median([_column_width_px(c) for c in columns])) if columns else 20.0
    for original in columns:
        col = dict(original)
        coarse = np.asarray(col["ranges"], dtype=float)
        width = float(np.median(coarse[:, 2] - coarse[:, 1]))
        radius = min(REFINE_RADIUS_PX, max(3, int(max(width, typical_width) * 0.32)))
        y0, y1 = int(coarse[0, 0]), int(coarse[-1, 0])
        # 逐列扫描顶部。底部可能与同亮度基座融合，补查柱外邻近列的基座跃迁。
        x0 = max(0, int(np.median(coarse[:, 1]) + width * 0.2))
        x1 = min(w - 1, int(np.median(coarse[:, 2]) - width * 0.2))
        cap_points = []
        limits = []
        top_strength = 1.0
        for side, seed_y, direction in (("top", y0, 1), ("bottom", y1, -1)):
            xs = np.arange(x0, max(x0 + 1, x1 + 1))
            cap_radius = min(radius, 8)
            lo, hi = max(0, seed_y - cap_radius), min(h - 1, seed_y + cap_radius)
            local = (direction * gy[lo:hi + 1, xs]).T
            pos, strengths, valid = _constrained_edge_path(local, np.full(len(xs), seed_y - lo), radius)
            for x, y, strength, ok in zip(xs, pos + lo, strengths, valid):
                cap_points.append({"side": side, "x": float(x), "y": float(y),
                                   "strength": float(strength), "valid": bool(ok), "source": "column_gradient"})
            valid_y = (pos + lo)[valid]
            source = "column_gradient"
            if side == "top":
                top_strength = float(np.median(strengths))
            if side == "bottom" and float(np.median(strengths)) < max(1.5, 0.15 * top_strength):
                exterior = []
                for edge_x, offset in ((coarse[-1, 1], -1), (coarse[-1, 2], 1)):
                    for distance in range(3, min(10, max(4, int(width * 0.15)))):
                        xx = int(round(edge_x + offset * distance))
                        if 0 <= xx < w:
                            segment = gy[lo:hi + 1, xx]
                            k = int(np.argmax(segment))
                            if segment[k] > 1.0:
                                exterior.append(lo + k)
                                cap_points.append({"side": "base_reference", "x": float(xx), "y": float(lo + k),
                                                   "strength": float(segment[k]), "valid": True, "source": "outside_column"})
                if len(exterior) >= 4 and float(np.std(exterior)) < max(2, radius * 0.6):
                    valid_y = np.asarray(exterior)
                    source = "base_reference"
            supported = len(valid_y) >= max(2, len(xs) // 3) if source != "base_reference" else True
            limits.append(float(np.median(valid_y)) if supported else float(seed_y))
            col[side + "_source"] = source if supported else "binary_fallback"
        new_top = max(0, int(round(limits[0])))
        new_bottom = min(h - 1, int(round(limits[1])))
        if new_bottom - new_top < MIN_COLUMN_HEIGHT - 1:
            new_top, new_bottom = y0, y1
        ys = np.arange(new_top, new_bottom + 1)
        seeds_l = np.interp(ys, coarse[:, 0], coarse[:, 1])
        seeds_r = np.interp(ys, coarse[:, 0], coarse[:, 2])
        cell_l, cell_r = col.get("cell", (0, w - 1))
        middle = (seeds_l + seeds_r) * .5
        # 灰度精修保留在本柱单元、各自半侧内，避免强邻柱/内部纹理抢走边缘。
        left, ls, lv = _constrained_edge_path(gx[ys], seeds_l, radius, cell_l, np.floor(middle - 1))
        right, rs, rv = _constrained_edge_path(-gx[ys], seeds_r, radius, np.ceil(middle + 1), cell_r)
        for coords, seeds, valid in ((left, seeds_l, lv), (right, seeds_r, rv)):
            clipped = (seeds <= 0.5) | (seeds >= w - 1.5)
            # 少数 binary 行被阴影/基座连到框边，不代表整根柱被截断；
            # 仍使用邻行约束的灰度精定位，不能把这些点硬锁在 ROI 框边。
            if np.mean(clipped) < .1:
                clipped[:] = False
            coords[clipped] = seeds[clipped]
            valid[clipped] = False
        bad = right <= left + 1
        left[bad], right[bad] = seeds_l[bad], seeds_r[bad]
        lv[bad] = rv[bad] = False
        rows = [(int(y), float(a), float(b)) for y, a, b in zip(ys, left, right)]
        points = cap_points
        for side, coords, strength, valid in (("left", left, ls, lv), ("right", right, rs, rv)):
            for y, x, st, ok in zip(ys, coords, strength, valid):
                points.append({"side": side, "x": float(x), "y": float(y), "strength": float(st),
                               "valid": bool(ok), "source": "row_gradient" if ok else "binary_fallback"})
        col.update(ranges=rows, coarse_ranges=original["ranges"], points=points,
                   left=int(np.floor(left.min())), right=int(np.ceil(right.max())),
                   y_top=new_top, y_bottom=new_bottom, top_subpixel=limits[0], bottom_subpixel=limits[1],
                   center_x=int(round(np.median((left + right) * 0.5))),
                   contour=build_contour(rows), refine_radius=radius,
                   edge_support=float(np.mean(np.r_[lv, rv])))
        col["clipped"] = bool(np.mean((left <= .5) | (right >= w - 1.5)) >= .1
                              or new_top <= 0 or new_bottom >= h - 1)
        col["y_proj"] = np.zeros(h, dtype=np.float32)
        col["y_proj"][ys] = right - left
        result.append(col)
    return result


def _detect_columns(img, binary, aux, roi_mode=False, on_stage=None):
    gray = aux.get("gray", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    filtered = aux.get("filtered")
    if filtered is None:
        filtered = cv2.bilateralFilter(cv2.medianBlur(gray, 3), 5, 20, 7)
    h, w = gray.shape
    sign_options = [1, -1] if PILLAR_POLARITY == "auto" else ([-1] if PILLAR_POLARITY == "dark" else [1])
    trials = []
    winner = None
    best_score = -1e9
    for sign in sign_options:
        oriented = filtered if sign == 1 else 255 - filtered
        enhanced = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(oriented)
        # 仅去除全局横向照明坡度，不用局部高通，以免实心柱变成两条亮边。
        xx = np.linspace(-1, 1, w, dtype=np.float32)
        column_mean = np.mean(oriented, axis=0)
        slope = float(np.dot(column_mean - column_mean.mean(), xx) / max(float(np.dot(xx, xx)), 1e-6))
        corrected = cv2.normalize(oriented.astype(np.float32) - slope * xx[None, :],
                                  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for name, source in (("filtered_otsu", oriented), ("clahe_otsu", enhanced), ("illumination_otsu", corrected)):
            _, mask = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            cols = _binary_seed_columns(mask, oriented, roi_mode)
            cols = _pillar_evidence(_apply_cd_hint(cols, w), gray, sign)
            score = _detection_confidence(cols, oriented)
            trials.append({"variant": name, "polarity": "bright" if sign == 1 else "dark", "count": len(cols), "score": float(score)})
            if cols and score > best_score:
                winner, best_score = (cols, mask, sign, name), score
        # binary 无法建立主体时，几何提议仅用于重建局部 binary 粗轮廓。
        if not any(t["count"] for t in trials if t["polarity"] == ("bright" if sign == 1 else "dark")):
            proposals, _ = _detect_columns_geometric(oriented, roi_mode=roi_mode)
            cols = _pillar_evidence(_apply_cd_hint(proposals or [], w), gray, sign)
            mask = np.zeros_like(gray)
            for col in cols:
                cv2.drawContours(mask, [col["contour"]], -1, 255, -1)
            score = _detection_confidence(cols, oriented) - 0.25
            trials.append({"variant": "geometric_seed", "polarity": "bright" if sign == 1 else "dark", "count": len(cols), "score": float(score)})
            if cols and score > best_score:
                winner, best_score = (cols, mask, sign, "geometric_seed"), score
    cols, chosen_mask, sign, variant = winner if winner else ([], np.zeros_like(gray), sign_options[0], "none")
    aux["threshold_mode"] = variant
    aux["binary_fraction"] = _binary_fraction(chosen_mask)
    coarse = [dict(c) for c in cols]
    if on_stage is not None:
        on_stage("step4_binary.png", chosen_mask)
        coarse_img = img.copy()
        for col in coarse:
            cv2.drawContours(coarse_img, [build_contour(col["ranges"])], -1, (0, 165, 255), 1)
        on_stage("step5_binary_contours.png", coarse_img)
    cols = sorted(_refine_binary_columns(cols, filtered, sign), key=lambda c: c["center_x"])
    for i, col in enumerate(cols, 1):
        col["detected_label"] = i
    detected_count = len(cols)
    excluded = []
    if not roi_mode and cols:
        # 按左右顺序排除两端，不判断完整/残柱。1~2 根时没有中间柱可保留。
        excluded = [cols[0]] if len(cols) == 1 else [cols[0], cols[-1]]
        cols = cols[1:-1]
    for i, col in enumerate(cols, 1):
        col.update(label=i, color=PR_COLORS[(i - 1) % len(PR_COLORS)])
    warning = ""
    if not cols and excluded:
        warning = f"检测到 {detected_count} 根柱，按规则排除最左/最右柱后无剩余柱；可用 ROI 单独测量。"
    elif not cols:
        warning = "未找到满足极性与边缘证据的柱子；请检查 ROI 或柱体明暗模式。"
    elif PILLAR_POLARITY == "auto":
        warning = "自动极性仅由灰度证据推断，请核对柱体与 trench 的语义。"
    elif any(c["edge_support"] < 0.6 for c in cols):
        warning = "部分轮廓梯度证据不足或被图像边界裁切；请检查过程图中的黄色点。"
    hint = _auto_cd_range()
    aux.update(detection_mode="binary_dp_subpixel", detection_variant=variant,
               detection_confidence=float(best_score) if cols else -1.0,
               detection_attempts=len(trials), detection_warning=warning,
               cd_hint_px={"estimate": hint[0], "tolerance": hint[1]} if hint else None,
               cd_range_px=_reported_cd_range(cols, w), cd_observed_px=_observed_cd_range(cols, w),
               selected_binary=chosen_mask, coarse_columns=coarse, candidate_scores=trials,
               excluded_columns=excluded, detected_count=detected_count, exclude_outermost=not roi_mode,
               selected_polarity="bright" if sign == 1 else "dark", filtered=filtered)
    profile = np.mean(chosen_mask > 0, axis=0).astype(np.float32)
    return cols, profile


def _offset_columns(columns, dx, dy):
    for col in columns:
        col["left"] += dx
        col["right"] += dx
        col["center_x"] += dx
        col["y_top"] += dy
        col["y_bottom"] += dy
        col["ranges"] = [
            (y + dy, s + dx, e + dx)
            for y, s, e in col["ranges"]
        ]
        col["contour"] = build_contour(col["ranges"])
        if "coarse_ranges" in col:
            col["coarse_ranges"] = [(y + dy, l + dx, r + dx) for y, l, r in col["coarse_ranges"]]
        for point in col.get("points", []):
            point["x"] += dx
            point["y"] += dy
        for key in ("top_subpixel", "bottom_subpixel"):
            if key in col:
                col[key] += dy
    return columns


# =====================================================================
# 绘图
# =====================================================================

def draw_axis_with_ticks(img, tick_step=50, nm_per_px=1.0):
    h, w = img.shape[:2]

    out = np.full(
        (h + AXIS_MARGIN_B, w + AXIS_MARGIN_L, 3),
        255,
        dtype=np.uint8
    )
    out[:h, AXIS_MARGIN_L:AXIS_MARGIN_L + w] = img

    unit = "nm" if nm_per_px != 1.0 else "px"

    def fmt(v):
        return f"{v * nm_per_px:.1f}" if nm_per_px != 1.0 else str(int(v))

    y_axis = h - 1
    x_axis = AXIS_MARGIN_L

    cv2.line(out, (x_axis, y_axis), (x_axis + w - 1, y_axis), (0, 0, 0), 1)
    cv2.line(out, (x_axis, 0), (x_axis, h - 1), (0, 0, 0), 1)

    for x in range(0, w, max(1, int(tick_step))):
        px = AXIS_MARGIN_L + x
        cv2.line(out, (px, y_axis), (px, y_axis + 6), (0, 0, 0), 1)
        cv2.putText(
            out, fmt(x),
            (px + 2, y_axis + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (0, 0, 0), 1
        )

    for y in range(0, h, max(1, int(tick_step))):
        cv2.line(out, (x_axis, y), (x_axis - 6, y), (0, 0, 0), 1)
        cv2.putText(
            out, fmt(y),
            (max(1, x_axis - 38), y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38, (0, 0, 0), 1
        )

    cv2.putText(
        out, unit,
        (max(1, AXIS_MARGIN_L + w - 40), y_axis + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 0, 0), 1
    )
    cv2.putText(
        out, unit,
        (2, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38, (0, 0, 0), 1
    )
    return out


def draw_1d_profile_vertical(
    data, width, height,
    color=(255, 0, 0),
    fill_color=(220, 220, 255),
    title=""
):
    data = np.asarray(data, dtype=np.float32)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    if len(data) < 2:
        return canvas

    dmin = float(data.min())
    dmax = float(data.max())
    if dmax <= dmin:
        norm = np.zeros_like(data)
    else:
        norm = (data - dmin) / (dmax - dmin)

    xs = np.linspace(0, width - 1, len(norm)).astype(np.int32)
    usable_h = max(10, height - 48)
    ys = (height - 12 - norm * usable_h).astype(np.int32)

    pts = np.column_stack((xs, ys))
    poly = np.vstack([
        pts,
        np.array([[xs[-1], height - 1], [xs[0], height - 1]], dtype=np.int32)
    ])
    cv2.fillPoly(canvas, [poly.astype(np.int32)], fill_color)
    cv2.polylines(canvas, [pts.astype(np.int32)], False, color, 1, cv2.LINE_AA)

    if title:
        cv2.rectangle(canvas, (0, 0), (width - 1, 24), (225, 225, 225), -1)
        cv2.putText(
            canvas, title, (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (0, 0, 0), 1
        )
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (160, 160, 160), 1)
    return canvas


def draw_1d_profile_horizontal(
    data, width, height,
    color=(255, 0, 0),
    fill_color=(245, 245, 245),
    title=""
):
    data = np.asarray(data, dtype=np.float32)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    if len(data) < 2:
        return canvas

    dmin = float(data.min())
    dmax = float(data.max())
    if dmax <= dmin:
        norm = np.zeros_like(data)
    else:
        norm = (data - dmin) / (dmax - dmin)

    ys = np.linspace(0, height - 1, len(norm)).astype(np.int32)
    xs = (5 + norm * max(5, width - 15)).astype(np.int32)
    pts = np.column_stack((xs, ys))

    poly = np.vstack([
        pts,
        np.array([[0, ys[-1]], [0, ys[0]]], dtype=np.int32)
    ])
    cv2.fillPoly(canvas, [poly.astype(np.int32)], fill_color)
    cv2.polylines(canvas, [pts.astype(np.int32)], False, color, 1, cv2.LINE_AA)

    if title:
        cv2.rectangle(canvas, (0, 0), (width - 1, 24), (225, 225, 225), -1)
        cv2.putText(
            canvas, title, (5, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40,
            (0, 0, 0), 1
        )
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (160, 160, 160), 1)
    return canvas


def render_main_image(img, columns, nm_per_px=1.0):
    h, w = img.shape[:2]
    main = img.copy()

    for col in columns:
        color = tuple(int(v) for v in col["color"])
        contour = col["contour"]
        if contour.size == 0:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        layer = np.zeros_like(img)
        layer[:] = color
        blended = cv2.addWeighted(img, 1.0, layer, ALPHA, 0)

        main[mask == 255] = blended[mask == 255]
        cv2.drawContours(main, [contour], -1, color, LINE_THICKNESS)

    main = draw_axis_with_ticks(main, 50, nm_per_px)

    xoff = AXIS_MARGIN_L
    for col in columns:
        if not col["ranges"]:
            continue

        color = tuple(int(v) for v in col["color"])
        top = min(r[0] for r in col["ranges"])
        bot = col["y_bottom"]
        cx = col["center_x"] + xoff

        label = f"PR{col['label']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        lx = int(np.clip(cx - tw // 2, 1, main.shape[1] - tw - 2))
        ly = int(max(top - 10, th + 3))

        cv2.rectangle(
            main,
            (max(0, lx - 3), max(0, ly - th - 2)),
            (min(main.shape[1] - 1, lx + tw + 3), min(main.shape[0] - 1, ly + 3)),
            (255, 255, 255),
            -1
        )
        cv2.putText(
            main, label, (lx, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62, color, 2
        )

        last = col["ranges"][-1]
        for xx in range(int(round(last[1])) + xoff, int(round(last[2])) + xoff + 1, 6):
            cv2.line(
                main,
                (xx, bot),
                (int(min(xx + 3, round(last[2]) + xoff)), bot),
                color, 1
            )

    return main


def render_combined(img, columns, structure_profile, nm_per_px=1.0, main_img=None):
    h, w = img.shape[:2]
    unit = "nm" if nm_per_px != 1.0 else "px"

    def fmt(v):
        return f"{v * nm_per_px:.1f}" if nm_per_px != 1.0 else str(int(v))

    main_img = main_img if main_img is not None else render_main_image(img, columns, nm_per_px)

    profile_h = 120
    x_profile = draw_1d_profile_vertical(
        structure_profile,
        w, profile_h,
        title=f"x structure profile ({unit})"
    )

    panels = []
    for col in columns:
        p = draw_1d_profile_horizontal(
            col["y_proj"],
            PANEL_W, h,
            color=col["color"],
            title=f"y PR{col['label']}"
        )

        yt = int(col["y_top"])
        yb = int(col["y_bottom"])

        for xx in range(0, PANEL_W, 8):
            cv2.line(p, (xx, yt), (min(xx + 4, PANEL_W - 1), yt), (0, 180, 0), 1)
            cv2.line(p, (xx, yb), (min(xx + 4, PANEL_W - 1), yb), (0, 0, 255), 1)

        cv2.putText(
            p, f"T={fmt(yt)}",
            (3, max(30, min(h - 5, yt + 13))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
            (0, 150, 0), 1
        )
        cv2.putText(
            p, f"B={fmt(yb)}",
            (3, max(30, min(h - 5, yb + 13))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
            (0, 0, 220), 1
        )
        panels.append(p)

    y_profile = (
        np.hstack(panels)
        if panels
        else np.full((h, PANEL_W, 3), 255, dtype=np.uint8)
    )

    mh, mw = main_img.shape[:2]
    pw = y_profile.shape[1]

    out = np.full((mh + profile_h, mw + pw, 3), 255, dtype=np.uint8)
    out[:mh, :mw] = main_img
    out[mh:mh + profile_h, AXIS_MARGIN_L:AXIS_MARGIN_L + w] = x_profile
    out[:h, mw:mw + pw] = y_profile

    cv2.line(out, (mw, 0), (mw, out.shape[0] - 1), (170, 170, 170), 1)
    cv2.line(out, (0, mh), (out.shape[1] - 1, mh), (170, 170, 170), 1)

    cv2.putText(
        out, f"unit: {unit}",
        (mw + max(4, pw - 70), mh + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (80, 80, 80), 1
    )
    return out


# =====================================================================
# 边界导出 + 统计
# =====================================================================

def _cjk_width(s):
    return sum(2 if ord(ch) > 0x7F else 1 for ch in str(s))


def write_aligned_table(txt_path, rows, left_first_text=False, footer=None):
    if not rows:
        with open(txt_path, "w", encoding="utf-8"):
            pass
        return

    ncols = max(len(r) for r in rows)
    widths = [0] * ncols

    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], _cjk_width(value))

    lines = []
    for row in rows:
        cells = []
        for i in range(ncols):
            text = str(row[i]) if i < len(row) else ""
            pad = " " * max(0, widths[i] - _cjk_width(text))
            if left_first_text and i == 0:
                cells.append(text + pad)
            else:
                cells.append(pad + text)
        lines.append("  ".join(cells).rstrip())

    if footer:
        lines.append(footer)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_boundaries(txt_path, columns, nm_per_px=1.0, detection_meta=None):
    unit = "nm" if nm_per_px != 1.0 else "px"

    def fmt(v):
        if nm_per_px == 1.0:
            return f"{float(v):.4f}".rstrip("0").rstrip(".")
        # 导出保留 4 位；统计本身不依赖此文本
        return f"{float(v) * nm_per_px:.4f}"

    all_y = sorted({r[0] for c in columns for r in c["ranges"]})
    maps = {
        c["label"]: {r[0]: (r[1], r[2]) for r in c["ranges"]}
        for c in columns
    }

    header = [f"y({unit})"] + [
        f"PR{c['label']}(xs,xe)"
        for c in columns
    ]

    rows = [header]
    for y in all_y:
        row = [fmt(y)]
        for c in columns:
            seg = maps[c["label"]].get(y)
            row.append(f"({fmt(seg[0])},{fmt(seg[1])})" if seg else "-")
        rows.append(row)

    footer_lines = [
        f"# 列格式=(x_start,x_end); 单位={unit}; '-'=该y处该柱无轮廓",
        "# 宽度口径：Line CD/先验/观测范围与统计表均为 xe-xs（边界距离）；"
        "内部几何像素数量可能按含端点格数计算",
    ]
    if isinstance(detection_meta, dict):
        # 边界文件保持可被旧解析器读取：所有元数据都放在注释行中。
        mode = detection_meta.get("mode", "unknown")
        variant = detection_meta.get("variant", "unknown")
        conf = detection_meta.get("confidence")
        attempts = detection_meta.get("attempts", 0)
        try:
            conf_text = f"{float(conf):.3f}"
        except Exception:
            conf_text = "-"
        footer_lines.extend([
            f"# 识别模式={mode}; 候选版本={variant}; 置信度={conf_text}; "
            f"候选尝试次数={attempts}",
        ])
        rng = detection_meta.get("cd_range_px")
        if isinstance(rng, dict):
            try:
                range_source = str(rng.get("source", "unknown"))
                range_label = "用户先验" if range_source == "user" else "自动估计"
                range_rule = "±1×输入容差" if range_source == "user" else "中心±输入容差"
                footer_lines.append(
                    f"# {range_label} Line CD 范围={float(rng.get('min')):.2f}~"
                    f"{float(rng.get('max')):.2f}px（{range_rule}；"
                    f"来源={range_source}）"
                )
                footer_lines.append(
                    f"# 候选保护范围={float(rng.get('protection_min')):.2f}~"
                    f"{float(rng.get('protection_max')):.2f}px（±2×输入容差；"
                    "边缘/软候选可能进一步放宽）"
                )
            except Exception:
                pass
        observed = detection_meta.get("cd_observed_px")
        if isinstance(observed, dict):
            try:
                footer_lines.append(
                    f"# 主体柱观测 CD 中位={float(observed.get('center')):.2f}px; "
                    f"范围={float(observed.get('min')):.2f}~"
                    f"{float(observed.get('max')):.2f}px（边界距离）"
                )
            except Exception:
                pass
            for key, label in (
                ("all", "全部柱观测 CD"),
                ("edge", "边缘残柱观测 CD"),
            ):
                detail = observed.get(key)
                if not isinstance(detail, dict):
                    continue
                try:
                    footer_lines.append(
                        f"# {label} 中位={float(detail.get('center')):.2f}px; "
                        f"范围={float(detail.get('min')):.2f}~"
                        f"{float(detail.get('max')):.2f}px; "
                        f"数量={int(detail.get('count', 0))}（边界距离）"
                    )
                except Exception:
                    pass

    write_aligned_table(
        txt_path,
        rows,
        footer="\n".join(footer_lines)
    )


def _split_three(rows):
    n = len(rows)
    if n <= 0:
        return [[], [], []]

    segs = [[], [], []]
    for i, row in enumerate(rows):
        segs[min(2, (i * 3) // n)].append(row)
    return segs


def _row_step(rows):
    if len(rows) < 2:
        return 0.0

    diffs = [
        rows[i + 1][0] - rows[i][0]
        for i in range(len(rows) - 1)
    ]
    positive = [d for d in diffs if d > 1e-12]
    return min(positive) if positive else 0.0


def _reg_slope(rows, right_side=False):
    if len(rows) < 2:
        return 0.0

    ys = np.array([r[0] for r in rows], dtype=np.float64)
    xs = np.array(
        [r[2] if right_side else r[1] for r in rows],
        dtype=np.float64
    )

    denom = float(np.sum((ys - ys.mean()) ** 2))
    if denom <= 1e-15:
        return 0.0

    return float(
        np.sum((ys - ys.mean()) * (xs - xs.mean())) / denom
    )


def _side_angle_from_slope(b):
    if b is None or not math.isfinite(b):
        return None
    return math.degrees(math.atan2(1.0, b))


def _side_angle(x_top, x_bot, dy):
    if dy <= 0:
        return None
    return math.degrees(math.atan2(dy, x_bot - x_top))


def analyze_pr_column(rows):
    keys = (
        "height",
        "cd_top", "cd_mid", "cd_bot",
        "ratio_tm", "ratio_td", "ratio_md",
        "angle_l_top", "angle_l_mid", "angle_l_bot",
        "angle_r_top", "angle_r_mid", "angle_r_bot",
        "ler_left", "ler_right",
    )
    out = {k: None for k in keys}

    if not rows:
        return out

    step = _row_step(rows)
    out["height"] = (rows[-1][0] - rows[0][0]) + step

    segs = _split_three(rows)

    def mean_cd(seg):
        if not seg:
            return None
        return float(np.mean([xe - xs for _, xs, xe in seg]))

    out["cd_top"], out["cd_mid"], out["cd_bot"] = [
        mean_cd(s)
        for s in segs
    ]

    def ratio(a, b):
        if a is None or b is None or abs(b) <= 1e-15:
            return None
        return a / b

    out["ratio_tm"] = ratio(out["cd_top"], out["cd_mid"])
    out["ratio_td"] = ratio(out["cd_top"], out["cd_bot"])
    out["ratio_md"] = ratio(out["cd_mid"], out["cd_bot"])

    def seg_angle(seg):
        if len(seg) < 2:
            return None, None

        if len(seg) >= 4:
            return (
                _side_angle_from_slope(_reg_slope(seg, False)),
                _side_angle_from_slope(_reg_slope(seg, True)),
            )

        # 两点方向向量应使用真实 y 差，不再 +step
        dy = seg[-1][0] - seg[0][0]
        return (
            _side_angle(seg[0][1], seg[-1][1], dy),
            _side_angle(seg[0][2], seg[-1][2], dy),
        )

    for key, seg in zip(("top", "mid", "bot"), segs):
        al, ar = seg_angle(seg)
        out[f"angle_l_{key}"] = al
        out[f"angle_r_{key}"] = ar

    return out


MORPHOMETRY = [
    ("height_nm", "高度", "nm"),
    ("cd_mean_nm", "平均 CD", "nm"),
    ("t_top_nm", "T-top 顶部展宽量", "nm"),
    ("top_rounding_nm", "Top-rounding 等效圆角半径", "nm"),
    ("middle_thin_nm", "中间细：收缩量", "nm"),
    ("middle_thick_nm", "中间粗：鼓出量", "nm"),
    ("footing_nm", "Footing 底部展宽量", "nm"),
    ("ler_left_nm", "左侧 LER 3σ", "nm"),
    ("ler_right_nm", "右侧 LER 3σ", "nm"),
    ("ler_nm", "两侧平均 LER 3σ", "nm"),
    ("lwr_nm", "LWR 3σ", "nm"),
    ("wave_amplitude_nm", "周期起伏：单边平均幅值", "nm"),
    ("wave_period_nm", "周期起伏：主周期", "nm"),
]
MORPH_SHAPES = [
    ("t_top_nm", "T-top"), ("top_rounding_nm", "Top-rounding"),
    ("middle_thin_nm", "中间细"), ("middle_thick_nm", "中间粗"),
    ("footing_nm", "Footing"), ("ler_nm", "LER"), ("lwr_nm", "LWR"),
]
ROUGHNESS_KEYS = {"ler_left_nm", "ler_right_nm", "ler_nm", "lwr_nm",
                  "wave_amplitude_nm", "wave_period_nm"}
PROFILE_SAMPLES = 401
MORPHOLOGY_DEFINITION = """每根柱以中段中心为 x=0，顶部为 y=0；按相对高度 0~1 对齐，等权平均左右边界坐标。
不归一化宽度；y 换算为各柱平均高度，所有几何值保留 nm 尺度。
T-top：顶部 0~18% 的宽度相对 20~35% 柱身线性外推的正偏差第 90 百分位。
Footing：底部 82~100% 相对 65~80% 柱身线性外推的正偏差第 90 百分位。
中间细/粗：中段 35~65% 相对 20~30%、70~80% 两肩宽度连线的负/正偏差第 90 百分位。
Top-rounding：顶部 0~20% 相对 15~35% 柱身线性外推的缺失截面积 A，等效半径 sqrt(A / (2*(1-π/4)))。
等效半径是轮廓面积指标，不是直接拟合的圆半径；严重 T-top、倾斜或遮挡时请结合轮廓图判断。
LER：中间 15~85% 侧壁 x(y) 去线性趋势后的 3σ，左右分别计算；LWR 对同一高度段的 R(y)-L(y) 去线性趋势后计算 3σ。
粗糙度主比较取逐柱原始采样测量值的均值，另存平均坐标轮廓的粗糙度，避免平均时互相抵消。
周期起伏：中间侧壁去二次趋势、Hann 窗频谱找至少 2 周期且每周期至少 4 个采样点的主峰；主峰邻域能量占比≥45% 后拟合正弦幅值。
周期起伏是图像中的周期成分，不等同于已证实的光学驻波；无显著周期时幅值为 0，周期留空。
形貌出现数量以量值≥1 nm 计数，仅作统一统计阈值；所有原始数值（含 0）均导出。
检测触及图像/ROI 边界的残柱仍显示逐柱值和截断提示，但不参与平均形貌 baseline；无完整柱时禁止比较。
比例尺来自用户输入，跨图比较可分别设置 nm/px；滤波、像素尺寸和测量长度会影响粗糙度，结果未作噪声去偏。
Δ=样本-baseline；百分比=Δ/abs(baseline)*100；baseline 为 0 时百分比留空，不显示无穷大。
"""


def _finite_number(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _detrended(values, coordinate, degree=1):
    values, coordinate = np.asarray(values, float), np.asarray(coordinate, float)
    centered = coordinate - np.mean(coordinate)
    return values - np.polyval(np.polyfit(centered, values, degree), centered)


def _periodic_edge_metrics(y, left, right):
    if len(y) < 24 or y[-1] <= y[0]:
        return None, None
    dt = float(np.median(np.diff(y)))
    signals = [_detrended(v, y, 2) for v in (left, right)]
    window = np.hanning(len(y))
    freq = np.fft.rfftfreq(len(y), dt)
    power = sum(np.abs(np.fft.rfft(v * window)) ** 2 for v in signals)
    eligible = (freq >= 2.0 / (y[-1] - y[0])) & (freq <= 1.0 / (4 * dt))
    if not np.any(eligible) or float(np.sum(power[eligible])) < 1e-12:
        return 0.0, None
    candidates = np.where(eligible)[0]
    peak = int(candidates[np.argmax(power[eligible])])
    concentration = np.sum(power[max(0, peak - 1):peak + 2]) / max(1e-12, np.sum(power[1:]))
    if concentration < .45:
        return 0.0, None
    phase = 2 * np.pi * freq[peak] * (y - y[0])
    t = (y - y.mean()) / max(1e-9, np.ptp(y))
    design = np.column_stack((np.ones(len(y)), t, t*t, np.sin(phase), np.cos(phase)))
    amplitudes = []
    for values in (left, right):
        coefs = np.linalg.lstsq(design, values, rcond=None)[0]
        amplitudes.append(float(np.hypot(coefs[-2], coefs[-1])))
    return float(np.mean(amplitudes)), float(1.0 / freq[peak])


def contour_morphology(y, left, right, valid_left=None, valid_right=None):
    """从物理坐标计算数值；用于逐柱原生采样和平均轮廓，避免两套公式。"""
    out = {key: None for key, _, _ in MORPHOMETRY}
    y, left, right = (np.asarray(v, dtype=float) for v in (y, left, right))
    if len(y) < 8 or y[-1] <= y[0] or not np.all(np.isfinite(np.r_[y, left, right])):
        return out
    height = float(y[-1] - y[0])
    z = (y - y[0]) / height
    width = right - left
    out.update(height_nm=height, cd_mean_nm=float(np.mean(width)))

    def reference(lo, hi):
        mask = (z >= lo) & (z <= hi)
        return np.polyval(np.polyfit(z[mask], width[mask], 1), z) if np.count_nonzero(mask) >= 2 else width

    def positive_q90(values):
        return max(0.0, float(np.percentile(values, 90))) if len(values) else None

    out["t_top_nm"] = positive_q90((width - reference(.20, .35))[z <= .18])
    out["footing_nm"] = positive_q90((width - reference(.65, .80))[z >= .82])
    upper, lower = (z >= .20) & (z <= .30), (z >= .70) & (z <= .80)
    if np.any(upper) and np.any(lower):
        shoulder = np.interp(z, [float(np.mean(z[upper])), float(np.mean(z[lower]))],
                             [float(np.mean(width[upper])), float(np.mean(width[lower]))])
        middle = (z >= .35) & (z <= .65)
        out["middle_thin_nm"] = positive_q90((shoulder - width)[middle])
        out["middle_thick_nm"] = positive_q90((width - shoulder)[middle])
    cap = z <= .20
    deficit = np.maximum(0.0, reference(.15, .35)[cap] - width[cap])
    area = float(np.trapezoid(deficit, y[cap]))
    out["top_rounding_nm"] = math.sqrt(max(0.0, area) / (2.0 * (1.0 - np.pi / 4.0)))
    body = (z >= .15) & (z <= .85)
    vl = np.ones(len(y), bool) if valid_left is None else np.asarray(valid_left, bool)
    vr = np.ones(len(y), bool) if valid_right is None else np.asarray(valid_right, bool)
    for key, values, valid in (("ler_left_nm", left, vl), ("ler_right_nm", right, vr),
                               ("lwr_nm", width, vl & vr)):
        mask = body & valid
        if mask.sum() >= 8 and mask.sum() >= .7 * body.sum():
            out[key] = 3.0 * float(np.std(_detrended(values[mask], y[mask])))
    if out["ler_left_nm"] is not None and out["ler_right_nm"] is not None:
        out["ler_nm"] = .5 * (out["ler_left_nm"] + out["ler_right_nm"])
    # 频谱不跨缺失点插值，避免把缺口伪装为周期信号。
    if np.all((vl & vr)[body]):
        out["wave_amplitude_nm"], out["wave_period_nm"] = _periodic_edge_metrics(y[body], left[body], right[body])
    return {key: (0.0 if value is not None and abs(value) < 1e-6 else value) for key, value in out.items()}


def morphology_labels(metrics):
    return "+".join(label for key, label in MORPH_SHAPES
                    if _finite_number(metrics.get(key)) and metrics[key] >= 1.0) or "未达1nm阈值"


def build_morphology(columns, nm_per_px):
    if not _finite_number(nm_per_px) or nm_per_px <= 0:
        raise ValueError("比例尺 nm/px 必须为正数")
    per_column, aligned_left, aligned_right, heights = [], [], [], []
    z_grid = np.linspace(0.0, 1.0, PROFILE_SAMPLES)
    for col in columns:
        rr = np.asarray(col["ranges"], dtype=float)
        if len(rr) < 8 or rr[-1, 0] <= rr[0, 0]:
            continue
        y, left, right = rr.T * nm_per_px
        y = y - y[0]
        z = y / y[-1]
        center = float(np.mean(((left + right) * .5)[(z >= .35) & (z <= .65)]))
        left, right = left - center, right - center
        validity = {}
        for side in ("left", "right"):
            by_row = {int(round(p["y"])): bool(p["valid"]) for p in col.get("points", []) if p["side"] == side}
            validity[side] = np.array([by_row.get(int(round(row)), not by_row) for row in rr[:, 0]])
        metrics = contour_morphology(y, left, right, validity["left"], validity["right"])
        clipped = bool(col.get("clipped", False))
        eligible = not clipped and float(col.get("edge_support", 1.0)) >= .6
        record = {"label": f"PR{col['label']}", "metrics": metrics, "included": eligible,
                  "reason": "图像/ROI 边界截断" if clipped else ("边缘证据不足" if not eligible else ""),
                  "top_source": col.get("top_source"), "bottom_source": col.get("bottom_source")}
        per_column.append(record)
        if eligible:
            aligned_left.append(np.interp(z_grid, z, left))
            aligned_right.append(np.interp(z_grid, z, right))
            heights.append(float(y[-1]))
    summary = {}
    included = [c for c in per_column if c["included"]]
    for key, _, _ in MORPHOMETRY:
        values = [c["metrics"][key] for c in included if _finite_number(c["metrics"].get(key))]
        summary[key] = {"mean": float(np.mean(values)) if values else None,
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "min": min(values) if values else None, "max": max(values) if values else None,
                        "n": len(values), "above_1nm": sum(v >= 1.0 for v in values)}
    profile, average_metrics = None, {key: None for key, _, _ in MORPHOMETRY}
    if heights:
        avg_y = z_grid * float(np.mean(heights))
        avg_l, avg_r = np.mean(aligned_left, axis=0), np.mean(aligned_right, axis=0)
        average_metrics = contour_morphology(avg_y, avg_l, avg_r)
        profile = {"z": z_grid.tolist(), "y_nm": avg_y.tolist(), "left_nm": avg_l.tolist(),
                   "right_nm": avg_r.tolist(), "width_nm": (avg_r - avg_l).tolist(),
                   "left_std_nm": np.std(aligned_left, axis=0).tolist(),
                   "right_std_nm": np.std(aligned_right, axis=0).tolist()}
    metrics = dict(average_metrics)
    for key in ROUGHNESS_KEYS:
        metrics[key] = summary[key]["mean"]
    return {"metrics": metrics, "mean_profile_metrics": average_metrics, "summary": summary,
            "mean_profile": profile, "per_column": per_column, "included_count": len(included),
            "excluded_count": len(per_column) - len(included), "nm_per_px": float(nm_per_px),
            "definition": MORPHOLOGY_DEFINITION}


def square_baseline(cd_nm, height_nm):
    if not all(_finite_number(v) and v > 0 for v in (cd_nm, height_nm)):
        raise ValueError("标准方波的 CD 和高度必须是正数（nm）")
    y = np.linspace(0.0, height_nm, PROFILE_SAMPLES)
    col = {"label": 1, "ranges": list(zip(y, np.full(len(y), -cd_nm / 2), np.full(len(y), cd_nm / 2)))}
    result = build_morphology([col], 1.0)
    result.update(name=f"方波 CD={cd_nm:g}nm H={height_nm:g}nm", kind="square", nm_per_px=None)
    return result


def compare_morphologies(baseline, samples):
    if not baseline.get("included_count"):
        raise ValueError("baseline 没有完整且可信的柱子；请调整 ROI")
    # 报告编号与名称分离，同名文件、不同目录及不同 ROI 均保留独立身份。
    baseline = dict(baseline, sample_id="B0")
    samples = [dict(sample, sample_id=f"S{i}", color_index=i-1) for i, sample in enumerate(samples, 1)]
    rows = []
    for sample in samples:
        if not sample.get("included_count"):
            raise ValueError(f"{sample.get('name', '样本')} 没有完整柱，无法比较")
        for key, title, unit in MORPHOMETRY:
            ref, value = baseline["metrics"].get(key), sample["metrics"].get(key)
            delta = float(value - ref) if _finite_number(value) and _finite_number(ref) else None
            pct = 100.0 * delta / abs(ref) if delta is not None and abs(ref) > 1e-8 else None
            stats = sample["summary"].get(key, {})
            rows.append({"sample_id": sample["sample_id"], "sample": sample["name"],
                         "source": sample.get("source", ""), "mode": sample.get("mode", ""),
                         "metric": key, "title": title, "unit": unit,
                         "baseline": ref, "value": value, "delta": delta, "percent": pct,
                         "basis": "逐柱原生采样均值" if key in ROUGHNESS_KEYS else "平均坐标轮廓",
                         "pillar_mean": stats.get("mean"), "pillar_std": stats.get("std"),
                         "n": stats.get("n", 0), "above_1nm": stats.get("above_1nm", 0),
                         "mean_profile_value": sample["mean_profile_metrics"].get(key)})
    return {"baseline": baseline, "samples": samples, "rows": rows,
            "definition": MORPHOLOGY_DEFINITION, "created_at": datetime.now().isoformat()}


def _sample_caption(sample, limit=25):
    name = sample["name"].replace("\n", " ")
    if len(name) > limit:
        name = name[:limit-1] + "…"
    return f"{sample['sample_id']} · {name}"


def draw_baseline_figure(report, figure=None, sample_id=None, delta_view="auto"):
    """同一总览内切换点柱图/表格；auto 保留导出按样本数量选择的布局。"""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("Baseline 轮廓图需要 Matplotlib")
    samples = [s for s in report["samples"] if sample_id is None or s["sample_id"] == sample_id]
    fig = figure or Figure(figsize=(15, 6.4), dpi=150)
    fig.clear()
    fig.set_facecolor("#ffffff")
    contour, widths, changes = fig.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 1.55]})
    colors = matplotlib.colormaps["tab10"]
    for ax in (contour, widths, changes):
        ax.set_facecolor("#f8fafc")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#cbd5e1")
        ax.tick_params(labelsize=8, colors="#475569")
        ax.set_axisbelow(True)
    ref = report["baseline"]
    for sample in [ref] + samples:
        profile = sample.get("mean_profile")
        if profile is None:
            continue
        is_base = sample["sample_id"] == "B0"
        color = "#26364a" if is_base else colors(sample.get("color_index", 0) % 10)
        style = dict(color=color, linestyle="--" if is_base else "-", linewidth=2 if is_base else 1.7,
                     alpha=1 if is_base or len(samples) <= 3 else .8)
        label = _sample_caption(sample, 23 if len(samples) <= 8 else 12)
        contour.plot(profile["left_nm"], profile["y_nm"], label=label, **style)
        contour.plot(profile["right_nm"], profile["y_nm"], **style)
        for end in (0, -1):
            contour.plot([profile["left_nm"][end], profile["right_nm"][end]], [profile["y_nm"][end]] * 2, **style)
        if not is_base and len(samples) == 1:
            for side in ("left", "right"):
                x, sd = np.asarray(profile[side+"_nm"]), np.asarray(profile[side+"_std_nm"])
                contour.fill_betweenx(profile["y_nm"], x-sd, x+sd, color=color, alpha=.14, linewidth=0)
        widths.plot(profile["width_nm"], np.asarray(profile["z"]) * 100, **style)
    contour.axvline(0, color="#cbd5e1", linewidth=.8, zorder=0)
    contour.set(xlabel="中心对齐 x (nm)", ylabel="距顶部 y (nm)", title="平均轮廓 · 保留真实宽高")
    contour.set_aspect("equal", adjustable="box")
    widths.set(xlabel="CD (nm)", ylabel="距顶部的相对高度 (%)", title="CD 随高度变化")
    for ax in (contour, widths):
        ax.invert_yaxis()
        ax.grid(alpha=.35, color="#cbd5e1")
        ax.title.set(fontsize=11, fontweight="bold", color="#20334a")
    if delta_view == "auto":
        delta_view = "bars" if len(samples) <= 1 else "table"
    _draw_baseline_changes(changes, samples, ref, delta_view, compact=True)
    handles, legend_labels = contour.get_legend_handles_labels()
    columns = min(4, max(1, len(legend_labels)))
    legend_rows = math.ceil(len(legend_labels)/columns)
    legend_space = min(.42, .095+.045*legend_rows)
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5, .05), ncol=columns,
               frameon=False, fontsize=8, handlelength=2.5)
    fig.text(.5, .018, "几何量：平均坐标轮廓；LER / LWR：逐柱均值。增减不代表合格与否。" +
             (" 单样本轮廓阴影为柱间 ±1σ。" if len(samples) == 1 else " 同编号对应同一样本。"),
             ha="center", fontsize=8, color="#64748b")
    fig.tight_layout(rect=(.008, legend_space, .99, .99), w_pad=2.3)
    return fig



def _draw_baseline_changes(ax, samples, ref, view, compact=False):
    """总览与独立导出共用形貌 Δ 数据和绘制逻辑。"""
    labels = [label for _, label in MORPH_SHAPES]
    values = np.array([[sample["metrics"].get(key) - ref["metrics"].get(key)
                        if _finite_number(sample["metrics"].get(key)) and _finite_number(ref["metrics"].get(key))
                        else np.nan for key, _ in MORPH_SHAPES] for sample in samples], dtype=float)
    ax.set_facecolor("#f8fafc")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_color("#cbd5e1")
    ax.tick_params(labelsize=9,colors="#475569")
    ax.set_axisbelow(True)
    if view == "table":
        matrix = values.T
        finite = matrix[np.isfinite(matrix)]
        extent = max(.01, float(np.max(np.abs(finite))) if finite.size else .01)
        ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r", vmin=-extent, vmax=extent, aspect="auto")
        ax.set_xticks(range(len(samples)), [sample["sample_id"] for sample in samples])
        ax.set_yticks(range(len(labels)), labels)
        for j in range(len(labels)):
            for i in range(len(samples)):
                value = matrix[j, i]
                ax.text(i, j, f"{value:+.2f}" if np.isfinite(value) else "—", ha="center", va="center",
                        fontsize=max(6, 9-len(samples)//6),
                        color="white" if np.isfinite(value) and abs(value) > .55*extent else "#253449")
        ax.set_xlabel("Δ (nm) · 红色增加 / 蓝色减少 / 白色接近 0")
        ax.set_xticks(np.arange(-.5, len(samples), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title("形貌增减 · 表格", fontsize=11, fontweight="bold", color="#20334a", pad=14)
        return
    finite=values[np.isfinite(values)]
    extent=max(1.0,float(np.max(np.abs(finite))) if finite.size else 1.0)
    n=max(1,len(samples))
    spacing=.72/n
    height=min(.42,spacing*.72)
    label_size = max(6, 8 - (n-1)*.5) if compact else 8
    for j in range(len(labels)):
        if j%2==0: ax.axhspan(j-.5,j+.5,color="#edf2f7",zorder=0)
    for i,sample in enumerate(samples):
        color=matplotlib.colormaps["tab10"](sample.get("color_index",i)%10)
        for j,value in enumerate(values[i]):
            y=j+(i-(n-1)/2)*spacing
            if not math.isfinite(value):
                ax.text(-extent*.96,y,f"{sample['sample_id']}  —",va="center",ha="left",fontsize=label_size,color="#64748b")
                continue
            ax.barh(y,value,height=height,color=color,alpha=.82,edgecolor="white",linewidth=.5,zorder=3)
            ax.plot(value,y,"o",color=color,markeredgecolor="white",markeredgewidth=.7,markersize=5,zorder=4)
            if n<=5:
                tone="#b45309" if value>1e-8 else "#1d4ed8" if value< -1e-8 else "#64748b"
                ax.annotate(f"{sample['sample_id']} {value:+.2f}",(value,y),
                            xytext=(5 if value>=0 else -5,0),textcoords="offset points",
                            ha="left" if value>=0 else "right",va="center",fontsize=label_size,
                            color=tone,fontweight="bold")
    ax.axvline(0,color="#475569",linewidth=1.1,zorder=2)
    ax.set_xlim(-extent*1.55,extent*1.55)
    ax.set_yticks(range(len(labels)),labels)
    ax.set_ylim(len(labels)-.5,-.5)
    ax.set_xlabel("Δ (nm)   ← 减少  |  增加 →")
    ax.set_title("形貌增减 · 点柱状图",fontsize=11 if compact else 13,fontweight="bold",color="#20334a",pad=14)
    ax.grid(axis="x",color="#cbd5e1",alpha=.42)


def draw_baseline_delta_bars(report, figure=None, sample_id=None):
    """保留独立点柱图导出；与总览切换视图共用绘制逻辑。"""
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("Baseline 轮廓图需要 Matplotlib")
    samples = [sample for sample in report["samples"] if sample_id is None or sample["sample_id"] == sample_id]
    fig = figure or Figure(figsize=(12.5, 7.0), dpi=150)
    fig.clear()
    ax = fig.add_subplot(111)
    _draw_baseline_changes(ax, samples, report["baseline"], "bars")
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=matplotlib.colormaps["tab10"](x.get("color_index",i)%10),
                    markersize=8,label=_sample_caption(x,40)) for i,x in enumerate(samples)]
    if handles:
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.025),
                   ncol=min(4,len(handles)),frameon=False,fontsize=8)
    fig.text(.5,.008,"Δ=样本−B0；单位nm。点旁编号与下方样本来源一一对应。",
             ha="center",fontsize=8,color="#64748b")
    fig.tight_layout(rect=(.02,.10 if handles else .04,.99,.98))
    return fig


def export_baseline_report(report, folder):
    os.makedirs(folder, exist_ok=True)
    paths = {}
    fields = ["sample_id", "sample", "source", "mode", "title", "unit", "baseline", "value", "delta", "percent", "basis",
              "pillar_mean", "pillar_std", "n", "above_1nm", "mean_profile_value", "metric"]

    def write_rows(key, name, rows, headers=None):
        path = paths[key] = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            if headers is None:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            else:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

    write_rows("csv", "baseline_comparison.csv", report["rows"])
    all_samples = [report["baseline"]] + report["samples"]
    write_rows("index", "sample_index.csv", ([s["sample_id"], s["name"], s.get("source", "标准方波"),
               s.get("mode", "理想轮廓"), s.get("selection", "平均轮廓"), s.get("nm_per_px"),
               s["included_count"], s["excluded_count"]] for s in all_samples),
               ["sample_id", "sample", "source", "mode", "selection", "nm_per_px", "included_count", "excluded_count"])
    by_metric = {(r["sample_id"], r["metric"]): r for r in report["rows"]}
    for field, name, label in (("value", "comparison_values.csv", "实测值"),
                               ("delta", "comparison_deltas.csv", "增减量"),
                               ("percent", "comparison_percent.csv", "变化率")):
        write_rows(field, name, ([title, "%" if field == "percent" else unit,
                   report["baseline"]["metrics"].get(key),
                   *[by_metric[(s["sample_id"], key)][field] for s in report["samples"]]]
                   for key, title, unit in MORPHOMETRY),
                   ["指标", "样本列单位", "B0 baseline (nm)", *[_sample_caption(s, 100) for s in report["samples"]]])
    paths["json"] = os.path.join(folder, "baseline_comparison.json")
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    profile_rows, pillar_rows = [], []
    for i, sample in enumerate(all_samples):
        identity = ["baseline" if i == 0 else "sample", sample["sample_id"], sample["name"],
                    sample.get("source", ""), sample.get("mode", "")]
        p = sample["mean_profile"]
        if p:
            for values in zip(*(p[key] for key in ("z", "y_nm", "left_nm", "right_nm", "width_nm", "left_std_nm", "right_std_nm"))):
                profile_rows.append([*identity, *values])
        for col in sample["per_column"]:
            pillar_rows.append([*identity, col["label"], col["included"], col["reason"],
                                *[col["metrics"].get(key) for key, _, _ in MORPHOMETRY]])
    identity_fields = ["role", "sample_id", "sample", "source", "mode"]
    write_rows("profiles", "average_coordinates.csv", profile_rows,
               identity_fields + ["relative_height", "y_nm", "left_nm", "right_nm", "width_nm", "left_std_nm", "right_std_nm"])
    pillar_fields = identity_fields + ["pillar", "included", "reason", *[key for key, _, _ in MORPHOMETRY]]
    write_rows("pillars", "per_pillar_morphology.csv", pillar_rows, pillar_fields)
    paths["readme"] = os.path.join(folder, "README.txt")
    with open(paths["readme"], "w", encoding="utf-8") as f:
        f.write("Baseline：" + _sample_caption(report["baseline"], 100) + "\n\n")
        f.write("先看 sample_index.csv 对应编号、文件路径与区域；同名样本不会合并。\n")
        f.write("comparison_values/deltas/percent.csv 为横向比较，每个样本一列。\n")
        f.write("baseline_comparison.csv 为全部指标长表；per_sample/ 为每个样本单独的指标、逐柱表和对比图。\n")
        f.write("baseline_overlay.png 为总览；baseline_delta_bars.png 为独立横向点柱图。\n")
        f.write("空白表示缺失；0 是有效的零值。基准为零时只给绝对差值。\n\n" + FORMULA_HELP + "\n\n" + PROFILE_FORMULA_HELP)
    if MATPLOTLIB_AVAILABLE:
        paths["plot"] = os.path.join(folder, "baseline_overlay.png")
        fig = draw_baseline_figure(report)
        fig.savefig(paths["plot"])
        fig.clear()
        paths["bar_plot"] = os.path.join(folder, "baseline_delta_bars.png")
        bar_fig = draw_baseline_delta_bars(report)
        bar_fig.savefig(paths["bar_plot"])
        bar_fig.clear()
    os.makedirs(os.path.join(folder, "per_sample"), exist_ok=True)
    for sample in report["samples"]:
        sid = sample["sample_id"]
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample["name"]).strip("_")[:48] or "sample"
        stem = os.path.join("per_sample", f"{sid}_{safe}")
        write_rows(f"{sid}_metrics", stem + "_metrics.csv", [r for r in report["rows"] if r["sample_id"] == sid])
        write_rows(f"{sid}_pillars", stem + "_pillars.csv", [r for r in pillar_rows if r[1] == sid], pillar_fields)
        if MATPLOTLIB_AVAILABLE:
            paths[f"{sid}_plot"] = os.path.join(folder, stem + "_comparison.png")
            fig = draw_baseline_figure(report, sample_id=sid)
            fig.savefig(paths[f"{sid}_plot"])
            fig.clear()
            paths[f"{sid}_bar_plot"] = os.path.join(folder, stem + "_delta_bars.png")
            bar_fig = draw_baseline_delta_bars(report, sample_id=sid)
            bar_fig.savefig(paths[f"{sid}_bar_plot"])
            bar_fig.clear()
    return paths


def export_baseline_for_app(report):
    """对比图和完整报告固定保存到软件同级目录。"""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = os.path.join(application_dir(), "baseline_comparison", run_id)
    report["output_folder"] = folder
    report["software_version"] = APP_VERSION
    export_baseline_report(report, folder)
    return folder


def _num(v, signed=False):
    if v is None:
        return "-"
    try:
        f = float(v)
    except Exception:
        return "-"
    if not math.isfinite(f):
        return "-"
    return f"{f:+.3f}" if signed else f"{f:.3f}"


def _cd_width_summary(values, source="observed"):
    """将一组轮廓宽度整理为可写入输出的摘要。

    所有用户可见的 Line CD、候选先验、观测范围和统计 CD 统一使用
    ``x_end - x_start`` 的边界距离；仅内部连通像素数量可能使用含端点
    的格数。这样手动量测的 px/nm 与自动输出可以直接对照。
    """
    vals = np.asarray(list(values or []), dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return None
    center = float(np.median(vals))
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "center": center,
        "tolerance": float(max(center - float(np.min(vals)),
                               float(np.max(vals)) - center)),
        "count": int(vals.size),
        "source": str(source),
    }


def _observed_cd_breakdown(columns, image_width=None):
    """返回主体柱、边缘残柱和全部柱的 Line CD 范围。

    ``_observed_cd_range`` 为保持旧调用仍返回主体优先的摘要；这个补充
    结构只用于输出，给用户同时展示被排除的边缘残柱，避免“实际范围”
    看起来比边界文件中的最窄柱宽得多。
    """
    cols = list(columns or [])
    if not cols:
        return {"primary": None, "complete": None, "edge": None, "all": None}

    if image_width is None:
        image_width = max(
            (int(c.get("right", 0)) for c in cols),
            default=0,
        ) + 1
    image_width = int(image_width)

    all_widths = []
    complete_widths = []
    edge_widths = []
    for col in cols:
        width = _column_width_px(col)
        if not math.isfinite(float(width)) or width <= 0:
            continue
        all_widths.append(width)
        is_edge = (
            int(col.get("left", 0)) <= 1 or
            int(col.get("right", 0)) >= image_width - 2
        )
        (edge_widths if is_edge else complete_widths).append(width)

    complete = _cd_width_summary(complete_widths, "complete")
    edge = _cd_width_summary(edge_widths, "edge")
    all_summary = _cd_width_summary(all_widths, "all")
    primary = complete if complete is not None and len(complete_widths) >= 2 else all_summary
    return {
        "primary": primary,
        "complete": complete,
        "edge": edge,
        "all": all_summary,
    }


def _add_cd_protection_range(rng):
    """给用户/自动先验摘要补充实际候选保护带（不改变检测逻辑）。"""
    if not isinstance(rng, dict):
        return rng
    out = dict(rng)
    try:
        center = float(out.get("center"))
        tolerance = max(1.0, float(out.get("tolerance")))
    except Exception:
        return out
    out["protection_min"] = float(max(3.0, center - 2.0 * tolerance))
    out["protection_max"] = float(center + 2.0 * tolerance)
    out["protection_tolerance"] = float(2.0 * tolerance)
    return out


def _detection_meta_from_aux(aux=None, columns=None, image_width=None):
    """整理识别元数据，供 payload 与文本输出共同使用。

    识别核心会把中间结果写入 ``aux``；这里统一成稳定、可序列化的
    字典，避免 GUI、边界文件和统计文件各自重复解释字段。

    ``image_width`` 可用于 ROI 结果：ROI 轮廓在导出前会偏移回整图坐标，
    此时 ``aux['gray']`` 仍是局部 ROI 图，不能再用它判断哪根柱贴着整图
    边缘。未显式提供时保持旧行为，优先从 ``aux`` 中的图像推断宽度。
    """
    aux = aux if isinstance(aux, dict) else {}

    # 元数据可能在 ROI 局部坐标检测后才生成；显式传入的整图宽度优先，
    # 防止偏移后的全局 x 坐标与局部灰度图宽度混用。
    try:
        image_width = int(image_width) if image_width is not None else None
    except Exception:
        image_width = None

    def infer_image_width():
        if image_width is not None and image_width > 0:
            return image_width
        source_image = aux.get("gray")
        if source_image is None:
            source_image = aux.get("img")
        try:
            shape = np.asarray(source_image).shape
            return int(shape[1]) if len(shape) >= 2 else None
        except Exception:
            return None

    def finite_or(value, default=None):
        try:
            value = float(value)
            return value if math.isfinite(value) else default
        except Exception:
            return default

    meta = {
        "mode": str(aux.get("detection_mode", aux.get("mode", "unknown"))),
        "variant": str(aux.get("detection_variant", aux.get("variant", "unknown"))),
        "confidence": finite_or(
            aux.get("detection_confidence", aux.get("confidence")), None
        ),
        "attempts": int(max(0, round(finite_or(
            aux.get("detection_attempts", aux.get("attempts")), 0
        ) or 0))),
        "column_count": int(len(columns or [])),
        "detected_count": int(aux.get("detected_count", len(columns or []))),
        "excluded_count": int(aux.get("excluded_count", len(aux.get("excluded_columns", [])))),
        "exclude_outermost": bool(aux.get("exclude_outermost", False)),
        "warning": str(aux.get("detection_warning", aux.get("warning", "")) or ""),
        "polarity": aux.get("selected_polarity", aux.get("polarity", PILLAR_POLARITY)),
    }

    hint = aux.get("cd_hint_px")
    if isinstance(hint, dict):
        estimate = finite_or(hint.get("estimate"), None)
        tolerance = finite_or(hint.get("tolerance"), None)
        if estimate is not None and tolerance is not None:
            meta["cd_hint_px"] = {
                "estimate": estimate,
                "tolerance": max(0.0, tolerance),
            }
    else:
        meta["cd_hint_px"] = None

    rng = aux.get("cd_range_px")
    if not isinstance(rng, dict) and columns:
        try:
            rng = _reported_cd_range(
                columns, image_width=infer_image_width()
            )
        except Exception:
            rng = None
    if isinstance(rng, dict):
        vals = {
            key: finite_or(rng.get(key), None)
            for key in ("min", "max", "center", "tolerance")
        }
        if all(v is not None for v in vals.values()):
            vals["source"] = str(rng.get("source", "unknown"))
            meta["cd_range_px"] = _add_cd_protection_range(vals)
        else:
            meta["cd_range_px"] = None
    else:
        meta["cd_range_px"] = None

    observed = aux.get("cd_observed_px")
    if not isinstance(observed, dict) and columns:
        try:
            observed = _observed_cd_range(
                columns, image_width=infer_image_width()
            )
        except Exception:
            observed = None
    if isinstance(observed, dict):
        vals = {
            key: finite_or(observed.get(key), None)
            for key in ("min", "max", "center", "tolerance")
        }
        if all(v is not None for v in vals.values()):
            vals["source"] = str(observed.get("source", "observed"))
            # 保留旧的主体优先 min/max 字段，同时追加主体/边缘/全部柱
            # 的明细，便于解释为什么边缘残柱不参与典型 CD 估计。
            existing_breakdown = {
                key: observed.get(key)
                for key in ("complete", "edge", "all")
                if isinstance(observed.get(key), dict)
            }
            # 当调用方显式提供整图宽度（ROI 轮廓已偏移回全局坐标）时，
            # 即使 aux 中已有局部 breakdown，也要按整图边界重新分类。
            if existing_breakdown and image_width is None:
                breakdown = {
                    "complete": existing_breakdown.get("complete"),
                    "edge": existing_breakdown.get("edge"),
                    "all": existing_breakdown.get("all"),
                    "primary": existing_breakdown.get("complete")
                    or existing_breakdown.get("all"),
                }
            else:
                gray_for_shape = aux.get("gray")
                breakdown = _observed_cd_breakdown(
                    columns,
                    image_width=(
                        image_width
                        if image_width is not None and image_width > 0
                        else (
                            int(np.asarray(gray_for_shape).shape[1])
                            if gray_for_shape is not None
                            and np.asarray(gray_for_shape).ndim >= 2 else None
                        )
                    ),
                )
            if breakdown.get("primary") is not None:
                primary = breakdown["primary"]
                # 检测路径传入的主体摘要优先保留，以兼容旧结果；
                # 新增字段只补充输出解释。
                for key in ("min", "max", "center", "tolerance"):
                    if key not in vals or vals[key] is None:
                        vals[key] = primary.get(key)
            vals["complete"] = breakdown.get("complete")
            vals["edge"] = breakdown.get("edge")
            vals["all"] = breakdown.get("all")
            vals["complete_count"] = int(
                (breakdown.get("complete") or {}).get("count", 0)
            )
            vals["edge_count"] = int(
                (breakdown.get("edge") or {}).get("count", 0)
            )
            vals["all_count"] = int(
                (breakdown.get("all") or {}).get("count", 0)
            )
            meta["cd_observed_px"] = vals
        else:
            meta["cd_observed_px"] = None
    else:
        meta["cd_observed_px"] = None

    return meta


def _detection_meta_rows(meta):
    """返回适合追加到文本表格的识别参数行。"""
    if not isinstance(meta, dict):
        return []

    def f(value, nd=3):
        try:
            value = float(value)
            return f"{value:.{nd}f}" if math.isfinite(value) else "-"
        except Exception:
            return "-"

    mode = meta.get("mode", "unknown")
    variant = meta.get("variant", "unknown")
    confidence = (
        f(meta.get("confidence"), 3)
        + "（候选排序分数，非百分比）"
    )
    attempts = meta.get("attempts", 0)
    try:
        attempts = str(max(0, int(attempts)))
    except Exception:
        attempts = "-"

    hint = meta.get("cd_hint_px")
    rng = meta.get("cd_range_px")
    range_label = "有效 CD 范围"
    if isinstance(hint, dict):
        estimate_text = f(hint.get("estimate"), 2) + " px（边界距离）"
        tolerance_text = "±" + f(hint.get("tolerance"), 2) + " px"
    elif isinstance(rng, dict):
        estimate_text = "自动估计（中心 " + f(rng.get("center"), 2) + " px，边界距离）"
        tolerance_text = "±" + f(rng.get("tolerance"), 2) + " px"
    else:
        estimate_text = "自动估计"
        tolerance_text = "-"

    if isinstance(rng, dict):
        range_source = str(rng.get("source", "unknown"))
        range_label = "用户先验范围" if range_source == "user" else "自动估计范围"
        range_rule = "用户先验±1×容差" if range_source == "user" else "自动中心±输入容差"
        range_text = (
            f"{f(rng.get('min'), 2)} ~ {f(rng.get('max'), 2)} px "
            f"（{range_rule}；来源：{range_source}）"
        )
        protection_text = (
            f"{f(rng.get('protection_min'), 2)} ~ "
            f"{f(rng.get('protection_max'), 2)} px "
            "（候选保护带±2×容差；边缘残柱/软候选可能进一步放宽）"
        )
    else:
        range_text = "-"
        protection_text = "-"

    observed = meta.get("cd_observed_px")
    if isinstance(observed, dict):
        observed_text = (
            f"中位 {f(observed.get('center'), 2)} px；"
            f"范围 {f(observed.get('min'), 2)} ~ {f(observed.get('max'), 2)} px "
            "（主体柱优先；宽度为边界距离）"
        )
        complete = observed.get("complete")
        all_cols = observed.get("all")
        edge = observed.get("edge")

        def observed_range_text(value, count_label):
            if not isinstance(value, dict):
                return None
            count = value.get("count", "?")
            return (
                f"中位 {f(value.get('center'), 2)} px；"
                f"范围 {f(value.get('min'), 2)} ~ {f(value.get('max'), 2)} px；"
                f"{count_label}={count}（边界距离）"
            )

        complete_text = observed_range_text(complete, "完整柱数")
        all_text = observed_range_text(all_cols, "总柱数")
        edge_text = observed_range_text(edge, "边缘残柱数")
    else:
        observed_text = "-"
        complete_text = all_text = edge_text = None

    rows = [
        ["识别模式", mode],
        ["候选版本", variant],
        ["识别置信度", confidence + "（候选排序分数，非准确率）"],
        ["候选尝试次数", attempts],
        ["两端排除规则", "自动识别强制排除最左/最右柱（完整柱也排除）" if meta.get("exclude_outermost") else "ROI 保留全部柱"],
        ["排除前 / 排除 / 保留", f"{meta.get('detected_count', 0)} / {meta.get('excluded_count', 0)} / {meta.get('column_count', 0)}"],
        ["估计 Line CD（边界距离）", estimate_text],
        ["允许误差（输入容差）", tolerance_text],
        [range_label, range_text],
        ["候选保护范围", protection_text],
        ["主体柱观测 CD", observed_text],
    ]
    if complete_text:
        rows.append(["完整柱观测 CD", complete_text])
    if all_text:
        rows.append(["全部柱观测 CD", all_text])
    if edge_text:
        rows.append(["边缘残柱观测 CD", edge_text])
    rows.append([
        "宽度口径",
        "Line CD/先验/观测范围/统计 CD 均为 xe-xs（边界距离）；内部像素格数可含端点",
    ])
    warning = str(meta.get("warning", "") or "").strip()
    if warning:
        rows.append(["识别提示", warning])
    return rows


STAT_ROWS = [
    ("height",        "高度 (nm)"),
    ("cd_top",        "上段平均 CD (nm)"),
    ("cd_mid",        "中段平均 CD (nm)"),
    ("cd_bot",        "下段平均 CD (nm)"),
    ("ratio_tm",      "CD 比值 上/中"),
    ("ratio_td",      "CD 比值 上/下"),
    ("ratio_md",      "CD 比值 中/下"),
    ("angle_l_top",   "上段侧壁角 左 (°)"),
    ("angle_r_top",   "上段侧壁角 右 (°)"),
    ("angle_l_mid",   "中段侧壁角 左 (°)"),
    ("angle_r_mid",   "中段侧壁角 右 (°)"),
    ("angle_l_bot",   "下段侧壁角 左 (°)"),
    ("angle_r_bot",   "下段侧壁角 右 (°)"),
    ("ler_left",      "左缘粗糙度 LER 3σ (nm)"),
    ("ler_right",     "右缘粗糙度 LER 3σ (nm)"),
]


def _mean_finite(values):
    vals = [
        float(v)
        for v in values
        if isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))
    ]
    return sum(vals) / len(vals) if vals else None


def analyze_columns(columns, nm_per_px):
    """统一统计引擎；保留尺寸/角度量测，所有形貌项目同时量化。"""
    labels, per_col, flags = [], [], []
    morphology = build_morphology(columns, nm_per_px)
    morph_by_label = {c["label"]: c for c in morphology["per_column"]}
    for col in columns:
        rows_nm = [(float(y)*nm_per_px, float(l)*nm_per_px, float(r)*nm_per_px) for y,l,r in col["ranges"]]
        if not rows_nm:
            continue
        label = f"PR{col['label']}"
        labels.append(label)
        measurements = analyze_pr_column(rows_nm)
        record = morph_by_label.get(label, {})
        metrics = record.get("metrics", {})
        measurements["ler_left"] = metrics.get("ler_left_nm")
        measurements["ler_right"] = metrics.get("ler_right_nm")
        measurements.update(metrics)
        measurements["height"] = metrics.get("height_nm")
        per_col.append(measurements)
        notes = [record["reason"]] if record.get("reason") else []
        if col.get("bottom_source") == "base_reference":
            notes.append("底部由柱外基座跃迁推断")
        if col.get("top_source") == "binary_fallback" or col.get("bottom_source") == "binary_fallback":
            notes.append("顶/底含 binary 估计，请核对")
        flags.append("；".join(notes))
    metrics_rows = [(key, f"{title} ({unit})") for key, title, unit in MORPHOMETRY
                    if key not in {"height_nm", "ler_left_nm", "ler_right_nm"}]
    descriptors = STAT_ROWS + metrics_rows
    rows = [[c.get(key) for c in per_col] for key, _ in descriptors]
    return {"desc": [title for _, title in descriptors], "cols": labels, "rows": rows,
            "means": [_mean_finite(v) for v in rows], "per_col": per_col, "flags": flags,
            "morphology": morphology}


def write_stats_file(out_txt_path, data, detection_meta=None):
    rows = [["统计项", "逐柱均值", *data["cols"]]]
    for title, values, mean in zip(data["desc"], data["rows"], data["means"]):
        rows.append([title, _num(mean), *[_num(v) for v in values]])
    morph = data["morphology"]
    rows += [["", ""], ["平均坐标形貌及逐柱离散", "平均轮廓/粗糙度均值", "逐柱标准差", "有效柱数", "≥1nm柱数"]]
    for key, title, unit in MORPHOMETRY:
        summary = morph["summary"][key]
        rows.append([f"{title} ({unit})", _num(morph["metrics"][key]), _num(summary["std"]),
                     str(summary["n"]), str(summary["above_1nm"])])
    for label, note in zip(data["cols"], data["flags"]):
        if note:
            rows.append([label + " 测量提示", note])
    morph_labels = [morphology_labels(c["metrics"]) for c in morph["per_column"]]
    rows.append(["形貌（可同时存在）", morphology_labels(morph["metrics"]), *morph_labels])
    if data.get("scale"):
        sc = data["scale"]
        rows.append(["本图比例尺", f"{sc['pixels']:g} px = {sc['nm']:g} nm；1 px = {sc['nm']/sc['pixels']:.8g} nm"])
    rows.extend(_detection_meta_rows(detection_meta or data.get("detection")))
    write_aligned_table(out_txt_path, rows, left_first_text=True, footer=MORPHOLOGY_DEFINITION)


def validate_scale(scale=None):
    scale = scale or {"pixels": SCALE_PIXELS, "nm": SCALE_NM}
    px, nm = float(scale["pixels"]), float(scale["nm"])
    if not all(math.isfinite(v) and v > 0 for v in (px, nm)):
        raise ValueError("比例尺像素长度和实际 nm 长度都必须为正数")
    return {"pixels": px, "nm": nm}


def image_key(path):
    return os.path.normcase(os.path.abspath(path))


def read_image_scale(image_path, fallback=None):
    """标定按图片名存于各自目录，同名不同目录互不影响。"""
    path = os.path.join(os.path.dirname(os.path.abspath(image_path)), IMAGE_SCALE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        value = data.get(os.path.basename(image_path))
        return validate_scale(value) if value else validate_scale(fallback)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return validate_scale(fallback)


def save_image_scale(image_path, scale):
    """只修改该图标定，原子替换 JSON；已有损坏配置不静默覆盖。"""
    scale = validate_scale(scale)
    path = os.path.join(os.path.dirname(os.path.abspath(image_path)), IMAGE_SCALE_FILE)
    data = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"标定文件格式错误：{path}")
    data[os.path.basename(image_path)] = scale
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temp, path)
    return scale


def _output_dir_for(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    base = os.path.dirname(os.path.abspath(image_path))
    out_dir = os.path.join(base, stem) if OUTPUT_SUBDIR else base
    os.makedirs(out_dir, exist_ok=True)
    return stem, out_dir


def _image_result_folder(image_path):
    source = os.path.abspath(image_path)
    folder = os.path.join(os.path.dirname(source), os.path.splitext(os.path.basename(source))[0])
    os.makedirs(folder, exist_ok=True)
    return folder


def export_detection_outputs(
    image_path,
    img,
    columns,
    structure_profile,
    tag,
    detection_meta=None,
    scale=None,
    output_dir=None,
):
    """
    tag="_tool" / "_user"
    返回 (px_main, stats_data, stats_path)
    """
    stem, out_dir = _output_dir_for(image_path) if output_dir is None else (os.path.splitext(os.path.basename(image_path))[0], output_dir)
    os.makedirs(out_dir, exist_ok=True)
    scale = validate_scale(scale or read_image_scale(image_path))
    nm_per_px = scale["nm"] / scale["pixels"]

    px_main = render_main_image(img, columns, 1.0)
    nm_main = render_main_image(img, columns, nm_per_px)

    prefix = f"{stem}{tag}"

    if tag == "_tool" and structure_profile is not None:
        combined_px = render_combined(
            img, columns, structure_profile,
            nm_per_px=1.0,
            main_img=px_main
        )
        combined_nm = render_combined(
            img, columns, structure_profile,
            nm_per_px=nm_per_px,
            main_img=nm_main
        )
        imwrite_unicode(os.path.join(out_dir, f"{prefix}_result.png"), combined_px)
        imwrite_unicode(os.path.join(out_dir, f"{prefix}_result_nm.png"), combined_nm)
    else:
        imwrite_unicode(os.path.join(out_dir, f"{prefix}_result.png"), px_main)
        imwrite_unicode(os.path.join(out_dir, f"{prefix}_result_nm.png"), nm_main)

    px_txt = os.path.join(out_dir, f"{prefix}_boundaries.txt")
    nm_txt = os.path.join(out_dir, f"{prefix}_boundaries_nm.txt")
    stats_path = os.path.join(out_dir, f"{prefix}_stats_nm.txt")

    meta = _detection_meta_from_aux(detection_meta, columns)
    export_boundaries(px_txt, columns, 1.0, detection_meta=meta)
    export_boundaries(nm_txt, columns, nm_per_px, detection_meta=meta)

    stats_data = analyze_columns(columns, nm_per_px)
    stats_data["detection"] = meta
    # 顶层别名方便表格/脚本直接读取；嵌套 detection 保留结构化字段。
    stats_data.update({
        "detection_mode": meta.get("mode", "unknown"),
        "detection_variant": meta.get("variant", "unknown"),
        "detection_confidence": meta.get("confidence"),
        "detection_attempts": meta.get("attempts", 0),
        "detection_warning": meta.get("warning", ""),
        "cd_hint_px": meta.get("cd_hint_px"),
        "cd_range_px": meta.get("cd_range_px"),
        "cd_observed_px": meta.get("cd_observed_px"),
    })
    stats_data["scale"] = scale
    write_stats_file(stats_path, stats_data, detection_meta=meta)
    morphology_path = os.path.join(out_dir, f"{prefix}_morphology.json")
    with open(morphology_path, "w", encoding="utf-8") as f:
        json.dump(stats_data["morphology"], f, ensure_ascii=False, indent=2, allow_nan=False)
    stats_data["morphology_path"] = morphology_path

    return px_main, stats_data, stats_path


class _ProcessImages:
    """各阶段完成即写入同一 test 目录，以运行标识避免整图/ROI/重跑覆盖。"""

    def __init__(self, image_path, tag):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        parent = _image_result_folder(image_path)
        self.image_path = image_path
        self.tag = tag
        self.run_id = f"{stem}{tag}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        self.directory = os.path.join(parent, "test")
        self.files = {}
        os.makedirs(self.directory, exist_ok=True)

    def path(self, name):
        stem, ext = os.path.splitext(name)
        filename = f"{stem}_{self.run_id}{ext}"
        self.files[name] = filename
        return os.path.join(self.directory, filename)

    def save(self, name, image):
        path = self.path(name)
        if not imwrite_unicode(path, image):
            raise OSError(f"无法写入过程图：{path}")


def export_process_images(output, img, aux, columns, offset=(0, 0)):
    """续写精定位、排除结果及全部扫描曲线，前五步已在识别时保存。"""
    if output is None:
        return None
    save = output.save
    selection_img = img.copy()
    for col in columns:
        cv2.drawContours(selection_img, [col["contour"]], -1, (0, 210, 0), 2)
    for col in aux.get("excluded_columns", []):
        cv2.drawContours(selection_img, [col["contour"]], -1, (0, 0, 255), 2)
        x = int(np.clip(col["center_x"] - 20, 0, max(0, img.shape[1] - 60)))
        cv2.putText(selection_img, "EXCLUDED", (x, max(15, col["y_top"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
    save("step6_outermost_exclusion.png", selection_img)
    point_img = img.copy()
    colors = {"left": (255, 190, 0), "right": (0, 200, 255), "top": (50, 220, 50),
              "bottom": (220, 80, 220), "base_reference": (220, 80, 220)}
    dx, dy = offset
    all_points = []
    for col in columns:
        for point in col.get("points", []):
            row = {"pillar": col["label"], **point}
            row["x_global"] = point["x"] + dx
            row["y_global"] = point["y"] + dy
            all_points.append(row)
            color = colors[point["side"]] if point["valid"] else (0, 255, 255)
            cv2.circle(point_img, (int(round(point["x"])), int(round(point["y"]))), 1, color, -1)
    save("step7_refined_points.png", point_img)
    save("step8_final_contours.png", render_main_image(img, columns))
    with open(output.path("points.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pillar", "side", "x", "y", "strength", "valid", "source", "x_global", "y_global"])
        writer.writeheader()
        writer.writerows(all_points)
    # 所有逐行/逐列灰度完整保存在矩阵；分页图遍历全部扫描行列，含选中点。
    g = aux["gray"]
    filtered = aux["filtered"]
    for col in columns:
        rr = np.asarray(col["ranges"], dtype=float)
        radius = col["refine_radius"]
        y0, y1 = int(rr[0, 0]), int(rr[-1, 0])
        x0 = max(0, int(np.floor(rr[:, 1].min())) - radius)
        x1 = min(g.shape[1] - 1, int(np.ceil(rr[:, 2].max())) + radius)
        yy0, yy1 = max(0, y0 - radius), min(g.shape[0] - 1, y1 + radius)
        np.savez_compressed(output.path(f"PR{col['label']}_profiles.npz"),
                            raw=g[yy0:yy1 + 1, x0:x1 + 1], filtered=filtered[yy0:yy1 + 1, x0:x1 + 1],
                            x=np.arange(x0, x1 + 1), y=np.arange(yy0, yy1 + 1), offset=np.array(offset), ranges=rr)
        groups = (("rows", range(y0, y1 + 1), x0, x1),
                  ("columns", range(int(np.floor(rr[:, 1].min())), int(np.ceil(rr[:, 2].max())) + 1), yy0, yy1))
        for axis, indices, start, end in groups:
            indices = list(indices)
            for page_start in range(0, len(indices), 24):
                page_indices = indices[page_start:page_start + 24]
                sheet = np.full((6 * 142, 4 * 300, 3), 248, dtype=np.uint8)
                for j, index in enumerate(page_indices):
                    ox, oy = (j % 4) * 300, (j // 4) * 142
                    values = g[index, start:end + 1] if axis == "rows" else g[start:end + 1, index]
                    smooth = filtered[index, start:end + 1] if axis == "rows" else filtered[start:end + 1, index]
                    def plot(values, color):
                        px = ox + 12 + np.arange(len(values)) * 274 / max(1, len(values) - 1)
                        py = oy + 120 - values.astype(float) * 88 / 255
                        cv2.polylines(sheet, [np.rint(np.column_stack((px, py))).astype(np.int32)], False, color, 1)
                    plot(values, (170, 170, 170))
                    plot(smooth, (180, 80, 30))
                    cv2.putText(sheet, f"PR{col['label']} {axis} {index} [{start}:{end}]", (ox + 8, oy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (45, 45, 45), 1)
                    for pt in col["points"]:
                        match = abs(pt["y"] - index) < 0.5 if axis == "rows" else abs(pt["x"] - index) < 0.5
                        along = pt["x"] if axis == "rows" else pt["y"]
                        if match and start <= along <= end:
                            px = ox + 12 + int(round((along - start) * 274 / max(1, end - start)))
                            color = colors[pt["side"]] if pt["valid"] else (0, 190, 190)
                            cv2.line(sheet, (px, oy + 28), (px, oy + 123), color, 1)
                    cv2.putText(sheet, "0-255 gray / blue filtered / marks edges", (ox + 8, oy + 136), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)
                step = 9 if axis == "rows" else 10
                save(f"step{step}_PR{col['label']}_{axis}_{page_start // 24 + 1:03d}.png", sheet)
    manifest = {"image": os.path.abspath(output.image_path), "tag": output.tag, "offset_xy": list(offset),
                "run_id": output.run_id, "status": "complete", "files": output.files,
                "coordinate_system": "local pixels; points CSV also includes global pixels",
                "polarity": aux.get("selected_polarity"), "warning": aux.get("detection_warning"),
                "algorithm": "binary ROI + signed Scharr + dynamic programming + subpixel peak",
                "candidates": aux.get("candidate_scores", []),
                "point_count": len(all_points), "column_count": len(columns),
                "detected_count": aux.get("detected_count", len(columns)),
                "exclude_outermost": aux.get("exclude_outermost", False),
                "excluded_columns": [{"detected_label": c["detected_label"], "center_x": c["center_x"],
                                      "reason": "outermost_left_or_right"} for c in aux.get("excluded_columns", [])],
                "columns": [{"label": c["label"], "edge_support": c["edge_support"],
                             "top_source": c["top_source"], "bottom_source": c["bottom_source"],
                             "top_subpixel": c["top_subpixel"], "bottom_subpixel": c["bottom_subpixel"],
                             "search_radius": c["refine_radius"]} for c in columns]}
    with open(output.path("README.txt"), "w", encoding="utf-8") as f:
        f.write(f"本次运行：{output.run_id}\n"
                "所有过程图直接保存在 test 文件夹，按相同运行标识筛选可查看本次结果。\n"
                "step1 原灰度；step2 保边滤波；step3 CLAHE；step4 选中 binary；step5 粗轮廓。\n"
                "step6 两端柱排除；step7 精定位点；step8 最终轮廓；step9 逐行曲线；step10 逐列曲线。\n"
                "黄色点为无足够梯度证据/裁切边界，保留 binary 位置；绿色为顶部，紫色为底部/基座参考。\n"
                "step9/step10 为每一行和每一列灰度曲线分页图，NPZ 保存完整原始及滤波灰度矩阵。\n"
                "points CSV 提供局部及全图坐标，valid=0 不应作为独立的精定位观测。\n"
                "base_reference 是柱外基座的测量点，柱底由其高度推断，未伪装成柱内可见界面。\n"
                "step6 红色为强制排除的两端柱，绿色为保留柱；统计、最终轮廓和 points CSV 仅含保留柱。\n"
                "无保留柱时仍输出 step1~step8 和空点表，不生成 step9/step10 曲线。\n")
        f.write("\n" + STEP_HELP)
    manifest_path = output.path("manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    aux["process_manifest"] = manifest_path
    aux["process_run_id"] = output.run_id
    return output.directory


def _finish_detection(image_path, img, columns, profile, aux, tag, process_dir, on_image=None, scale=None, **extra):
    """整图与 ROI 共用同一条结果导出/元数据/回调路径。"""
    meta = _detection_meta_from_aux(aux, columns, image_width=img.shape[1])
    scale = validate_scale(scale or read_image_scale(image_path))
    paint, stats, stats_path = export_detection_outputs(image_path, img, columns, profile, tag, meta, scale=scale)
    payload = {"path": image_path, "process_dir": process_dir, "img": img, "columns": columns,
               "process_manifest": aux.get("process_manifest"), "process_run_id": aux.get("process_run_id"),
               "structure_profile": profile, "tag": tag, "paint": paint, "stats": stats, "stats_path": stats_path, "scale": scale,
               "detection": meta, "quality": aux.get("quality", {}),
               "threshold_mode": aux.get("threshold_mode", "unknown"),
               "binary_fraction": aux.get("binary_fraction", 0), **extra}
    for key in ("mode", "variant", "confidence", "attempts", "warning"):
        payload["detection_" + key] = meta.get(key)
    for key in ("cd_hint_px", "cd_range_px", "cd_observed_px", "detected_count", "excluded_count"):
        payload[key] = meta.get(key)
    if on_image is not None:
        on_image(img, paint, os.path.basename(image_path), payload)
    return payload


def process_image(image_path, on_image=None, scale=None):
    output = _ProcessImages(image_path, "_tool")
    img, binary, aux = preprocess_image(image_path, on_stage=output.save)
    columns, profile = _detect_columns(img, binary, aux, roi_mode=False, on_stage=output.save)
    process_dir = export_process_images(output, img, aux, columns)
    return _finish_detection(image_path, img, columns, profile, aux, "_tool", process_dir, on_image, scale=scale)


def _clip_roi(roi, shape):
    """统一 ROI 裁切范围，保证阶段图与实际识别使用同一坐标。"""
    h, w = shape[:2]
    x0, y0, x1, y1 = [int(v) for v in roi]
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w - 1))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h - 1))

    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError(
            f"ROI 区域过小：{x1 - x0 + 1}×{y1 - y0 + 1}px"
        )
    return x0, y0, x1, y1


def process_image_roi(image_path, roi, on_image=None, scale=None):
    output = _ProcessImages(image_path, "_user") if DEBUG_OUTPUT else None

    def save_roi_stage(name, image):
        x0, y0, x1, y1 = _clip_roi(roi, image.shape)
        output.save(name, image[y0:y1 + 1, x0:x1 + 1])

    img, binary, aux = preprocess_image(image_path, on_stage=save_roi_stage if output else None)
    x0, y0, x1, y1 = _clip_roi(roi, img.shape)

    sub_img = img[y0:y1 + 1, x0:x1 + 1].copy()
    sub_bin = binary[y0:y1 + 1, x0:x1 + 1].copy()

    sub_aux = {key: value[y0:y1 + 1, x0:x1 + 1].copy() if isinstance(value, np.ndarray) else value
               for key, value in aux.items()}

    columns, local_profile = _detect_columns(
        sub_img,
        sub_bin,
        sub_aux,
        roi_mode=True,
        on_stage=output.save if output else None
    )
    process_dir = export_process_images(output, sub_img, sub_aux, columns, (x0, y0))
    columns = _offset_columns(columns, x0, y0)
    return _finish_detection(image_path, img, columns, None, sub_aux, "_user", process_dir, on_image, scale=scale,
                             local_structure_profile=local_profile, roi=(x0, y0, x1, y1))


def batch_process(on_image=None):
    """批量处理文件夹内所有 tif（供脚本/GUI 调用的便捷入口）。"""
    tif_files = sorted({
        p
        for ext in INPUT_GLOBS
        for p in glob.glob(os.path.join(INPUT_DIR, ext))
    })

    ok = 0
    fail = 0

    for path in tif_files:
        try:
            process_image(path, on_image=on_image)
            ok += 1
        except Exception:
            fail += 1

    return ok, fail


# =====================================================================
# GUI
# =====================================================================

core = sys.modules[__name__]

# 可见参数保留宽度先验；其余由图像证据自动确定。
PARAM_DESC = {}
MAIN_PARAMS = [
    ("AUTO_CD_ESTIMATE_PX", "Line CD (px，0=自动)", "float", 0, 100000, 1),
    ("AUTO_CD_TOLERANCE_PX", "允许误差 ±(px)", "float", 1, 10000, 1),
]
PARAM_DESC.update({
    "AUTO_CD_ESTIMATE_PX": "完整柱 Line CD 的大致边界距离（px）。填 0 时由候选柱的中位宽自动估计。",
    "AUTO_CD_TOLERANCE_PX": "用户先验的半宽范围。输出会同时列出 ±1 倍用户范围与 ±2 倍候选保护范围；边缘残柱/软候选还会自动放宽。",
})

FORMULA_HELP = """## 1  符号、范围与单位
本文描述 v6.0.0 实际实现。几何分区为程序定义的工程指标，各项可以同时出现。
灰度、binary、坐标三个量分开使用；尺寸来自坐标，不直接由灰度值换算。

第 i 行轮廓为 (y_i, L_i, R_i)，i=0…N−1；y 从顶部向底部增加，L/R 为左右 x 坐标（px）。
本图比例尺 s=标尺实际长度(nm)/标尺像素长度(px)，须为正数。
Y_i=s(y_i−y_0)，X_Li=sL_i，X_Ri=sR_i；CD_i=X_Ri−X_Li；H=Y_(N−1)。
注意：CD 与 H 均为边界距离，没有“+1 px”。顶/底梯度点可为亚像素，最终逐行轮廓
采用四舍五入后的顶/底行；H 来自这些行，不直接用 top_subpixel/bottom_subpixel 相减。
N<8、H≤0 或存在非有限轮廓坐标时，形貌数值为空。|量值|<10⁻⁶ nm 归零以消除数值残差。

## 2  基础尺寸、分段 CD、角度
单柱平均 CD=(1/N)Σ_i CD_i。三段按行索引分组：g_i=min(2,floor(3i/N))，
g=0/1/2 分别为上/中/下段；每段 CD=该段全部轮廓行 CD_i 的算术平均。
该处包含保留下来的 binary 回退坐标，不只取 valid=1 的灰度点。
三种比例为 CD_top/CD_mid、CD_top/CD_bot、CD_mid/CD_bot；分母绝对值≤10⁻¹⁵ 时为空。

左右侧壁分别在各段拟合 X=a+bY（最小二乘）：
b=Σ(Y_i−Ȳ)(X_i−X̄)/Σ(Y_i−Ȳ)²，a=X̄−bȲ。
θ=atan2(1,b)×180/π。每段≥4 行时使用此拟合；2~3 行用首尾
θ=atan2(Y_last−Y_first, X_last−X_first)×180/π；不足2行为空。
图像 y 向下：竖直为90°，随 y 增大向右偏移为<90°，向左为>90°；左右墙均用此方向定义。
手动三点角度 A−O−B：u=A−O，v=B−O，θ=acos(clip(u·v/(|u||v|),−1,1))×180/π。
任一线段长度为0时不输出角度；nm 长度为像素欧氏长度乘 s。

## 3  平均轮廓如何构造
对每根柱 j，z_i=Y_i/H_j，顶部 z=0，底部 z=1。
中段中心 c_j=mean((X_Li+X_Ri)/2，0.35≤z_i≤0.65)，左右 X 均减去 c_j。
在 z_k=k/400（k=0…400）处，对左右坐标分别作分段线性插值，得到 L_jk、R_jk。
两相邻采样间 X(z)=X_i+(X_(i+1)−X_i)(z−z_i)/(z_(i+1)−z_i)。
纳入 M 根完整且可信的柱后：L̄_k=Σ_j L_jk/M，R̄_k=Σ_j R_jk/M，
H̄=Σ_j H_j/M，Ȳ_k=z_kH̄，CD̄_k=R̄_k−L̄_k。
每根柱等权；只对齐高度比例与中心，不将宽度归一化。高度相同的百分比可对应不同的 nm。
左右坐标离散 σ_Lk=sqrt(Σ_j(L_jk−L̄_k)²/M)，右侧同理。单样本图的阴影是柱间±1σ，
不是测量误差或置信区间。只有一根柱时该离散为0。

纳入条件：未被图像/ROI截断，且 edge_support≥0.6。
edge_support=(左 valid=1 数+右 valid=1 数)/(2×轮廓行数)。
侧边≥10%轮廓行贴框，或最终顶行/底行到达图像边缘，判为截断。
保留列表中的截断柱/弱证据柱仍有逐柱结果及原因，但不进入 Baseline 平均。
整图强制排除的最左/最右柱已在此之前移除；它们不在逐柱统计里，排除情况见 step6/manifest。

## 4  几何形貌的逐项公式
以下 X、Y、W 均为 nm，z 为0~1。单柱用原生逐行轮廓，平均形貌用上节401点轮廓。
定义 W(z)=R(z)−L(z)。在区间[a,b]内最小二乘拟合 W=α+βz，得到全高度参考线 F_[a,b](z)。
β=Σ(z−z̄)(W−W̄)/Σ(z−z̄)²，α=W̄−βz̄；参考段不足2点时实现回退为 W 本身。
Q90(v)：排序 v_(0)…v_(n−1)，h=0.9(n−1)，k=floor(h)，t=h−k；
Q90=(1−t)v_(k)+t·v_(min(k+1,n−1))。先求百分位，再与0取较大值；不是先逐点删掉负值。

T-top 顶部展宽量：max(0,Q90({W(z)−F_[0.20,0.35](z) : z≤0.18}))。
Footing 底部展宽量：max(0,Q90({W(z)−F_[0.65,0.80](z) : z≥0.82}))。
两者是左右边界合计的宽度差，不是单侧突出距离。

中间细/中间粗：肩部 A=[0.20,0.30]、B=[0.70,0.80]，求 (z_A,W_A)、(z_B,W_B) 各自均值。
S(z)=W_A+(W_B−W_A)(z−z_A)/(z_B−z_A)。在0.35≤z≤0.65内：
中间细=max(0,Q90(S−W))；中间粗=max(0,Q90(W−S))。若两肩缺任一侧则两项为空。

Top-rounding 等效圆角半径：参考线特定采用 F_[0.15,0.35]。
取 z≤0.20 的 D_i=max(0,F_[0.15,0.35](z_i)−W_i)，以实际 Y 作梯形积分：
A=Σ_i (D_i+D_(i+1))(Y_(i+1)−Y_i)/2，只积分选中顶部采样点之间的区间。
r_eq=sqrt(max(0,A)/(2(1−π/4)))。两侧相等理想圆角的缺失面积各为 r²(1−π/4)，据此换算。
这是面积等效值，非圆弧拟合结果；严重 T-top、倾斜或遮挡时需配合轮廓判断。

## 5  LER / LWR 与周期起伏
只考察主体0.15≤z≤0.85。左LER仅取左valid=1；右LER仅取右valid=1；
LWR要求同一行左右均valid=1。每项至少8个有效点，且有效数≥该主体总点数的70%，否则为空。
对每项 v=X_L、X_R 或 W，以其有效 Y 做一次最小二乘去趋势 v=a+bY，残差 e_i=v_i−a−bY_i。
σ(e)=sqrt(Σ_i(e_i−ē)²/n)，即总体标准差（ddof=0）。
LER_L=3σ(e_L)，LER_R=3σ(e_R)，LER=(LER_L+LER_R)/2；任一侧缺失时平均LER为空。
LWR=3σ(e_W)。左右边缘的相关性会影响 LWR，它不等于两侧LER之和。
软件没有噪声去偏，滤波和测量长度也影响数值。

周期起伏为辅助量，不是已证实的光学驻波。主体左右所有点均须有效，且至少24个采样点。
先对左右 X(Y) 去二次趋势，残差 e_L/e_R；Hann窗 w_i=0.5−0.5cos(2πi/(n−1))。
P_k=|rFFT(w·e_L)_k|²+|rFFT(w·e_R)_k|²，采样间隔 d=median(Y_(i+1)−Y_i)，f_k=k/(nd)。
仅在 2/(Y_last−Y_first)≤f≤1/(4d) 内选最大P主峰 k*（至少2周期、每周期至少4点）。
C=(P_(k*−1)+P_k*+P_(k*+1))/Σ_(k≥1)P_k，边界按已有频率截断。
可选频段总能量<10⁻¹²或C<0.45，输出幅值0、周期空；采样不足/存在缺口则两项均空。
峰通过后，对左右原始坐标分别拟合 X=a+bt+ct²+u·sin(φ)+v·cos(φ)，
t=(Y−mean(Y))/(Y_last−Y_first)，φ=2πf_k*(Y−Y_first)。
单边幅值 A_L/R=sqrt(u²+v²)，报告 A=(A_L+A_R)/2，周期 T=1/f_k*（nm）。
幅值是正弦单边振幅，不是峰峰值；频率受当前采样长度的 FFT 分辨率限制。

## 6  逐柱统计与 Baseline 的两种平均
对某指标 m，只取已纳入柱中该项为有限值的 n 根：
逐柱均值 m̄=Σm_j/n；逐柱标准差 SD=sqrt(Σ(m_j−m̄)²/(n−1))，n<2时SD为空。
同时保存最小值 min(m_j)、最大值 max(m_j)、有效柱数 n、≥1nm柱数 Σ1(m_j≥1)。
1nm仅是统一出现量统计阈值，不是验收标准；高度、CD、周期也会机械地统计这个计数。
不同指标有效柱数可以不同。主界面均值包含所有显示柱的有限逐柱值，可能包含截断柱；
Baseline 则只用纳入柱，且其“样本值”不总是逐柱均值。

Baseline 的高度、平均CD、T-top、圆角、中间细/粗、Footing：对平均坐标轮廓重新计算。
LER_L/R/平均LER、LWR、周期幅值、周期：采用逐柱原生采样测量值的均值。
原因是平均坐标可能把反相粗糙起伏相互抵消。mean_profile_metrics另存平均坐标自身全部指标，
它与主比较值及逐柱均值可能不同；“样本明细”可同时核对主值和逐柱统计。

标准方波基准：L(z)=−CD0/2，R(z)=CD0/2，Y(z)=H0z；几何偏差及粗糙度为0，
周期幅值为0、周期为空。基准的单根记录是理想模板，不是一根实测柱。
每个指标 Δ=m_sample−m_baseline；变化率 %=100Δ/|m_baseline|。
缺少任一值则Δ为空；|m_baseline|≤10⁻⁸时%为空，仍保留有定义的绝对差值。
例如LER从2nm变3nm：Δ=+1nm，变化率+50%；方波LER=0，样本3nm：Δ=+3nm，%为空。
增减方向不代表工艺优劣；表格显示“—”或“-”表示无值，0是有效测量值。

## 7  图像质量辅助量
以下对 step2 滤波灰度计算，不参与 nm 换算：
对比度=std(gray)，使用全部像素的总体标准差。
边缘强度=mean(sqrt(Sobel_x(gray)²+Sobel_y(gray)²))，Sobel核3×3。
灰阶覆盖数=count(256档直方图中非空的灰阶)；binary占比=count(binary>0)/总像素数。
候选置信度是排序评分，不是识别准确率百分比。详细评分和算法见“step1–10 算法详解”。
灰度剖面、结构投影及CD曲线见“Profile 计算公式”。
"""

PROFILE_FORMULA_HELP = """## 1  灰度 Profile：行、列、直线
G(y,x) 为 step1 得到的8位灰度，纵轴0~255，无nm单位。
横向行：固定 y0，g_k=G(y0,k)，k=0…图宽−1；纵向列：固定 x0，g_k=G(k,x0)。
两种模式相邻样本间隔1px，距离d=|k2−k1| px。

直线端点 A=(x0,y0)、B=(x1,y1)，D=sqrt((x1−x0)²+(y1−y0)²)，
n=max(2,round(D)+1)，t_k=k/(n−1)，P_k=A+t_k(B−A)，k=0…n−1。
实际取值位置 Q_k=(clip(rint(P_k.x),0,图宽−1),clip(rint(P_k.y),0,图高−1))，
g_k=G(Q_k.y,Q_k.x)。rint为最近整数，恰好半整数取偶数；灰度不是双线性插值。
斜线可能重复采到同一像素。图上标记使用连续P_k，灰度来自最近整数Q_k。
横轴显示“采样索引 index”；斜线每个索引步长为D/(n−1)px，不固定等于1px。
两选点距离 d_px=|k2−k1|D/(n−1)，d_nm=s·d_px；Δgray=g_(较大索引)−g_(较小索引)。
整条线长D，采样点数n；采样点数不是线长。例如10×10px斜线，D≈14.142px，n=15。

## 2  binary 结构 Profile：下方横向投影
选中候选的binary图 B(y,x)∈{0,1}，对整张工作图（整图或ROI）的所有 H_img 行计算
p(x)=(1/H_img)Σ_y B(y,x)。值域0~1，表示该列被binary占据的比例，不是灰度或CD。
它包含之后可能被两端排除的候选柱。种子提取时使用的主体带投影另行平滑5点并以0.5分段。
综合结果图先作 p_norm=(p−min(p))/(max(p)−min(p))，再缩放到面板高度；
max=min时归零。纵坐标仅为显示比例，不能读取实际占据比例。

## 3  单柱宽度 Profile：右侧逐柱投影
在柱覆盖的每一行，w(y_i)=R_i−L_i（px），其他行记为0；实际CD=s·w(y_i)。
综合图为布局将(w−min(w))/(max(w)−min(w))缩放到面板宽度，max=min时归零。
因此它是柱宽随高度的曲线，不是逐列灰度。
不能把不同面板上曲线伸出的屏幕宽度直接当成跨柱CD比较值。

## 4  Baseline 平均轮廓与 CD Profile
每根柱以顶部y=0、中段中心x=0对齐，在z_k=k/400上线性插值左右轮廓。
L̄(z_k)=Σ_j L_j(z_k)/M，R̄(z_k)=Σ_j R_j(z_k)/M，CD̄(z_k)=R̄−L̄，Ȳ=z_k·mean(H_j)。
左图：横轴x(nm)、纵轴距顶部Ȳ(nm)，保持宽高实际比例。
中图：横轴CD̄(nm)、纵轴100z(%)，顶部0%、底部100%；只对高度百分比作归一化，CD不缩放。
筛选单样本时左图阴影为左右坐标各自的柱间±1σ：σ_L=sqrt(Σ_j(L_j−L̄)²/M)，右侧同理。
该阴影表达柱与柱的形状离散，不等于LER，也不是置信区间。

## 5  step9/step10 的核查曲线
step9 对每根保留柱的每一行绘 G(y,x) 与 filtered(y,x)，横轴为标注区间内的x坐标。
step10 对柱范围内每一列绘 G(y,x) 与 filtered(y,x)，横轴为y坐标。
灰色为原灰度、蓝色为滤波灰度，纵轴统一0~255；彩色竖线表示所识别边界位置，黄色为弱证据。
每页最多24条，所有行/列都有分页；NPZ保存全部原始/滤波矩阵及x、y、ROI偏移、逐行轮廓。
没有保留柱则不生成这两组曲线。points CSV 的局部坐标+ROI偏移=原图全局坐标。
"""

SHAPE_HELP = "形貌数值与平均坐标说明\n\n" + MORPHOLOGY_DEFINITION + "\n\n逐项公式请查看“统计公式”。"

PROFILE_HELP = """灰度曲线（剖面）与量测说明
================================================

一、三种曲线模式
------------------------------------------------
1) 直线剖面
   用“直线量测”工具在图上点 A、B 两点，沿这条线取逐像素灰度。
   用途：看穿过柱子的那一条线上，灰度从哪里升起、从哪里落下，
        判断边缘位置与阈值是否合适。

2) 横向行剖面（y = 某行）
   在“行 y / 列 x”里输入行号，显示该行逐像素的灰度曲线。
   用途：一眼看清这一行上柱子的排列、间距、宽度，以及背景是否干净；
        原图上会用一条水平虚线标出这一行的位置。

3) 纵向列剖面（x = 某列）
   输入列号，显示该列自上而下的灰度曲线。
   用途：看单根柱（或柱间空隙）沿高度方向的灰度变化，
        判断顶部/底部边界、footing 起止位置；
        原图上会用一条竖直虚线标出这一列。

二、曲线上的交互
------------------------------------------------
  · 单击曲线            → 选中该点，原图上用绿色十字 “P” 标出位置，
                          下方提示行给出 (x, y) 与灰度值；
   · Shift+单击（或右键）→ 设为量取端点，最多两个（M1、M2）；
   · 拖动橙色 M1/M2     → 实时拉动端点，原图橙色连线、距离和灰度差同步更新；
   · 两个端点都设好后    → 曲线上橙色区间 + 原图上橙色连线，
                           显示 Δpx 与按比例尺换算的 nm，以及 Δgray；
  · “清除量测”按钮      → 清除端点与选点。

曲线横轴是采样索引；距离由当前采样线换算：
  · 横向行剖面上量一段 = 横向距离（可直接当 CD / 间距用，单位 nm）；
  · 纵向列剖面上量一段 = 纵向距离（可直接当高度 / footing 高度用）；
  · 直线剖面上量一段   = 沿该直线的实际长度。

三、小技巧
------------------------------------------------
  · 想量多根柱的间距：切“横向行剖面”，取柱底附近那一行，
    在曲线上 Shift+点击相邻两个亮区的中点；
  · 想核对 footing：结合底部左右边界和横向灰度剖面；
    中心列的灰度跃迁可辅助检查底界，不能独立决定底部展宽的起点。
   · 量测结果会随比例尺自动换算 nm，改比例尺后立即生效。
   · 也可以直接在原图拖动蓝色 A/B 端点，直线剖面会同步刷新；
   · 行/列剖面的橙色量测端点独立于直线采样 A/B，不会意外改变原曲线。
 """

STEP_HELP = """## 如何按顺序读过程图
所有过程图都在 原图目录/图片名/test/，按图片名、tool/user和时间戳区分运行。
以下是保存图对应的计算与证据；精定位在step6保存之前已经完成，step7/8展示其不同结果。
算法采用保边滤波、binary定位和受约束梯度追踪，无需外部模型或联网。

## step1  原始灰度 gray
读原图；BGR/BGRA用G=0.114B+0.587G+0.299R转灰度（alpha不参与）。8位灰度直接保留。
其他位深取lo=Q1、hi=Q99，G=uint8(clip(255(I−lo)/(hi−lo),0,255))；
若hi≤lo则尝试原图min/max，仍相等输出0。后续“原灰度”均指该8位G，不是16位原数值。
检查：柱/沟真实明暗、截图遮挡、过曝，以及框内是否包含上下界面与柱旁背景。

## step2  保边滤波 filtered
M(p)=median(G(q), q∈3×3邻域)，先去除孤立噪点。
双边滤波 F(p)=Σ_q w(p,q)M(q)/Σ_q w(p,q)，
w=exp(−||p−q||²/(2·7²))·exp(−(M(p)−M(q))²/(2·20²))，OpenCV直径5邻域。
侧壁梯度精修用F，原灰度G仍保留作人工核查。细于滤波尺度的纹理可能被削弱。

## step3  局部对比度增强 enhanced
对F做CLAHE：8×8分块，对各块直方图限幅（clipLimit=1.8）后重分配超量频数，
由累计分布映射到0~255，并插值衔接块间映射。目的为低对比主体提供另一种粗定位候选。
step3展示原明暗方向的增强图；暗柱候选内部会先反相再CLAHE。

## step4  选择 binary 候选
亮柱sign=+1、暗柱sign=−1。亮柱用F，暗柱用255−F；默认亮柱，只有选自动才评估两个极性。
每种极性有3路：滤波图、CLAHE图、全局横向一阶照明校正图。
校正：x∈[−1,1]，μ_x=mean_y F(y,x)，b=Σ_x(μ_x−mean(μ))x/Σ_x x²；
F_corrected=minmax_to_0_255(F−bx)。只去全局斜坡，不用局部高通把实体拆成双边。
每路Otsu阈值 t*=argmax_t ω0ω1(μ0−μ1)²，B=1(I>t*)；随后3×3先膨胀再腐蚀的闭运算。
μ0/μ1为阈值两侧灰度均值，ω0/ω1为像素比例。step4保存评分最高的候选，不固定使用CLAHE。
检查：白色主体是否覆盖柱身；光照/基座相连时比较manifest中的候选分数。

## step5  binary 描边、分柱与拒绝 trench
主体y范围：每行去一阶横向趋势，C=高斯11点平滑(Q90−Q10)；
E=高斯11点平滑(Q85(|Sobel_x(Gaussian5(gray))|))。
A=高斯11点平滑(0.72·clip(C/Q95(C),0,2)+0.28·clip(E/Q95(E),0,2))。
分母设正数保护。对归一化A再做Otsu；主体起点门槛为
max(0.22,clip(0.75·Otsu比率,0.12,0.65))×max(A)，取足够长连续区间。
整图优先从上方0.55图高以内开始的长区间；ROI取消此优先；底界用稳健活动度收紧融合区。

主体上下各裁去约1/6作种子带，p_seed(x)=高斯5点平滑(mean_y B)。
p_seed≥0.5且连续≥5px形成种子；相邻柱间隙中点形成各柱独立cell。
每行只取cell内含种子中心的连续白段（≥3px）；不接受连接到内部cell边界的整片基座。
按相邻有效行坐标差>8分段，取最长段（至少20行），小缺口才线性插值。
普通柱中位宽≥15px、贴边柱≥3px，高度行数/宽度≥1.2。

极性检查：抽样行步长max(1,floor(N/32))；取柱中央约一半的灰度中位数和两侧距边界2~12px的背景。
d_y=sign·(median(柱内G)−median(两侧G))，噪声η=1.4826·median(|G−median5(G)|)。
只有median(d_y)>max(1.5,1.8η)，且至少65%抽样行d_y>1，才接受该柱。
这一步用于拒绝反相沟槽；自动明暗仍是灰度推断，不能保证复杂SEM中所有柱/沟语义正确。

候选总分：S=C_count+1.45F_width+1.20F_consistency+0.85F_height+0.55F_gap+0.45F_edge+C_gray−P_edge。
C_count=min(3,0.55Σ coverage_j²)，coverage为候选柱与主体高度交集/主体高度，避免短碎块靠数量胜出。
F_width=mean(exp(−|w_j−用户CD|/max(1,容差)))；CD=0自动模式取1。
F_consistency=exp(−max(MAD(w)/max(2,0.22median(w)),std(w)/max(2,0.18median(w))))。
有至少2个非贴边柱时宽度统计优先用它们。F_height=clip(median(柱行数)/图高,0,1)。
F_gap=clip(median(中心间距)/max(3,0.8median(w)),0,1)，单柱取0.15。
F_edge=clip(median(抽样左右边界绝对x梯度均值)/8,0,1)。
C_gray=0.35·clip((柱中央与相邻沟中央灰度中位数之差)/15,−1,1)；单柱或无数据取0。
P_edge=0.25×贴边柱数；全是贴边候选再加1.5。分数仅用于候选排序，不是概率。
用户CD先验使用中心±2容差作为候选保护带，贴边/临界宽度会软放宽；若过滤后不足原候选一半则保留原组。
三路binary均无合格柱才启用几何灰度提议：逐行去照明趋势，(gray−Q10)/max(Q90−Q10,2)
归一化后取列中位数、平滑11点，Otsu分段形成cell；从强活动行向上下按有符号梯度追踪。
该路径重建mask后仍经过极性与宽度检查，分数再减0.25，然后进入同一精修流程。
step5在原灰度上画选中候选的橙色粗轮廓，尚未强制排除两端柱。

## step6  两端柱排除结果
先精修所有候选（见step7），按中心x排序；整图强制去掉最左、最右检测柱，无论残柱还是完整柱。
只检测到1~2根时全部排除；ROI不执行这条规则。红色排除、绿色保留。
检查：没有结果是否因为只识别到两根；此时可改用ROI。

## step7  灰度精定位点
半径 r=min(32,max(3,floor(0.32·max(本柱中位宽,候选柱典型宽))))。
g_x=sign·Scharr_x(F)/32，g_y=sign·Scharr_y(F)/32；左侧响应g_x、右侧−g_x、顶部g_y、底部−g_y。
对每条路径先计算 G75=max(1,Q75(每行搜索带内非负响应的最大值))。
单点代价 U_i(k)=−clip(max(g_i(k),0)/G75,0,3)+0.15(offset_k/r)²。
跨行代价 V(k,l)=0.07min((x_ik−x_(i−1)l)²,36)+0.10|x_ik−x_(i−1)l|。
动态规划 D_i(k)=U_i(k)+min_l(D_(i−1)(l)+V(k,l))，末行选最小值并回溯。
每个搜索点必须在本柱cell及对应左右半侧，防止追到邻柱。
梯度峰相邻值a,b,c若b≥a且b≥c、a−2b+c<−10⁻⁶，
δ=clip(0.5(a−c)/(a−2b+c),−0.5,0.5)，边界x=x_peak+δ，再限制在有效搜索带内。
只有峰强度>max(0.8,0.10G75)时标valid=1，否则回退binary种子位置并标valid=0。

顶/底：对柱内中央约20%~80%的各列同样追踪y峰，y搜索范围为粗顶/底±min(r,8)。
支持的峰数量至少max(2,floor(列数/3))时用中位y，否则回退粗顶/底。
若底部峰中位强度<max(1.5,0.15顶部强度)，检查柱外距离3~9px（窄柱范围更小）的g_y正峰；
至少4个参考点、y标准差<max(2,0.6r)才以中位y推断底界，来源标base_reference。
这些点是柱外基座证据，不是可见柱内底界。顶部/底部最终取整为轮廓起止行。
色标：左侧青蓝、右侧橙、顶部绿、底部/基座紫；无效点黄。点表保留strength、valid与source。

## step8  最终逐行轮廓
展示step7得到的左右边界及回退点组成的闭合轮廓，并填色区分柱。
轮廓未仅保留valid点；无效行的binary回退也在其中。形貌的几何量使用该轮廓，粗糙度另筛valid。
侧边有≥10%种子行贴框时锁住贴框种子并标无效；左右交叉或间距≤1px时回退种子。
截图绘制会取整数像素，TXT/JSON计算仍保留左右浮点坐标。
检查：顶/底含回退或基座推断时，应结合step7点源与原图核对。

## step9  每根柱的逐行灰度核查
在保留柱每一行，沿x绘制原灰度G（灰色）、F（蓝色）、边缘位置竖线。
x范围为该柱左右极值各扩展r并裁剪到工作图。每页6×4=24条，所有行依次分页。
检查：左右定位点是否在正确的灰度跃迁附近；黄色代表弱证据。

## step10  每根柱的逐列灰度核查
从floor(最左边界)到ceil(最右边界)逐列沿y作图，y范围为顶/底各扩展r。
叠加该列所有匹配的定位点，包括存在的top/bottom点；没有顶底搜索点的列不会虚构点。
同样24条/页，完整G/F矩阵另存PR*_profiles.npz。
检查：顶部灰度跃迁、底部融合与column_gradient/base_reference/binary_fallback的来源。

## 输出与空结果
阶段完成即写图。无保留柱仍保存step1~8及空点表，不生成step9/10。
点表x/y为工作图局部坐标，x_global=x+ROI_x0、y_global=y+ROI_y0；整图偏移为0。
manifest包含本次文件清单、候选得分、排除的检测柱号与各柱的边缘支持度。
自动模式始终保存过程图，ROI是否保存由过程输出开关控制。灰度/图像质量不能替代人工轮廓核查。
"""

FILE_TAG_HELP = """输出文件说明
识别过程中，各阶段图完成后立即保存到输出目录下的 test 文件夹。
含 step1~step10 阶段图、点坐标 CSV、manifest JSON 与完整曲线 NPZ。
文件名带图片名、_tool/_user 与时间戳；全部过程图在同一文件夹，重跑不覆盖。
黄色为无足够证据的点；局部/全图坐标分别记录。

================================================

文件名中的标记：
  · _tool：自动整图识别结果
  · _user：人工框选 ROI 识别结果

两类结果互不覆盖，放在同一张图对应的输出目录里。

自动结果（_tool）
  xx_tool_result.png            综合结果图（主图 + 下方列投影 + 右侧逐柱投影）
  xx_tool_result_nm.png         同上，但坐标轴按 nm 标注
  xx_tool_boundaries.txt        逐行边界（px；末尾附识别模式/CD范围注释）
  xx_tool_boundaries_nm.txt     逐行边界（nm；末尾附识别模式/CD范围注释）
  xx_tool_stats_nm.txt          尺寸、形貌、比例尺与识别参数统计表（nm）
  xx_tool_morphology.json       平均坐标、各柱形貌、柱间离散与统计定义

人工 ROI 结果（_user）
  xx_user_result.png / xx_user_result_nm.png
  xx_user_boundaries.txt / xx_user_boundaries_nm.txt
  xx_user_stats_nm.txt          尺寸、形貌、比例尺与识别参数统计表（nm）
  xx_user_morphology.json       平均坐标、各柱形貌、柱间离散与统计定义

过程图（自动识别固定保存；开关只控制 ROI）
  test/                         全部 step 阶段图、行列曲线与点坐标
  step1~step5                    灰度、滤波、增强、binary、粗轮廓
  step6_outermost_exclusion_…png 红色两端柱排除，绿色中间柱保留
  step7 / step8                  精定位点 / 最终轮廓
  step9 / step10                 每根保留柱的逐行 / 逐列灰度曲线分页图
  points_…csv / manifest_…json   保留点、排除规则、数量、证据来源及本次文件清单

输出位置：
  · 过程图固定保存到 原图目录/图片名/test/，不受常规输出开关影响。
  · 常规结果：勾选同名文件夹时放入该文件夹，否则放在原图目录。
  · Baseline 图表固定放入软件同级 baseline_comparison/时间戳/。
  · 本图标定保存在原图目录的 pr_image_scales.json，可随图片一同迁移。

Baseline 文件还包含：sample_index.csv（B0/S1…编号、完整路径、模式和比例尺）、
comparison_values.csv / comparison_deltas.csv / comparison_percent.csv（横向指标矩阵）、
per_pillar_morphology.csv（所有逐柱数据及排除原因）、per_sample/Sx_*（每款 PR 单独的指标、
逐柱 CSV 与增减图）。同名图片只要路径或 ROI 不同，编号也不同，不会合并。

每次识别的文本输出都会附带：识别模式、候选版本（含亮/暗极性）、
置信度（候选排序分数，非百分比）、候选尝试次数、估计 Line CD/允许误差、
用户先验范围、候选保护范围、完整主体柱/全部柱/边缘残柱观测范围。
Line CD 与统计 CD 均按左右边界距离输出。
"""

ABOUT_HELP = """PR Analyzer v6.0.0
原作者：E924744 Kang An

binary 粗轮廓 + 有符号 Scharr + 动态规划 + 亚像素定位。
支持整图和 ROI、亮/暗柱模式、逐行/逐列灰度曲线、独立比例尺及 Baseline 对比。
过程图含最终选中候选、识别点坐标、证据标记与全部灰度曲线。
采用可解释的传统视觉算法，无需模型下载。自动明暗不等于语义识别。
算法、指标公式和输出说明见本程序帮助菜单。
"""


QUICKSTART_HELP = """## 1  选图与标定
点击“选择 tif”或“浏览文件夹”。输入这张图的标尺像素长度与实际 nm 长度，例如 100 px = 50 nm。
标定按图片保存，不会应用到其他图片。首次使用未标定的图片，默认 1 px = 0.5 nm，请核对。

## 2  分析与核对
点击“分析选中图像”。自动分析会排除最左、最右两根柱，完整柱也会排除。
要分析单根柱或少量柱，在原图框出完整柱及周围背景，再点击“分析框选区域”；ROI 不排除两端。

## 3  独立调整每图比例尺
在左下“已分析文件”中选中图片，点击“设置所选图比例尺”，也可双击文件。
修改会同步更新该图的自动/ROI 统计、手动量测和 Baseline 样本；已有像素轮廓无需重识别。

## 4  量测与比较
选择“直线量测”，在图上设 A、B 点；拖动端点时右侧实时显示灰度曲线。
菜单“Baseline 对比”可选择标准方波或一款 PR，与多款图片同时比较。

## 5  找到输出
过程图：原图目录 / 图片名 / test。
对比图和 CSV 等报告：软件目录 / baseline_comparison / 本次时间戳。
各阶段完成后即保存；多次分析的过程图用不同时间戳区分。
"""

SCALE_HELP = """## 每张图单独保存标定
入口：左下“已分析文件”列表中的每张图，双击或点击“设置所选图比例尺”。
输入“标尺像素长度”和“实际长度 nm”。换算为：1 px = 实际长度 ÷ 像素长度。
批量分析逐张读取标定；同名但在不同目录的图片也相互独立。

## 哪些数据会跟着更新
仅本图的 nm 尺寸、LER/LWR、形貌、统计文件、直线/角度量测长度及 Baseline 样本同步重算。
比例尺不改变像素边缘坐标，也不会重新生成 test 过程图。
已有 Baseline 报告作为历史记录保留，修改后需要重新点击“计算并导出对比”。

## 重开软件与迁移
标定记录在图片所在目录的 pr_image_scales.json。重开软件会自动读取。
把图片复制到另一台电脑时，可同时复制此文件。图片重命名后需要重新设置标定。
新图默认 100 px = 50 nm；应以图中真实标尺为准。
"""

BASELINE_HELP = """## 1  选择基准与样本
标准方波：输入CD与高度（nm）；PR样本：选择完整柱平均轮廓，或指定一根完整柱。
“添加已分析结果”加入自动/ROI结果；“添加多张图片”识别所选图片并输出各自test过程图。
双击样本可改名称或本图nm/px。截断/弱证据柱不参与平均，完整柱数为0的样本不能比较。
选作PR基准的样本从待比较项中移除。每次报告固定以B0表示基准、S1…表示各待比较样本。
同名文件、不同目录、不同ROI仍有不同编号。编号限本次报告；原始路径和区域可在下方来源栏核对。

## 2  横向比较：先看不同 PR 的区别
“全部指标与逐柱统计”默认每个样本一列、每项指标一行，共13项数值。
“横向数值”可切换实测值(nm)、增减Δ(nm)、变化率(%)；B0列始终展示原基准值(nm)。
点击样本列可在下方查看其完整名称、路径和区域；列多时用表底横向滚动条。
结果筛选可以只显示一个Sx，图与表同步筛选，导出仍保存全部样本。

## 3  样本明细与逐柱数值
样本明细：B0/Sx各成一组，可展开/折叠，显示主比较值、Δ、%、逐柱均值/标准差、有效数及≥1nm计数。
逐柱数值：每个样本下列PR1、PR2…的所有13项值，并说明是否纳入平均。
灰字表示该柱未纳入，原因可能是框选截断或边缘证据不足。整图两端柱此前已移除，详见step6。
不同Sx中的PR1表示各图自己的第一根保留柱，不表示跨图同一根实物柱。
方波的B0逐柱记录是理想模板。空值显示“-”；有效0保留，样本标准差需至少2根有效柱。

## 4  如何读三张对比图
左图按真实nm比例叠加平均轮廓；B0为深色虚线，Sx在总览和筛选图中保持相同颜色。
单独筛选一个Sx时，左右轮廓阴影表示该样本柱间±1σ，便于看重复性，不是置信区间或LER。
中图显示CD随相对高度变化，顶部0%、底部100%，便于比较头部、腰部和底脚。
右图显示T-top、Top-rounding、中间细/粗、Footing、LER、LWR的Δ。
在“轮廓与形貌增减图”页内用“形貌增减显示”切换“点柱状图”或“表格”，默认点柱状图。
切换只改变右图的展示方式，左侧轮廓和中间CD曲线继续显示；单个/多个样本都可自由切换。
点柱状图：不同样本分色，向右增加、向左减少；样本不超过5款时标出Sx和Δ，更多时可筛选查看。
表格：每列一个Sx；红色增加、蓝色减少、白色接近0，色深对应同一nm差值尺度，格内显示具体Δ。
导出仍保留总览和独立横向点柱图；切换显示不重新识别、不改变统计数值或已导出的文件。
当不同量级导致小差异颜色不明显时，直接读数字或筛选单个样本。增减不表示合格与否。

## 5  计算口径与更新
几何量从平均坐标轮廓计算；LER/LWR及周期量先逐柱测量再平均，避免不同相位互相抵消。
Δ=样本−基准；%=100Δ/|基准|。基准接近0时%为空，绝对差仍保留。
所有指标公式、有效点条件和例子见“统计公式”；各类曲线见“Profile 计算公式”。
修改基准、样本或比例尺后原图表清空，需要重新计算；筛选/切换表格不改变数据或历史报告。

## 6  导出文件如何查找
点击“计算并导出对比”保存至EXE/PY旁 baseline_comparison/时间戳/，历史报告不覆盖。
sample_index.csv：编号、名称、原图完整路径、区域、比例尺与纳入/排除数。
comparison_values/deltas/percent.csv：三张横向表，每个样本一列。
baseline_comparison.csv/json：全指标长表与完整报告；average_coordinates.csv：平均坐标。
per_pillar_morphology.csv：全部逐柱数值；per_sample/Sx_*：各样本独立的指标表、逐柱表和对比图。
baseline_overlay.png：所有样本总览；README.txt内含本版公式。CSV为UTF-8 BOM，可用Excel打开。
"""

PIPELINE_HELP = """## 先定范围，再精修边缘
1. 原灰度归一化，经中值/双边滤波保边降噪，再做 CLAHE 增强。
2. 比较滤波灰度、增强图、亮度校正图的 binary 候选，以覆盖完整主体和灰度证据评分。
3. 对照同一行柱内与柱外灰度，检查亮/暗极性，降低把 trench 当柱的风险。
4. 在 binary 附近按行扫描有符号梯度，用动态规划追踪连续侧壁。搜索限制在各柱单元内。
5. 按列精修顶部与底部。柱底融入基座时，检查柱外基座跃迁，并记录推断来源。
6. 灰度梯度峰做亚像素定位，浮点坐标直接用于尺寸与形貌计算。

## 框选区域为什么可能失败
框大不代表有效证据更多。尽量包含完整柱、上方背景、底部界面及两侧空隙，减少文字/标尺/大面积基底。
默认“亮柱”不会自动反相；实际为暗柱时需切换。自动模式仍需人工区分柱与 trench。
全图只有 1~2 根柱时，按两端排除规则会无结果；可改为 ROI。
被 ROI 框切断的柱会有逐柱记录，但不参与完整形貌平均。

## 从过程图检查结果
step4/5：binary 是否覆盖柱体；step6：红色两端柱被排除。
step7：检查精定位点；黄色点是弱证据或被边框截断，不能当作独立精确观测。
step9/10：查看逐行/逐列灰度曲线与识别点；points CSV 保留完整坐标与有效性。
候选分数用于排序，不是识别准确率百分比。本方法无需联网或模型文件。
"""


class HelpWindow(tk.Toplevel):
    """内置、可搜索的分主题帮助；不依赖外部说明文件。"""

    def __init__(self, app, title, content):
        super().__init__(app)
        self.title(APP_TITLE + " · 帮助")
        self.geometry(f"{min(1050, self.winfo_screenwidth()-80)}x{min(760, self.winfo_screenheight()-110)}")
        self.topics = {"快速上手": QUICKSTART_HELP, "每图比例尺": SCALE_HELP,
                       "Baseline 对比": BASELINE_HELP, "识别流程": PIPELINE_HELP,
                       "统计公式": FORMULA_HELP, "Profile 计算公式": PROFILE_FORMULA_HELP,
                       "step1–10 算法详解": STEP_HELP, "形貌指标公式": MORPHOLOGY_DEFINITION,
                       "灰度曲线与量测说明": PROFILE_HELP, "输出文件说明": FILE_TAG_HELP, "关于": ABOUT_HELP}
        self.query = tk.StringVar()
        self.font_size = 11
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")
        ttk.Label(top, text="使用帮助", font=("Microsoft YaHei UI", 15, "bold")).pack(side="left", padx=(0, 20))
        ttk.Label(top, text="搜索").pack(side="left")
        self.search_entry = ttk.Entry(top, textvariable=self.query, width=25)
        self.search_entry.pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(top, text="A−", width=4, command=lambda: self._zoom(-1)).pack(side="left", padx=3)
        ttk.Button(top, text="A+", width=4, command=lambda: self._zoom(1)).pack(side="left")
        body = ttk.Frame(self, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        self.nav = ttk.Treeview(body, show="tree", selectmode="browse", height=10)
        self.nav.column("#0", width=180, stretch=False)
        self.nav.pack(side="left", fill="y", padx=(0, 12))
        frame = ttk.Frame(body)
        frame.pack(side="left", fill="both", expand=True)
        self.box = tk.Text(frame, wrap="word", relief="flat", bg="white", fg="#20334a",
                           padx=22, pady=18, spacing1=3, spacing3=8, font=("Microsoft YaHei UI", self.font_size))
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.box.yview)
        self.box.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.box.pack(fill="both", expand=True)
        self.box.tag_configure("title", font=("Microsoft YaHei UI", 17, "bold"), foreground="#175a7a", spacing3=20)
        self.box.tag_configure("heading", font=("Microsoft YaHei UI", 12, "bold"), foreground="#2463aa", spacing1=12, spacing3=8)
        self.box.tag_configure("match", background="#ffec9b")
        self.nav.bind("<<TreeviewSelect>>", self._render)
        self.query.trace_add("write", self._filter)
        self.bind("<Control-f>", lambda _e: self.search_entry.focus_set())
        self.select_topic(title, content)

    def select_topic(self, title, content):
        self.topics[title] = content
        self.query.set("")
        self._filter()
        self.nav.selection_set(title)
        self.nav.see(title)
        self._render()
        self.lift()

    def _filter(self, *_args):
        selected = self.nav.selection()
        self.nav.delete(*self.nav.get_children())
        term = self.query.get().strip().casefold()
        for title, content in self.topics.items():
            if not term or term in (title + content).casefold():
                self.nav.insert("", "end", iid=title, text=title)
        choices = self.nav.get_children()
        if choices:
            self.nav.selection_set(selected[0] if selected and selected[0] in choices else choices[0])
        self._render()

    def _render(self, _event=None):
        import re
        self.box.configure(state="normal")
        self.box.delete("1.0", "end")
        selected = self.nav.selection()
        if not selected:
            self.box.insert("end", "未找到相关主题，请换一个关键词。")
        else:
            title = selected[0]
            self.box.insert("end", title + "\n", "title")
            for line in self.topics[title].splitlines():
                if line and set(line.strip()) <= {"=", "-"}:
                    continue
                heading = line.startswith("## ") or bool(re.match(r"^(\d+[.、]|[一二三四五六七八九]、|■)", line))
                self.box.insert("end", line.removeprefix("## ") + "\n", "heading" if heading else "body")
            term, start = self.query.get().strip(), "1.0"
            if term:
                while True:
                    found = self.box.search(term, start, nocase=True, stopindex="end")
                    if not found:
                        break
                    start = f"{found}+{len(term)}c"
                    self.box.tag_add("match", found, start)
        self.box.configure(state="disabled")

    def _zoom(self, change):
        self.font_size = min(16, max(9, self.font_size + change))
        self.box.configure(font=("Microsoft YaHei UI", self.font_size))
        self.box.tag_configure("heading", font=("Microsoft YaHei UI", self.font_size + 1, "bold"))


def comparison_sample(payload, name=None, nm_per_px=None):
    """保存可独立对比的坐标快照，不保留整张图像；比例尺与该次识别绑定。"""
    import copy
    columns = copy.deepcopy(payload["columns"])
    scale = nm_per_px if nm_per_px is not None else payload["stats"]["morphology"]["nm_per_px"]
    source = os.path.abspath(payload["path"])
    mode = "ROI " + str(tuple(payload["roi"])) if payload.get("roi") else "自动"
    return {"name": name or f"{os.path.basename(source)} · {mode}", "source": source, "mode": mode,
            "columns": columns, "nm_per_px": float(scale)}


def sample_morphology(sample, selection="平均轮廓"):
    columns = sample["columns"]
    if selection != "平均轮廓":
        columns = [col for col in columns if f"PR{col['label']}" == selection]
    result = build_morphology(columns, sample["nm_per_px"])
    result.update(name=sample["name"] + (" · " + selection if selection != "平均轮廓" else ""),
                  source=sample["source"], mode=sample["mode"], kind="image", selection=selection)
    return result


class BaselineWindow(tk.Toplevel):
    """多图片/ROI 的独立对比视图；工作线程只分析图像，Tk 更新由主线程轮询。"""

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Baseline 对比 · 平均轮廓与形貌量值")
        self.geometry(f"{min(1400, self.winfo_screenwidth()-80)}x{min(920, self.winfo_screenheight()-110)}")
        self.minsize(min(1000, self.winfo_screenwidth()-80), min(680, self.winfo_screenheight()-110))
        self.samples = []
        self.report = None
        self.report_dir = None
        self.busy = False
        self.messages = queue.Queue()
        self.protocol("WM_DELETE_WINDOW", self._close)
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        self.base_mode = tk.StringVar(value="标准方波")
        for value in ("标准方波", "PR 样本"):
            ttk.Radiobutton(top, text=value, value=value, variable=self.base_mode,
                            command=self._base_changed).pack(side="left", padx=(0, 10))
        ttk.Label(top, text="CD (nm)").pack(side="left")
        self.cd = tk.StringVar(value="60")
        ttk.Entry(top, textvariable=self.cd, width=7).pack(side="left", padx=4)
        ttk.Label(top, text="高度 (nm)").pack(side="left")
        self.height = tk.StringVar(value="100")
        ttk.Entry(top, textvariable=self.height, width=7).pack(side="left", padx=4)
        self.base_name = tk.StringVar()
        pr_row = ttk.Frame(self, padding=(10, 0, 10, 8))
        pr_row.pack(fill="x")
        ttk.Label(pr_row, text="PR 基准图").pack(side="left", padx=(0, 8))
        self.base_combo = ttk.Combobox(pr_row, textvariable=self.base_name, state="readonly", width=42)
        self.base_combo.pack(side="left", padx=8)
        self.base_combo.bind("<<ComboboxSelected>>", self._base_changed)
        self.base_column = tk.StringVar(value="平均轮廓")
        self.column_combo = ttk.Combobox(pr_row, textvariable=self.base_column, state="readonly", width=12)
        self.column_combo.pack(side="left")
        ttk.Label(pr_row, text="完整柱平均 / 指定一根柱", foreground="#52657b").pack(side="left", padx=10)
        toolbar = ttk.Frame(self, padding=(10, 0, 10, 6))
        toolbar.pack(fill="x")
        self.buttons = []
        for index, (text, command) in enumerate((("添加已分析结果", self._add_cached), ("添加多张图片…", self._add_images),
                              ("修改名称 / 比例尺", self._edit_sample), ("移除所选", self._remove_samples),
                              ("计算并导出对比", self._compare), ("打开对比文件夹", self._open_folder))):
            button = ttk.Button(toolbar, text=text, command=command, style="Accent.TButton" if index == 4 else "TButton")
            button.grid(row=0, column=index, sticky="ew", padx=(0, 6), pady=3)
            toolbar.columnconfigure(index, weight=1)
            self.buttons.append(button)
        ttk.Label(self, text="添加的所有样本一同对比；选择 PR baseline 后自动从被比较样本中排除该项。可使用已分析的 ROI，或指定其中一根 PR。",
                  foreground="#52657b", padding=(12, 2), wraplength=950).pack(fill="x")
        self.sample_tree = ttk.Treeview(self, columns=("scale", "count", "mode", "source"), height=3, selectmode="extended")
        self.sample_tree.heading("#0", text="样本 / PR 类型（可改名）")
        self.sample_tree.column("#0", width=220)
        for key, title, width in (("scale", "nm/px", 80), ("count", "完整/总柱数", 100),
                                  ("mode", "区域", 150), ("source", "原始文件", 420)):
            self.sample_tree.heading(key, text=title)
            self.sample_tree.column(key, width=width)
        self.sample_tree.pack(fill="x", padx=10)
        sample_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.sample_tree.xview)
        self.sample_tree.configure(xscrollcommand=sample_scroll.set)
        sample_scroll.pack(fill="x", padx=10)
        self.sample_tree.bind("<Double-1>", lambda _e: self._edit_sample())
        result_bar = ttk.Frame(self, padding=(10, 8, 10, 0))
        result_bar.pack(fill="x")
        ttk.Label(result_bar, text="结果筛选").pack(side="left", padx=(0, 8))
        self.result_sample = tk.StringVar(value="全部样本")
        self.result_sample_combo = ttk.Combobox(result_bar, textvariable=self.result_sample,
                                               state="disabled", width=35, values=("全部样本",))
        self.result_sample_combo.pack(side="left", fill="x", expand=True)
        self.result_sample_combo.bind("<<ComboboxSelected>>", self._refresh_result_views)
        ttk.Label(result_bar, text="图表同步筛选 · 导出全部", foreground="#52657b").pack(side="left", padx=10)
        ttk.Button(result_bar, text="指标公式与口径", command=lambda: self.app._show_doc("统计公式", FORMULA_HELP)).pack(side="right")
        panes = ttk.Notebook(self)
        self.result_notebook = panes
        panes.pack(fill="both", expand=True, padx=10, pady=8)
        plot_frame = ttk.Frame(panes)
        panes.add(plot_frame, text="轮廓与形貌增减图")
        plot_bar = ttk.Frame(plot_frame, padding=(8, 6))
        plot_bar.pack(fill="x")
        ttk.Label(plot_bar, text="形貌增减显示").pack(side="left", padx=(0, 10))
        self.delta_view = tk.StringVar(value="点柱状图")
        for view in ("点柱状图", "表格"):
            ttk.Radiobutton(plot_bar, text=view, value=view, variable=self.delta_view,
                            command=self._refresh_result_plot).pack(side="left", padx=(0, 12))
        ttk.Label(plot_bar, text="Δ = 样本 − B0 · 多样本可用上方筛选查看",
                  foreground="#52657b").pack(side="left", padx=10)
        self.figure = self.canvas = None
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(10, 3.5), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        table = ttk.Frame(panes)
        panes.add(table, text="全部指标与逐柱统计")
        table_bar = ttk.Frame(table, padding=(0, 6))
        table_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.table_view = tk.StringVar(value="横向比较")
        self.table_value = tk.StringVar(value="实测值 (nm)")
        for title, var, choices, width in (("查看方式", self.table_view, ("横向比较", "样本明细", "逐柱数值"), 12),
                                           ("横向数值", self.table_value, ("实测值 (nm)", "增减量 Δ (nm)", "变化率 (%)"), 17)):
            ttk.Label(table_bar, text=title).pack(side="left", padx=(0, 6))
            combo = ttk.Combobox(table_bar, textvariable=var, state="readonly", values=choices, width=width)
            combo.pack(side="left", padx=(0, 12))
            combo.bind("<<ComboboxSelected>>", self._refresh_result_table)
            if var is self.table_value:
                self.table_value_combo = combo
        self.expand_buttons = []
        for caption, opened in (("展开", True), ("折叠", False)):
            button = ttk.Button(table_bar, text=caption, command=lambda value=opened: self._expand_results(value))
            button.pack(side="left", padx=3)
            self.expand_buttons.append(button)
        self.result_tree = ttk.Treeview(table, show="tree headings", selectmode="browse")
        sy = ttk.Scrollbar(table, orient="vertical", command=self.result_tree.yview)
        sx = ttk.Scrollbar(table, orient="horizontal", command=self.result_tree.xview)
        self.result_tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.result_tree.grid(row=1, column=0, sticky="nsew")
        sy.grid(row=1, column=1, sticky="ns")
        sx.grid(row=2, column=0, sticky="ew")
        table.rowconfigure(1, weight=1)
        table.columnconfigure(0, weight=1)
        self.result_tree.tag_configure("increase", foreground="#b34d21")
        self.result_tree.tag_configure("decrease", foreground="#17639d")
        self.result_tree.tag_configure("group", background="#e8eff7", foreground="#174675", font=("Microsoft YaHei UI", 10, "bold"))
        self.result_tree.tag_configure("stripe", background="#f3f6fa")
        self.result_tree.tag_configure("excluded", foreground="#7b8795")
        self.result_tree.bind("<ButtonRelease-1>", self._result_clicked)
        footer = ttk.Frame(self)
        footer.pack(side="bottom", fill="x", before=panes)
        context_frame = ttk.Frame(footer, padding=(10, 0))
        context_frame.pack(fill="x")
        self.result_context = tk.Text(context_frame, height=2, wrap="word", relief="flat", bg="#f3f6fa",
                                      fg="#52657b", font=("Microsoft YaHei UI", 9), padx=8, pady=4, state="disabled")
        context_scroll = ttk.Scrollbar(context_frame, orient="vertical", command=self.result_context.yview)
        self.result_context.configure(yscrollcommand=context_scroll.set)
        context_scroll.pack(side="right", fill="y")
        self.result_context.pack(fill="x")
        self.status = tk.StringVar(value="请选择 baseline 并添加样本。")
        ttk.Label(footer, textvariable=self.status, padding=(12, 5), wraplength=950).pack(fill="x")
        for var in (self.cd, self.height, self.base_mode, self.base_name, self.base_column):
            var.trace_add("write", self._invalidate)
        self._add_cached()
        self._poll_job = self.after(100, self._poll)

    def _invalidate(self, *_args):
        self.report = None
        self.report_dir = None
        self.result_tree.delete(*self.result_tree.get_children())
        self.result_sample.set("全部样本")
        self.result_sample_combo.configure(values=("全部样本",), state="disabled")
        self._show_result_context([])
        if self.figure is not None:
            self.figure.clear()
            self.canvas.draw_idle()
        self.status.set("参数或样本已变化，请点击“计算并导出对比”。")

    def _selected_report_samples(self):
        if self.report is None:
            return []
        index = self.result_sample_combo.current() - 1
        return [self.report["samples"][index]] if 0 <= index < len(self.report["samples"]) else self.report["samples"]

    def _show_result_context(self, samples):
        lines = []
        for sample in samples:
            lines.append(f"{sample['sample_id']} · {sample['name']}  |  {sample.get('source', '标准方波')}  |  "
                         f"{sample.get('mode', '理想轮廓')} · {sample.get('selection', '平均轮廓')}  |  "
                         f"纳入 {sample['included_count']} / 排除 {sample['excluded_count']} 柱")
        self.result_context.configure(state="normal")
        self.result_context.delete("1.0", "end")
        self.result_context.insert("end", "\n".join(lines) or "计算后显示样本编号与完整来源；点击表格可查看对应样本。")
        self.result_context.configure(state="disabled")

    def _refresh_result_views(self, _event=None):
        if self.report is None:
            return
        self._refresh_result_table()
        self._refresh_result_plot()

    def _refresh_result_plot(self):
        if self.report is None or self.figure is None:
            return
        samples = self._selected_report_samples()
        sample_id = samples[0]["sample_id"] if self.result_sample_combo.current() > 0 else None
        view = "table" if self.delta_view.get() == "表格" else "bars"
        draw_baseline_figure(self.report, self.figure, sample_id=sample_id, delta_view=view)
        self.canvas.draw_idle()

    def _refresh_result_table(self, _event=None):
        tree = self.result_tree
        tree.delete(*tree.get_children())
        view = self.table_view.get()
        self.table_value_combo.configure(state="readonly" if view == "横向比较" else "disabled")
        for button in self.expand_buttons:
            button.configure(state="disabled" if view == "横向比较" else "normal")
        if self.report is None:
            return
        samples = self._selected_report_samples()
        baseline = self.report["baseline"]
        self._show_result_context([baseline] + samples)
        tree.heading("#0", text="形貌 / 指标" if view == "横向比较" else "样本 / 指标（可展开）")
        tree.column("#0", width=275, minwidth=200, stretch=False)
        if view == "横向比较":
            columns = [("B0", "B0 · Baseline (nm)", 170)] + [(s["sample_id"], _sample_caption(s, 18), 195) for s in samples]
        elif view == "样本明细":
            columns = [("ref", "B0 (nm)", 85), ("value", "样本值 (nm)", 95), ("delta", "增减 Δ (nm)", 100),
                       ("pct", "变化 (%)", 85), ("mean", "逐柱均值 (nm)", 110), ("sd", "逐柱标准差", 100),
                       ("n", "有效柱数", 75), ("hits", "≥1nm柱数", 85), ("basis", "计算依据", 150)]
        else:
            columns = [("included", "是否纳入 / 原因", 185)] + [(k, title + " (nm)", max(125, len(title)*12)) for k, title, _ in MORPHOMETRY]
        tree.configure(columns=[key for key, _, _ in columns], displaycolumns="#all")
        from tkinter import font as tkfont
        heading_font = tkfont.Font(font=ttk.Style(self).lookup("Treeview.Heading", "font") or ("Microsoft YaHei UI", 10, "bold"))
        for key, title, width in columns:
            tree.heading(key, text=title)
            tree.column(key, width=max(width, heading_font.measure(title)+24), minwidth=60,
                        stretch=False, anchor="w" if key in ("basis", "included") else "e")
        lookup = {(r["sample_id"], r["metric"]): r for r in self.report["rows"]}
        if view == "横向比较":
            field = {"实测值 (nm)": "value", "增减量 Δ (nm)": "delta", "变化率 (%)": "percent"}[self.table_value.get()]
            for i, (key, title, unit) in enumerate(MORPHOMETRY):
                tree.insert("", "end", iid=key, text=title, values=[_num(baseline["metrics"].get(key))] +
                            [_num(lookup[(s["sample_id"], key)][field], field != "value") for s in samples],
                            tags=("stripe",) if i % 2 else ())
        else:
            for sample in [baseline] + samples:
                sid = sample["sample_id"]
                tree.insert("", "end", iid=sid, text=_sample_caption(sample, 60), open=True, tags=("group",))
                if view == "逐柱数值":
                    for i, col in enumerate(sample["per_column"]):
                        tree.insert(sid, "end", iid=f"{sid}:{i}", text=col["label"],
                                    values=["纳入" if col["included"] else col["reason"]] +
                                           [_num(col["metrics"].get(k)) for k, _, _ in MORPHOMETRY],
                                    tags=(("stripe",) if i % 2 else ()) + (() if col["included"] else ("excluded",)))
                else:
                    for i, (key, title, _) in enumerate(MORPHOMETRY):
                        stats = sample["summary"][key]
                        row = lookup.get((sid, key))
                        delta, pct = (row["delta"], row["percent"]) if row else (None, None)
                        values = (_num(baseline["metrics"].get(key)), _num(sample["metrics"].get(key)),
                                  _num(delta, True), _num(pct, True), _num(stats["mean"]), _num(stats["std"]),
                                  stats["n"], stats["above_1nm"], "逐柱原生采样均值" if key in ROUGHNESS_KEYS else "平均坐标轮廓")
                        tag = "increase" if delta is not None and delta > 1e-8 else "decrease" if delta is not None and delta < -1e-8 else ""
                        tree.insert(sid, "end", iid=f"{sid}:{key}", text=title, values=values,
                                    tags=(("stripe",) if i % 2 else ()) + ((tag,) if tag else ()))

    def _expand_results(self, opened):
        for iid in self.result_tree.get_children():
            if self.result_tree.get_children(iid):
                self.result_tree.item(iid, open=opened)

    def _result_clicked(self, event):
        if self.report is None:
            return
        sid = None
        if self.table_view.get() == "横向比较":
            column = self.result_tree.identify_column(event.x)
            if column and column != "#0":
                sid = self.result_tree["columns"][int(column[1:])-1]
        else:
            row = self.result_tree.identify_row(event.y)
            sid = row.split(":")[0] if row else None
        if sid:
            self._show_result_context([s for s in [self.report["baseline"]] + self.report["samples"] if s["sample_id"] == sid])

    def _base_changed(self, *_args):
        is_pr = self.base_mode.get() == "PR 样本"
        self.base_combo.configure(state="readonly" if is_pr else "disabled")
        self.column_combo.configure(state="readonly" if is_pr else "disabled")
        index = self.base_combo.current()
        choices = ["平均轮廓"]
        if 0 <= index < len(self.samples):
            choices += [f"PR{c['label']}" for c in self.samples[index]["columns"]]
        self.column_combo.configure(values=choices)
        self.base_column.set("平均轮廓")

    def _refresh_samples(self):
        self.sample_tree.delete(*self.sample_tree.get_children())
        for i, sample in enumerate(self.samples):
            morph = sample_morphology(sample)
            self.sample_tree.insert("", "end", iid=str(i), text=sample["name"],
                                    values=(f"{sample['nm_per_px']:.6g}", f"{morph['included_count']}/{len(sample['columns'])}",
                                            sample["mode"], sample["source"]))
        names = [f"{i+1}. {s['name']}" for i,s in enumerate(self.samples)]
        old = self.base_combo.current()
        self.base_combo.configure(values=names)
        if names:
            self.base_combo.current(min(max(old, 0), len(names)-1))
        else:
            self.base_name.set("")
        self._base_changed()
        self._invalidate()

    def _upsert_sample(self, sample):
        import copy
        for old in self.samples:
            if (old["source"], old["mode"]) == (sample["source"], sample["mode"]):
                old.update(copy.deepcopy(sample), name=old["name"])
                return
        self.samples.append(copy.deepcopy(sample))

    def _add_cached(self):
        if self.busy:
            return
        for sample in self.app._comparison_samples.values():
            self._upsert_sample(sample)
        self._refresh_samples()

    def _remove_samples(self):
        if self.busy:
            return
        for index in sorted((int(i) for i in self.sample_tree.selection()), reverse=True):
            del self.samples[index]
        self._refresh_samples()

    def _edit_sample(self):
        if self.busy or self.app.running or len(self.sample_tree.selection()) != 1:
            return
        sample = self.samples[int(self.sample_tree.selection()[0])]
        dialog = tk.Toplevel(self)
        dialog.title("样本名称与比例尺")
        dialog.transient(self)
        name, scale = tk.StringVar(value=sample["name"]), tk.StringVar(value=str(sample["nm_per_px"]))
        for row, title, var in ((0, "PR 类型 / 名称", name), (1, "该图 nm/px", scale)):
            ttk.Label(dialog, text=title).grid(row=row, column=0, padx=12, pady=10)
            ttk.Entry(dialog, textvariable=var, width=38).grid(row=row, column=1, padx=12, pady=10)
        def save():
            try:
                value = float(scale.get())
                if not math.isfinite(value) or value <= 0 or not name.get().strip():
                    raise ValueError("名称不能为空，nm/px 必须为正数")
                current_scale = read_image_scale(sample["source"])
                self.app._apply_image_scale(sample["source"], {
                    "pixels": current_scale["pixels"],
                    "nm": value * current_scale["pixels"],
                })
                sample.update(name=name.get().strip(), nm_per_px=value)
                cached = self.app._comparison_samples.get((sample["source"], sample["mode"]))
                if cached:
                    cached["name"] = sample["name"]
                self._refresh_samples()
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("参数错误", str(exc), parent=dialog)
        ttk.Button(dialog, text="保存", command=save).grid(row=2, column=1, sticky="e", padx=12, pady=10)
        dialog.grab_set()

    def _add_images(self):
        if self.busy or self.app.running:
            messagebox.showinfo("正在分析", "请等待当前分析完成。", parent=self)
            return
        paths = filedialog.askopenfilenames(parent=self, title="选择多款 PR 图像", filetypes=[("图像", "*.tif *.tiff *.png *.jpg *.bmp"), ("所有文件", "*.*")])
        if not paths:
            return
        try:
            self.app._collect_params()
            scales = {path: read_image_scale(path) for path in paths}
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc), parent=self)
            return
        self.busy = True
        self.app._set_running(True)
        for button in self.buttons:
            button.configure(state="disabled")
        self.status.set("正在识别所选图片，并保存各阶段 test 过程图…")
        def worker():
            errors = []
            try:
                for path in paths:
                    self.messages.put(("status", f"正在识别 {os.path.basename(path)}…"))
                    try:
                        payload = process_image(path, scale=scales[path])
                        self.messages.put(("payload", payload))
                    except Exception as exc:
                        errors.append(f"{os.path.basename(path)}：{exc}")
            finally:
                self.messages.put(("done", errors))
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, value = self.messages.get_nowait()
                if kind == "payload":
                    self.app._cache_payload(value)
                    self._upsert_sample(comparison_sample(value))
                elif kind == "status":
                    self.status.set(value)
                elif kind == "done":
                    self.busy = False
                    self.app._set_running(False)
                    for button in self.buttons:
                        button.configure(state="normal")
                    self._refresh_samples()
                    self.status.set("导入完成。请核对每张图的比例尺后计算对比。" + (" 失败：" + "；".join(value) if value else ""))
                    self.app._resume_pending_work()
        except queue.Empty:
            pass
        self._poll_job = self.after(100, self._poll)

    def _compare(self):
        if self.busy or self.app.running:
            return
        try:
            if self.base_mode.get() == "标准方波":
                baseline = square_baseline(float(self.cd.get()), float(self.height.get()))
                samples = self.samples
            else:
                index = self.base_combo.current()
                if index < 0:
                    raise ValueError("请选择 PR baseline")
                baseline = sample_morphology(self.samples[index], self.base_column.get())
                samples = [s for i,s in enumerate(self.samples) if i != index]
            if not samples:
                raise ValueError("请至少添加一个待比较样本（PR baseline 需要另一项样本）")
            results, excluded = [], []
            for sample in samples:
                result = sample_morphology(sample)
                if result["included_count"]:
                    results.append(result)
                else:
                    excluded.append(sample["name"])
            if not results:
                raise ValueError("所有样本均无完整柱。请在主界面用 ROI 识别后添加已分析结果。")
            report = compare_morphologies(baseline, results)
            report["skipped_samples"] = excluded
            folder = export_baseline_for_app(report)
            self.report, self.report_dir = report, folder
            self.result_sample_combo.configure(state="readonly", values=["全部样本"] + [_sample_caption(s, 70) for s in report["samples"]])
            self.result_sample_combo.current(0)
            self._refresh_result_views()
            self.status.set(f"已对比 {len(results)} 款样本；图表已保存到软件同级 baseline_comparison/{os.path.basename(folder)}。" + (" 跳过无完整柱样本：" + "、".join(excluded) if excluded else ""))
        except Exception as exc:
            messagebox.showerror("无法完成对比", str(exc), parent=self)

    def _open_folder(self):
        if self.report_dir and os.path.isdir(self.report_dir):
            os.startfile(self.report_dir)

    def _close(self):
        if self.busy:
            self.status.set("正在识别，完成后可关闭窗口。")
            return
        self.after_cancel(self._poll_job)
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE + " · 柱体识别与精密测量")
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1600, screen_w - 60)}x{min(960, screen_h - 90)}")
        self.minsize(min(1120, screen_w - 60), min(680, screen_h - 90))

        self.q = queue.Queue()
        self.running = False
        self.last_dir = core.INPUT_DIR

        self._orig_bgr = None
        self._tool_bgr = None
        self._user_bgr = None
        self._photos = []
        self._scale = 1.0
        self._img_x = 0
        self._img_y = 0
        self._ow = 0
        self._oh = 0
        self._resize_job = None
        # 最近一次 _tool 综合图包含的柱子数量，用于在 _user 图先到达时
        # 预估 _tool 综合图的固有拼图尺寸，保证缩放比稳定不跳变
        self._n_columns = 0

        self._roi = None
        self._drag = None

        self._m1 = None
        self._m2 = None
        # 直线量测拖拽状态。_measure_drag 为 0/1 时表示正在拖动 A/B。
        self._measure_drag = None
        self._measure_creating = False
        self._measure_moved = False
        self._measure_preview_point = None
        self._line_profile_data = None
        self._line_profile_selected_x = None
        self._m2_preview = None

        # 灰度曲线：模式（直线 / 横向行 / 纵向列）、选点、量测端点
        self._profile_meta = {"mode": "line"}
        self._profile_pick = None
        self._profile_span = []
        # Matplotlib 曲线上的可拖拽对象："pick"、"span0" 或 "span1"。
        self._profile_drag = None

        self._ang = []

        self._last_img_name = None
        self._analysis_cache = {"tool": None, "user": None}
        self._file_cache = {}
        self._last_image_path = None
        self._loading_scale = False
        self._pending_scale_paths = set()
        self._pending_scale_edits = {}
        self._file_list_paths = {}
        self._comparison_samples = {}
        self._baseline_window = None
        self._help_window = None
        self._display_scale = validate_scale()
        self._last_stat_src = {"tool": None, "user": None}
        self._analyzed = []

        self._live_job = None
        self._live_dirty = False
        self._scale_job = None
        self._scale_dirty = False

        self._build_ui()
        self._refresh_files()
        self._load_selected_scale()
        self.combo.bind("<<ComboboxSelected>>", self._load_selected_scale)

        # 识别参数变化 -> 防抖重识别
        for var in self.spins.values():
            var.trace_add("write", self._schedule_live)

        # 比例尺变化 -> 只重导出/重统计，不做检测
        for var in self.scale_vars.values():
            var.trace_add("write", self._on_scale_var_change)

        self._queue_job = self.after(80, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------

    def _build_ui(self):
        try:
            ttk.Style(self).theme_use("clam")
        except Exception:
            pass

        try:
            st = ttk.Style(self)
            st.configure(".", font=("Microsoft YaHei UI", 9), background="#f3f6fa", foreground="#20334a")
            st.configure("TButton", padding=(10, 6))
            st.configure("Accent.TButton", background="#2463aa", foreground="white", font=("Microsoft YaHei UI", 9, "bold"))
            st.map("Accent.TButton", background=[("active", "#184d8a"), ("disabled", "#bbc6d4")])
            st.configure("Treeview", rowheight=27, background="white", fieldbackground="white")
            st.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"), padding=5)
            st.configure("TNotebook.Tab", padding=(16, 7))
            st.configure(
                "TLabelframe.Label",
                font=("Microsoft YaHei UI", 9, "bold"),
                foreground="#175a7a"
            )
        except Exception:
            pass

        menubar = tk.Menu(self, tearoff=0)
        menubar.add_command(label="Baseline 对比", command=self._open_baseline)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="快速上手", command=lambda: self._show_doc("快速上手", QUICKSTART_HELP))
        help_menu.add_command(label="每图比例尺", command=lambda: self._show_doc("每图比例尺", SCALE_HELP))
        help_menu.add_command(label="Baseline 对比指南", command=lambda: self._show_doc("Baseline 对比", BASELINE_HELP))
        help_menu.add_separator()
        help_menu.add_command(
            label="识别流程（含纵向扫描）",
            command=self._show_pipeline
        )
        help_menu.add_command(label="参数说明", command=self._show_params_help)
        help_menu.add_command(
            label="统计公式",
            command=lambda: self._show_doc("统计公式", FORMULA_HELP)
        )
        help_menu.add_command(
            label="Profile 计算公式",
            command=lambda: self._show_doc("Profile 计算公式", PROFILE_FORMULA_HELP)
        )
        help_menu.add_command(
            label="step1–10 算法详解",
            command=lambda: self._show_doc("step1–10 算法详解", STEP_HELP)
        )
        help_menu.add_separator()
        help_menu.add_command(
            label="形貌判定说明",
            command=lambda: self._show_doc("形貌判定说明", SHAPE_HELP)
        )
        help_menu.add_command(
            label="灰度曲线与量测说明",
            command=lambda: self._show_doc("灰度曲线与量测说明", PROFILE_HELP)
        )
        help_menu.add_command(
            label="输出文件说明",
            command=lambda: self._show_doc("输出文件说明", FILE_TAG_HELP)
        )
        help_menu.add_separator()
        help_menu.add_command(label="关于", command=self._show_about)
        self.config(menu=menubar)

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Label(top, text="图像文件夹：").pack(side="left")

        default_dir = core.INPUT_DIR
        if getattr(sys, "frozen", False):
            default_dir = os.path.dirname(os.path.abspath(sys.executable))

        self.var_dir = tk.StringVar(value=default_dir)
        self.entry_dir = ttk.Entry(top, textvariable=self.var_dir, width=32)
        self.entry_dir.pack(side="left", padx=4)

        ttk.Button(top, text="浏览文件夹", command=self._browse_dir).pack(side="left")
        ttk.Button(top, text="选择 tif", command=self._browse_file).pack(side="left", padx=4)

        self.var_sel = tk.StringVar()
        self.combo = ttk.Combobox(
            top,
            textvariable=self.var_sel,
            state="readonly",
            width=18
        )
        self.combo.pack(side="left", padx=(8, 2))

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(actions, text=APP_TITLE, font=("Microsoft YaHei UI", 12, "bold"), foreground="#2463aa").pack(side="left", padx=(0, 18))
        self.btn_run1 = ttk.Button(
            actions,
            style="Accent.TButton",
            text="▶ 分析选中图像",
            command=self._start_single
        )
        self.btn_run1.pack(side="left", padx=4)

        self.btn_run_all = ttk.Button(
            actions,
            text="批量分析全部",
            command=self._start_batch
        )
        self.btn_run_all.pack(side="left", padx=2)

        self.btn_open = ttk.Button(
            actions,
            text="打开输出文件夹",
            command=self._open_output
        )
        self.btn_open.pack(side="left", padx=4)

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(2, 4))

        # 左
        left = ttk.Frame(body, width=300)
        body.add(left, weight=0)
        left.pack_propagate(False)

        pf = ttk.LabelFrame(left, text="识别参数")
        pf.pack(fill="x", padx=4, pady=(2, 6))

        self.spins = {}

        for i, spec in enumerate(MAIN_PARAMS):
            attr, name, kind, lo, hi, inc = spec
            row = i
            col = 0

            cell = ttk.Frame(pf)
            cell.grid(row=row, column=col, sticky="w", padx=6, pady=3)

            ttk.Label(cell, text=name).pack(side="left")

            var = tk.StringVar(value=str(getattr(core, attr)))
            tk.Spinbox(
                cell,
                from_=lo,
                to=hi,
                increment=inc,
                width=7,
                textvariable=var,
                font=("Segoe UI", 10)
            ).pack(side="left", padx=3)

            self.spins[attr] = var

        ttk.Label(
            pf,
            text=(
                "binary 粗定位 → 行列灰度精修。"
                "自动识别强制排除左右最外两根，包含完整柱；ROI 保留全部。"
            ),
            foreground="#666666",
            wraplength=266,
            justify="left"
        ).grid(
            row=len(MAIN_PARAMS),
            column=0,
            columnspan=2,
            sticky="w",
            padx=8,
            pady=(4, 6)
        )

        polarity_row = ttk.Frame(left)
        polarity_row.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(polarity_row, text="柱体明暗").pack(side="left")
        self.var_polarity = tk.StringVar(value={"bright": "亮柱（推荐）", "dark": "暗柱", "auto": "自动（需核对）"}[core.PILLAR_POLARITY])
        self.combo_polarity = ttk.Combobox(polarity_row, textvariable=self.var_polarity, state="readonly", width=18,
                                          values=("亮柱（推荐）", "暗柱", "自动（需核对）"))
        self.combo_polarity.pack(side="left", padx=6)
        self.var_polarity.trace_add("write", self._schedule_live)
        sf = ttk.LabelFrame(left, text="当前选中图 · 比例尺与输出")
        sf.pack(fill="x", padx=4)

        self.scale_vars = {}
        self.scale_widgets = []

        for attr, name, lo, hi, inc in [
            ("SCALE_PIXELS", "比例尺长度(px)", 1, 100000, 1),
            ("SCALE_NM", "实际长度(nm)", 0.01, 100000, 0.5),
        ]:
            cell = ttk.Frame(sf)
            cell.pack(fill="x", padx=8, pady=3)

            ttk.Label(cell, text=name).pack(side="left")

            var = tk.StringVar(value=str(getattr(core, attr)))
            spin = tk.Spinbox(
                cell,
                from_=lo,
                to=hi,
                increment=inc,
                width=10,
                textvariable=var,
                font=("Segoe UI", 10)
            )
            spin.pack(side="left", padx=4)
            self.scale_widgets.append(spin)

            self.scale_vars[attr] = var

        self.lbl_scale_info = ttk.Label(sf, text="")
        self.lbl_scale_info.pack(anchor="w", padx=8, pady=(0, 3))

        self.var_subdir = tk.BooleanVar(value=bool(core.OUTPUT_SUBDIR))
        ttk.Checkbutton(
            sf,
            text="常规结果也放入图片同名文件夹",
            variable=self.var_subdir
        ).pack(anchor="w", padx=8)

        self.var_debug = tk.BooleanVar(value=bool(core.DEBUG_OUTPUT))
        ttk.Checkbutton(
            sf,
            text="ROI 保存过程图（自动识别固定保存）",
            variable=self.var_debug
        ).pack(anchor="w", padx=8, pady=(0, 4))

        ttk.Label(
            sf,
            text="下方双击文件可标定。过程图：图片名/test；对比报告：软件同级 baseline_comparison。",
            foreground="#777777",
            wraplength=266,
            justify="left"
        ).pack(anchor="w", padx=8, pady=(2, 6))

        ttk.Button(sf, text="打开识别过程 test 文件夹", command=self._open_process).pack(fill="x", padx=8, pady=5)
        self._update_scale_info()

        hf = ttk.LabelFrame(left, text="已分析文件")
        hf.pack(fill="both", expand=True, padx=4, pady=(8, 4))

        file_body = ttk.Frame(hf)
        file_body.pack(fill="both", expand=True, padx=5, pady=4)
        self.lst_files = ttk.Treeview(file_body, columns=("scale",), height=5, selectmode="browse")
        self.lst_files.heading("#0", text="图像")
        self.lst_files.column("#0", width=165, minwidth=85, stretch=True)
        self.lst_files.heading("scale", text="nm / px")
        self.lst_files.column("scale", width=76, minwidth=65, stretch=False, anchor="e")
        file_scroll = ttk.Scrollbar(file_body, orient="vertical", command=self.lst_files.yview)
        self.lst_files.configure(yscrollcommand=file_scroll.set)
        self.lst_files.pack(side="left", fill="both", expand=True)
        file_scroll.pack(side="right", fill="y")
        self.lst_files.bind("<<TreeviewSelect>>", self._on_pick_file)
        self.lst_files.bind("<Double-1>", self._edit_image_scale)
        self.btn_image_scale = ttk.Button(hf, text="设置所选图比例尺…", command=self._edit_image_scale)
        self.btn_image_scale.pack(fill="x", padx=6, pady=(0, 5))

        # 中
        mid = ttk.Frame(body)
        body.add(mid, weight=1)
        self.frm_mid = mid

        tools = ttk.Frame(mid)
        tools.pack(fill="x", pady=(0, 2))
        roi_actions = ttk.Frame(mid)
        roi_actions.pack(fill="x", pady=(0, 3))

        self.var_canvas_tool = tk.StringVar(value="roi")

        for text, value in [
            ("框选区域", "roi"),
            ("直线量测", "dist"),
            ("角度量测", "angle"),
        ]:
            ttk.Radiobutton(
                tools,
                text=text,
                value=value,
                variable=self.var_canvas_tool,
                command=self._on_tool_change
            ).pack(side="left", padx=(0, 10))

        self.btn_roi_run = ttk.Button(
            roi_actions,
            text="分析框选区域",
            command=self._start_roi
        )
        self.btn_roi_run.pack(side="right")

        self.btn_roi_clear = ttk.Button(
            roi_actions,
            text="清除标记",
            command=self._clear_canvas_marks
        )
        self.btn_roi_clear.pack(side="right", padx=8)

        self.lbl_roi_info = ttk.Label(
            mid,
            text=(
                "提示：拖出 ROI；直线量测点 A、B 后可拖动端点，"
                "右侧灰度曲线也支持点击和拖拽。"
            ),
            foreground="#555555"
        )
        self.lbl_roi_info.pack(fill="x", pady=(0, 2))

        workspace = ttk.Frame(mid)
        workspace.pack(fill="both", expand=True)
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1, uniform="images")
        workspace.rowconfigure(1, weight=1, uniform="images")
        orig_row = ttk.Panedwindow(workspace, orient="horizontal")
        orig_row.grid(row=0, column=0, sticky="nsew")

        self.frm_orig = ttk.LabelFrame(orig_row, text="原始图像")
        orig_row.add(self.frm_orig, weight=1)

        self.cv_orig = tk.Canvas(
            self.frm_orig,
            bg="#f2f4f7",
            highlightthickness=0,
            cursor="crosshair"
        )
        self.cv_orig.pack(fill="both", expand=True)
        self._bind_canvas()

        self.frm_line_profile = None
        if MATPLOTLIB_AVAILABLE:
            self.frm_line_profile = ttk.LabelFrame(
                orig_row,
                text="实时灰度曲线"
            )
            orig_row.add(self.frm_line_profile, weight=1)

            ctl = ttk.Frame(self.frm_line_profile)
            ctl.pack(fill="x", padx=4, pady=(4, 0))
            pos_ctl = ttk.Frame(self.frm_line_profile)
            pos_ctl.pack(fill="x", padx=4, pady=2)

            self.var_profile_mode = tk.StringVar(value="直线剖面")
            ttk.Combobox(
                ctl,
                textvariable=self.var_profile_mode,
                state="readonly",
                width=11,
                values=("直线剖面", "横向行剖面", "纵向列剖面")
            ).pack(side="left")
            self.var_profile_mode.trace_add("write", self._on_profile_mode_change)

            ttk.Label(pos_ctl, text="行 y / 列 x：").pack(side="left", padx=(8, 0))
            self.var_profile_pos = tk.StringVar(value="0")
            self.spn_profile_pos = tk.Spinbox(
                pos_ctl,
                from_=0,
                to=0,
                width=6,
                textvariable=self.var_profile_pos,
                command=self._on_profile_pos_change
            )
            self.spn_profile_pos.pack(side="left")
            self.spn_profile_pos.bind("<Return>", self._on_profile_pos_edit)
            self.spn_profile_pos.bind("<FocusOut>", self._on_profile_pos_edit)

            ttk.Button(
                ctl,
                text="清除量测",
                command=self._clear_profile_measure
            ).pack(side="right")

            self._line_profile_fig = Figure(
                figsize=(4, 2.2),
                dpi=100,
                facecolor="#f7f9fc",
                edgecolor="#d6dee8",
            )
            self._line_profile_ax = self._line_profile_fig.add_subplot(111)
            self._line_profile_canvas = FigureCanvasTkAgg(
                self._line_profile_fig,
                master=self.frm_line_profile
            )
            self._line_profile_canvas.get_tk_widget().configure(
                background="#f7f9fc",
                highlightthickness=0,
            )
            self._line_profile_canvas.get_tk_widget().pack(fill="both", expand=True)
            self._line_profile_canvas.mpl_connect(
                "button_press_event",
                self._on_profile_click
            )
            self._line_profile_canvas.mpl_connect(
                "motion_notify_event",
                self._on_profile_motion
            )
            self._line_profile_canvas.mpl_connect(
                "button_release_event",
                self._on_profile_release
            )

            self.lbl_profile_info = ttk.Label(
                self.frm_line_profile,
                text=(
                    "曲线交互：单击定位；Shift+单击/右键选两点；"
                    "拖动橙色端点可同步调整图片测量线。"
                ),
                foreground="#555555",
                wraplength=420,
                justify="left"
            )
            self.lbl_profile_info.pack(fill="x", padx=4, pady=(0, 4))

        result_row = ttk.Frame(workspace)
        result_row.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        self.frm_tool = ttk.LabelFrame(result_row, text="自动识别")
        self.frm_tool.pack(side="left", fill="both", expand=True, padx=(0, 2))
        # 锁定面板尺寸，稍后动态设置为正方形
        self.frm_tool.pack_propagate(False)

        self.lbl_tool = tk.Label(
            self.frm_tool,
            text="（尚无自动识别结果）",
            bg="white",
            fg="#999999",
            anchor="nw",
        )
        # 固定锚点为左上角，避免与 _user 预览尺寸不同时位置跳动
        self.lbl_tool.pack(fill="both", expand=True)
        # 锁定面板尺寸：只随窗口布局变化，不随图片请求尺寸反馈伸缩，
        # 否则"大图撑大面板->缩放变小->面板回弹"会循环，预览忽大忽小
        self.frm_tool.pack_propagate(False)

        ttk.Button(
            self.frm_tool,
            text="保存预览图…",
            command=lambda: self._save_preview_img("tool")
        ).pack(pady=3)

        self.frm_user = ttk.LabelFrame(result_row, text="ROI 识别")
        self.frm_user.pack(side="left", fill="both", expand=True, padx=(2, 0))
        # 锁定面板尺寸，稍后动态设置为正方形
        self.frm_user.pack_propagate(False)

        self.lbl_user = tk.Label(
            self.frm_user,
            text="（尚无人工识别结果）",
            bg="white",
            fg="#999999",
            anchor="nw",
        )
        # 固定锚点为左上角，与 _tool 预览保持一致
        self.lbl_user.pack(fill="both", expand=True)
        # 与 frm_tool 相同：锁定面板尺寸，保证两预览区域大小恒定
        self.frm_user.pack_propagate(False)

        ttk.Button(
            self.frm_user,
            text="保存预览图…",
            command=lambda: self._save_preview_img("user")
        ).pack(pady=3)

        mid.bind("<Configure>", self._on_mid_resize)
        # 初始化预览区为正方形
        self._resize_preview_squares()

        # 右
        right = ttk.Notebook(body, width=390)
        body.add(right, weight=0)
        right.pack_propagate(False)

        self._stat_blocks = {}
        for kind, title in [
            ("tool", "工具自动识别统计（_tool）"),
            ("user", "人工 ROI 统计（_user）"),
        ]:
            blk = self._make_stats_block(right, kind, title)
            right.add(blk["frame"], text="自动统计" if kind == "tool" else "ROI 统计")
            blk["save"].configure(
                command=lambda k=kind: self._save_stats_txt(k)
            )
            self._stat_blocks[kind] = blk

        # 底
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom", padx=10, pady=(0, 6))

        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(bar, textvariable=self.var_status).pack(side="left")

        self.pbar = ttk.Progressbar(
            bar,
            mode="determinate",
            length=260,
            maximum=100
        )
        self.pbar.pack(side="right", padx=(10, 4))

        self.lbl_pct = ttk.Label(bar, text="0%", width=5)
        self.lbl_pct.pack(side="right")

    def _make_stats_block(self, parent, kind, title):
        frame = ttk.LabelFrame(parent, text=title)

        summary_var = tk.StringVar(value="（尚无结果）")
        ttk.Label(
            frame,
            textvariable=summary_var,
            foreground="#555555",
            wraplength=345
        ).pack(anchor="w", padx=6, pady=(2, 0))

        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True, padx=4, pady=2)

        tree = ttk.Treeview(
            body,
            columns=("avg",),
            show="tree headings",
            selectmode="none"
        )

        tree.heading("#0", text="计算项")
        tree.column("#0", width=170, minwidth=130, anchor="w", stretch=True)

        ys = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        xs = ttk.Scrollbar(body, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)

        tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")

        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

        foot = ttk.Frame(frame)
        foot.pack(fill="x", padx=4, pady=(0, 3))

        save = ttk.Button(foot, text="保存统计表另存为…")
        save.pack(side="left")

        return {
            "frame": frame,
            "tree": tree,
            "var": summary_var,
            "save": save,
        }

    def _folder(self):
        f = self.var_dir.get().strip().strip('"')
        return os.path.normpath(f or core.INPUT_DIR)

    def _list_tifs(self):
        folder = self._folder()
        if not os.path.isdir(folder):
            return []
        return sorted({
            p
            for ext in core.INPUT_GLOBS
            for p in glob.glob(os.path.join(folder, ext))
        })

    def _refresh_files(self):
        names = [os.path.basename(p) for p in self._list_tifs()]
        self.combo["values"] = names

        if names and self.var_sel.get() not in names:
            self.var_sel.set(names[0])
        elif not names:
            self.var_sel.set("")

        self._scan_existing_results()
        self._load_selected_scale()

    def _scan_existing_results(self):
        folder = self._folder()
        if not os.path.isdir(folder):
            return

        for p in self._list_tifs():
            stem = os.path.splitext(os.path.basename(p))[0]
            cand_dirs = [folder, os.path.join(folder, stem)]
            found = False

            for d in cand_dirs:
                if not os.path.isdir(d):
                    continue

                for fn in os.listdir(d):
                    if fn.startswith(stem + "_") and fn.endswith("_result.png"):
                        found = True
                        break

                if found:
                    break

            if found and os.path.abspath(p) not in self._analyzed:
                self._analyzed.append(os.path.abspath(p))

        self._refresh_analyzed_list()

    def _refresh_analyzed_list(self):
        self._file_list_paths = {}
        for i, path in enumerate(self._analyzed):
            if os.path.dirname(image_key(path)) != image_key(self._folder()):
                continue
            scale = read_image_scale(path)
            iid = str(i)
            self._file_list_paths[iid] = path
            options = dict(text=os.path.basename(path), values=(f"{scale['nm']/scale['pixels']:.6g}",))
            if self.lst_files.exists(iid):
                self.lst_files.item(iid, **options)
            else:
                self.lst_files.insert("", "end", iid=iid, **options)
        for iid in self.lst_files.get_children():
            if iid not in self._file_list_paths:
                self.lst_files.delete(iid)

    def _selected_image_path(self):
        name = self.var_sel.get()
        return os.path.abspath(os.path.join(self._folder(), name)) if name else self._last_image_path

    def _set_scale_fields(self, scale):
        self._loading_scale = True
        try:
            self.scale_vars["SCALE_PIXELS"].set(f"{scale['pixels']:.12g}")
            self.scale_vars["SCALE_NM"].set(f"{scale['nm']:.12g}")
            self._update_scale_info()
        finally:
            self._loading_scale = False

    def _load_selected_scale(self, _event=None):
        if self._pending_scale_edits and not self.running:
            self._flush_scale_edits(refresh=False)
            self.after(80, self._run_scale_refresh)
        if hasattr(self, "scale_vars") and self._selected_image_path():
            self._set_scale_fields(read_image_scale(self._selected_image_path()))

    def _scale_from_fields(self):
        def number(key):
            return float(self.scale_vars[key].get().strip().replace("，", ".").replace(",", "."))
        return validate_scale({"pixels": number("SCALE_PIXELS"), "nm": number("SCALE_NM")})

    def _apply_image_scale(self, path, scale, refresh=True):
        if self.running:
            raise ValueError("请等待当前分析完成后修改比例尺")
        path = os.path.abspath(path)
        scale = save_image_scale(path, scale)
        if self._last_image_path and image_key(path) == image_key(self._last_image_path):
            self._display_scale = scale
        if self._selected_image_path() and image_key(path) == image_key(self._selected_image_path()):
            self._set_scale_fields(scale)
        self._pending_scale_paths.add(path)
        for sample in self._comparison_samples.values():
            if image_key(sample["source"]) == image_key(path):
                sample["nm_per_px"] = scale["nm"] / scale["pixels"]
        win = self._baseline_window
        if win is not None and win.winfo_exists():
            for sample in win.samples:
                if image_key(sample["source"]) == image_key(path):
                    sample["nm_per_px"] = scale["nm"] / scale["pixels"]
            win._refresh_samples()
        self._refresh_analyzed_list()
        if refresh:
            self._run_scale_refresh()
        self._refresh_line_profile_live()

    def _flush_scale_edits(self, refresh=True):
        if self._scale_job is not None:
            self.after_cancel(self._scale_job)
            self._scale_job = None
        if self.running:
            self._scale_dirty = True
            return
        for key, (path, scale) in list(self._pending_scale_edits.items()):
            self._apply_image_scale(path, scale, refresh=False)
            del self._pending_scale_edits[key]
        if refresh:
            self._run_scale_refresh()

    def _edit_image_scale(self, _event=None):
        if self.running:
            return
        selected = self.lst_files.selection()
        path = self._file_list_paths.get(selected[0]) if selected else self._selected_image_path()
        if not path:
            return
        scale = read_image_scale(path)
        dialog = tk.Toplevel(self)
        dialog.title("本图比例尺 · " + os.path.basename(path))
        dialog.transient(self)
        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=os.path.basename(path), font=("Microsoft YaHei UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        px, nm = tk.StringVar(value=f"{scale['pixels']:g}"), tk.StringVar(value=f"{scale['nm']:g}")
        for row, title, variable in ((1,"标尺像素长度 (px)",px),(2,"标尺实际长度 (nm)",nm)):
            ttk.Label(frame,text=title).grid(row=row,column=0,sticky="w",pady=6,padx=(0,12))
            ttk.Entry(frame,textvariable=variable,width=16).grid(row=row,column=1,pady=6)
        ttk.Label(frame,text="只更新这张图的自动 / ROI 结果和 Baseline 数据。\n标定保存在图片目录，重开软件仍有效。",foreground="#5b6b7d").grid(row=3,column=0,columnspan=2,sticky="w",pady=12)
        def save():
            try:
                self._apply_image_scale(path,{"pixels":float(px.get()),"nm":float(nm.get())})
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("无法保存比例尺",str(exc),parent=dialog)
        ttk.Button(frame,text="保存本图标定",command=save,style="Accent.TButton").grid(row=4,column=1,sticky="e")
        dialog.grab_set()

    def _browse_dir(self):
        if self.running:
            return
        d = filedialog.askdirectory(
            initialdir=self._folder(),
            title="选择包含 tif 图像的文件夹"
        )
        if d:
            self.var_dir.set(os.path.normpath(d))
            self._refresh_files()

    def _browse_file(self):
        if self.running:
            return
        f = filedialog.askopenfilename(
            initialdir=self._folder(),
            title="选择 tif 图像",
            filetypes=[
                ("TIF 图像", "*.tif *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if f:
            self.var_dir.set(os.path.dirname(f))
            self._refresh_files()
            self.var_sel.set(os.path.basename(f))
            self._load_selected_scale()

    def _on_pick_file(self, _evt=None):
        if self.running:
            return
        selected = self.lst_files.selection()
        if not selected:
            return
        path = self._file_list_paths.get(selected[0])
        if not path:
            return
        self.var_sel.set(os.path.basename(path))
        self._load_selected_scale()
        caches = self._file_cache.get(image_key(path), {})
        if caches:
            for payload in list(caches.values()):
                self._set_preview(payload["tag"], payload["img"], payload["paint"], os.path.basename(path), payload)
        else:
            # 列表仍可编辑标定，不因选择一张历史文件立即启动阻塞性的识别。
            self.var_status.set(f"已选中 {os.path.basename(path)}，可设置本图比例尺；点击“分析选中图像”查看新结果。")

    def _collect_params(self):
        folder = self._folder()
        if not os.path.isdir(folder):
            raise ValueError("图像文件夹不存在")

        core.INPUT_DIR = folder
        core.OUTPUT_SUBDIR = bool(self.var_subdir.get())
        core.DEBUG_OUTPUT = bool(self.var_debug.get())
        core.PILLAR_POLARITY = {"亮柱（推荐）": "bright", "暗柱": "dark", "自动（需核对）": "auto"}[self.var_polarity.get()]

        for attr, name, kind, *_ in MAIN_PARAMS:
            raw = self.spins[attr].get().strip().replace("，", ".").replace(",", ".")
            value = int(float(raw)) if kind == "int" else float(raw)

            if not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限数值")
            if attr == "AUTO_CD_ESTIMATE_PX" and value < 0:
                raise ValueError("估计 Line CD 只能填 0 或正数")
            if attr == "AUTO_CD_TOLERANCE_PX" and value <= 0:
                raise ValueError("允许误差必须大于 0 px")

            setattr(core, attr, value)

        self._collect_scale_only()

    def _collect_scale_only(self):
        self._flush_scale_edits(refresh=False)
        scale = self._scale_from_fields()
        path = self._selected_image_path()
        if path:
            previous = read_image_scale(path)
            if previous != scale:
                self._apply_image_scale(path, scale, refresh=False)
        core.INPUT_DIR = self._folder()
        core.OUTPUT_SUBDIR = bool(self.var_subdir.get())

    def _update_scale_info(self):
        try:
            px = float(self.scale_vars["SCALE_PIXELS"].get())
            nm = float(self.scale_vars["SCALE_NM"].get())
            if px > 0:
                self.lbl_scale_info.configure(
                    text=f"换算：1 px = {nm / px:.6f} nm"
                )
                return
        except Exception:
            pass

        self.lbl_scale_info.configure(text="换算：请输入有效数值")

    # -----------------------------------------------------------------
    # 实时识别 / 比例尺轻量刷新
    # -----------------------------------------------------------------

    def _params_are_valid(self):
        for var in list(self.spins.values()) + list(self.scale_vars.values()):
            try:
                value = float(
                    var.get().strip()
                    .replace("，", ".")
                    .replace(",", ".")
                )
                if not math.isfinite(value):
                    return False
            except Exception:
                return False
        try:
            if float(self.spins["AUTO_CD_ESTIMATE_PX"].get()) < 0:
                return False
            if float(self.spins["AUTO_CD_TOLERANCE_PX"].get()) <= 0:
                return False
            if float(self.scale_vars["SCALE_PIXELS"].get()) <= 0:
                return False
            if float(self.scale_vars["SCALE_NM"].get()) <= 0:
                return False
        except Exception:
            return False
        return True

    def _schedule_live(self, *_args):
        if not self._last_img_name:
            return

        if not self._params_are_valid():
            return

        if self.running:
            self._live_dirty = True
            return

        if self._live_job is not None:
            self.after_cancel(self._live_job)

        self._live_job = self.after(320, self._run_live)

    def _run_live(self):
        self._live_job = None

        if self.running or not self._last_img_name:
            return

        p = self._last_image_path
        if not os.path.isfile(p):
            return

        self._start_jobs(
            [{"path": p, "mode": "full", "roi": None}],
            f"参数变化 → 自动重识别：{self._last_img_name}"
        )

    def _on_scale_var_change(self, *_args):
        if self._loading_scale:
            return
        self._update_scale_info()
        path = self._selected_image_path()
        if not path:
            return
        try:
            scale = self._scale_from_fields()
        except ValueError:
            self._pending_scale_edits.pop(image_key(path), None)
            if self._scale_job is not None:
                self.after_cancel(self._scale_job)
                self._scale_job = None
            return
        self._pending_scale_edits[image_key(path)] = (path, scale)
        if self.running:
            self._scale_dirty = True
            return
        if self._scale_job is not None:
            self.after_cancel(self._scale_job)
        def apply():
            self._scale_job = None
            try:
                self._flush_scale_edits()
            except Exception as exc:
                self.var_status.set("标定未保存：" + str(exc))
        self._scale_job = self.after(500, apply)

    def _run_scale_refresh(self):
        if self.running:
            self._scale_dirty = True
            return
        paths = list(self._pending_scale_paths)
        self._pending_scale_paths.clear()
        if not paths:
            return
        jobs = [(payload, read_image_scale(path)) for path in paths
                for payload in self._file_cache.get(image_key(path), {}).values()]
        if not jobs:
            self.var_status.set("本图比例尺已保存；下次分析将使用新标定。")
            return
        self._set_running(True)
        self.var_status.set("按各图独立比例尺更新尺寸、形貌和输出…")
        def worker():
            try:
                for payload, scale in jobs:
                    paint, stats, stats_path = export_detection_outputs(
                        payload["path"],payload["img"],payload["columns"],payload.get("structure_profile"),payload["tag"],
                        detection_meta=payload.get("detection"),scale=scale,output_dir=os.path.dirname(payload["stats_path"]))
                    updated = dict(payload,stats=stats,stats_path=stats_path,scale=scale,paint=paint)
                    self.q.put(("rescaled",updated))
                self.q.put(("scale_done",None))
            except Exception:
                self.q.put(("scale_done",traceback.format_exc()))
        threading.Thread(target=worker,daemon=True).start()

    # -----------------------------------------------------------------
    # 运行
    # -----------------------------------------------------------------

    def _make_cb(self, out_tag):
        def cb(orig, paint, name, payload):
            self.q.put(("img", out_tag, orig, paint, name, payload))
        return cb

    def _start_jobs(self, jobs, label):
        if self.running:
            return

        try:
            self._collect_params()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        if not jobs:
            return

        for j in jobs:
            if not os.path.isfile(j["path"]):
                messagebox.showwarning("提示", f"文件不存在：{j['path']}")
                return
            j["scale"] = read_image_scale(j["path"])

        self._set_running(True)
        self.last_dir = core.INPUT_DIR

        self.var_status.set(label)
        self.pbar.configure(maximum=len(jobs), value=0)
        self.lbl_pct.configure(text="0%")

        threading.Thread(
            target=self._worker,
            args=(jobs,),
            daemon=True
        ).start()

    def _start_single(self):
        name = self.var_sel.get()
        if not name:
            messagebox.showwarning("提示", "请先选择图像")
            return

        p = os.path.join(self._folder(), name)
        self._start_jobs(
            [{"path": p, "mode": "full", "roi": None}],
            f"准备分析：{name}"
        )

    def _start_batch(self):
        paths = self._list_tifs()
        if not paths:
            messagebox.showwarning("提示", "没有 tif/tiff 图像")
            return

        jobs = [
            {"path": p, "mode": "full", "roi": None}
            for p in paths
        ]

        self._start_jobs(
            jobs,
            f"准备批量分析：{len(paths)} 张"
        )

    def _start_roi(self):
        if self._roi is None or self._orig_bgr is None:
            messagebox.showwarning("提示", "请先框选 ROI")
            return

        if not self._last_img_name:
            return

        p = self._last_image_path
        if not os.path.isfile(p):
            messagebox.showwarning("提示", f"文件不存在：{p}")
            return

        self._start_jobs(
            [{
                "path": p,
                "mode": "roi",
                "roi": self._roi
            }],
            f"ROI 识别：{self._last_img_name}"
        )

    def _worker(self, jobs):
        total = len(jobs)
        ok = 0
        fails = []

        with open(os.devnull, "w", encoding="utf-8") as sink, \
                contextlib.redirect_stdout(sink), \
                contextlib.redirect_stderr(sink):

            for i, job in enumerate(jobs, 1):
                name = os.path.basename(job["path"])
                self.q.put(("prog", i, total, name))

                try:
                    if job["mode"] == "roi":
                        core.process_image_roi(
                            job["path"],
                            job["roi"],
                            on_image=self._make_cb("_user"), scale=job["scale"]
                        )
                    else:
                        core.process_image(
                            job["path"],
                            on_image=self._make_cb("_tool"), scale=job["scale"]
                        )

                    ok += 1
                except Exception:
                    tb = traceback.format_exc().strip().splitlines()
                    why = tb[-1] if tb else "未知错误"
                    fails.append(f"{name}: {why}")

        self.q.put(("done", ok, total, fails))

    def _set_running(self, on):
        self.running = bool(on)
        state = "disabled" if on else "normal"

        for widget in (
            self.btn_run1,
            self.btn_run_all,
            self.btn_open,
            self.btn_roi_run,
            self.btn_roi_clear,
            self.btn_image_scale,
            self.entry_dir,
            *self.scale_widgets,
        ):
            widget.configure(state=state)

        self.combo.configure(
            state="disabled" if on else "readonly"
        )
        self.lst_files.state(["disabled"] if on else ["!disabled"])

    # -----------------------------------------------------------------
    # 预览
    # -----------------------------------------------------------------

    def _bgr_to_photo(self, bgr):
        """
        直接让 OpenCV 按 BGR 编码 PNG。
        不再先 BGR->RGB 再交给 cv2.imencode，避免红蓝交换。
        """
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return None
        return tk.PhotoImage(data=buf.tobytes())

    def _cache_payload(self, payload):
        path = os.path.abspath(payload["path"])
        self._file_cache.setdefault(image_key(path), {})[payload["tag"]] = payload
        sample = comparison_sample(payload)
        key = (sample["source"], sample["mode"])
        previous = self._comparison_samples.get(key)
        if previous:
            sample["name"] = previous["name"]
        self._comparison_samples[key] = sample
        if path not in self._analyzed:
            self._analyzed.append(path)
        self._refresh_analyzed_list()
        win = self._baseline_window
        if win is not None and win.winfo_exists():
            import copy
            for old in win.samples:
                if (old["source"], old["mode"]) == key:
                    name = old["name"]
                    old.update(copy.deepcopy(sample), name=name)
                    win._refresh_samples()
                    break

    def _set_preview(self, tag, orig, paint, name, payload):
        if self._file_cache.get(image_key(payload["path"]), {}).get(tag) is not payload:
            payload = dict(payload, paint=paint)
            self._cache_payload(payload)
        path = os.path.abspath(payload["path"])
        changed = self._last_image_path is None or image_key(self._last_image_path) != image_key(path)

        if changed:
            self._roi = None
            self._m1 = None
            self._m2 = None
            self._measure_drag = None
            self._measure_creating = False
            self._measure_moved = False
            self._measure_preview_point = None
            self._m2_preview = None
            self._ang = []
            self._tool_bgr = self._user_bgr = None
            self._analysis_cache = {"tool": None, "user": None}
            self._last_stat_src = {"tool": None, "user": None}
            self._n_columns = 0
            for block in self._stat_blocks.values():
                block["tree"].delete(*block["tree"].get_children())
                block["var"].set("（尚无结果）")
            self._line_profile_data = None
            self._profile_meta = {"mode": "line"}
            self._profile_pick = None
            self._profile_span = []
            self._profile_drag = None

        self._last_img_name = name
        self._last_image_path = path
        self._display_scale = read_image_scale(path)
        self.var_dir.set(os.path.dirname(path))
        self.var_sel.set(os.path.basename(path))
        self.combo.configure(values=[os.path.basename(p) for p in self._list_tifs()])
        self._set_scale_fields(self._display_scale)
        self.frm_orig.configure(text="原始图像 · " + name)
        self._orig_bgr = orig

        if tag == "_tool":
            self._tool_bgr = paint
            self._analysis_cache["tool"] = payload
            # 记录本次综合图的柱数，供 _tool 未到达时预估拼图尺寸
            try:
                self._n_columns = len(payload.get("columns", []) or [])
            except Exception:
                self._n_columns = 0
        else:
            self._user_bgr = paint
            self._analysis_cache["user"] = payload

        self._redraw_preview()

        kind = "tool" if tag == "_tool" else "user"
        self._populate_stats_tree(kind, payload["stats"])

        self._last_stat_src[kind] = payload.get("stats_path")

        q = payload.get("quality", {})
        mode = payload.get("threshold_mode", "")
        detection_mode = payload.get("detection_mode", "unknown")
        detection_variant = payload.get("detection_variant", "unknown")
        detection_confidence = payload.get("detection_confidence", -1.0)
        detection_attempts = payload.get("detection_attempts", 0)
        detection_warning = str(payload.get("detection_warning", "") or "").strip()
        cd_range = payload.get("cd_range_px") or {}
        try:
            cd_text = (
                f"CD范围={float(cd_range.get('min')):.1f}~"
                f"{float(cd_range.get('max')):.1f}px"
            )
        except Exception:
            cd_text = "CD范围=自动"
        n = len(payload.get("columns", []))

        self.var_status.set(
            f"{name}: {n} 根柱 | 识别={detection_mode} | 阈值策略={mode} | "
            f"候选={detection_variant} | 置信度={float(detection_confidence):.2f} "
            f"| 尝试={int(detection_attempts)} | "
            f"{cd_text} | "
            f"对比度={q.get('contrast', 0):.1f} | "
            f"边缘={q.get('edge_strength', 0):.1f}"
            + (f" | 提示={detection_warning}" if detection_warning else "")
        )

    def _redraw_preview(self):
        self._resize_job = None
        self.cv_orig.delete("all")
        self._photos = []

        if self._orig_bgr is None:
            return

        self.update_idletasks()

        oh, ow = self._orig_bgr.shape[:2]
        self._ow = ow
        self._oh = oh

        cw = max(self.cv_orig.winfo_width() - 4, 60)
        ch = max(self.cv_orig.winfo_height() - 4, 60)

        s = min(2.0, cw / ow, ch / oh)
        s = max(s, 0.02)
        self._scale = s

        draw = self._orig_bgr
        if abs(s - 1.0) > 0.001:
            draw = cv2.resize(
                draw,
                (max(1, int(ow * s)), max(1, int(oh * s))),
                interpolation=cv2.INTER_AREA
            )

        ph = self._bgr_to_photo(draw)
        if ph is not None:
            self._photos.append(ph)
            self._img_x = max(0, (cw - draw.shape[1]) // 2)
            self._img_y = max(0, (ch - draw.shape[0]) // 2)
            self.cv_orig.create_image(self._img_x, self._img_y, anchor="nw", image=ph)

        self._put_previews()
        self._redraw_marks()

        # 原图重绘后同步刷新灰度剖面（行/列模式会随图像尺寸更新）
        if MATPLOTLIB_AVAILABLE and hasattr(self, "var_profile_mode"):
            self._update_profile_series()
            self._draw_line_profile()
            self._draw_profile_overlay()

    def _tool_full_size(self, main_h, main_w):
        """按 render_combined 的拼图方式计算 _tool 综合图的固有尺寸。

        综合图 = 主图(含左轴 AXIS_MARGIN_L/下轴 AXIS_MARGIN_B)
                 + 底部 x 投影条(profile_h=120)
                 + 右侧逐柱 y 面板(每根 PANEL_W=100)。
        只依赖主图尺寸与柱子数量，与消息到达顺序无关，
        用于在 _tool 图尚未到达时预估缩放比，避免预览跳变。
        """
        n = max(int(self._n_columns or 0), 0)
        return main_h + 120, main_w + PANEL_W * n

    def _put_previews(self):
        """tool 与 user 两个预览使用同一缩放比例，保证两边主图一样大、位置固定。

        _tool 综合图在主图右侧拼了逐柱面板（PANEL_W×柱数）、底部拼了
        投影条（120px），_user 图只有主图。缩放比按"_tool 综合图恰好
        放进其面板"计算（_tool 图更大，天然是受限的那张），再套用到
        两张整图上。这样：
        - 两预览的主图显示尺寸一致；
        - 缩放比只取决于面板大小与图像内容，与 _tool/_user 消息的
          到达顺序无关，预览不再忽大忽小；
        - 配合 anchor="nw"，图固定在各自面板左上角。
        """
        pairs = [
            ("tool", self._tool_bgr, self.frm_tool, self.lbl_tool),
            ("user", self._user_bgr, self.frm_user, self.lbl_user),
        ]

        # 确定参与统一缩放的"_tool 侧"整图尺寸：
        # 优先用已到达的 _tool 综合图；未到达时按固有拼图尺寸预估。
        tool_h = tool_w = None
        if self._tool_bgr is not None:
            tool_h, tool_w = self._tool_bgr.shape[:2]
        elif self._orig_bgr is not None:
            oh, ow = self._orig_bgr.shape[:2]
            tool_h, tool_w = self._tool_full_size(oh, ow + AXIS_MARGIN_L)

        if tool_h is None:
            # 两图都未就绪：清空 user 预览后返回
            if self._user_bgr is None:
                self.lbl_user.configure(image="", text="（尚无人工识别结果）")
            return

        # 统一缩放比：_tool 整图恰好放进 frm_tool（扣除 LabelFrame 边框与标题）
        fw_t = max(self.frm_tool.winfo_width() - 8, 80)
        fh_t = max(self.frm_tool.winfo_height() - 42, 60)
        common = max(0.02, min(2.0, fw_t / tool_w, fh_t / tool_h))

        for kind, bgr, frame, label in pairs:
            if bgr is None:
                label.configure(image="", text=(
                    "（尚无自动识别结果）" if kind == "tool"
                    else "（尚无人工识别结果）"
                ))
                continue

            h, w = bgr.shape[:2]
            s = common
            # 兜底：若统一缩放后仍超出自身面板（面板被压得很小），
            # 再按自身 fit 收缩，防止溢出
            fw = max(frame.winfo_width() - 8, 80)
            fh = max(frame.winfo_height() - 42, 60)
            s = min(s, max(0.02, min(2.0, fw / w, fh / h)))

            draw = bgr
            if abs(s - 1.0) > 0.001:
                draw = cv2.resize(
                    draw,
                    (max(1, int(w * s)), max(1, int(h * s))),
                    interpolation=cv2.INTER_AREA
                )

            ph = self._bgr_to_photo(draw)
            if ph is not None:
                self._photos.append(ph)
                label.configure(image=ph, text="")

    def _resize_preview_squares(self):
        """将两个预览区设为正方形，边长取可用宽度的 1/2 减去间隔。"""
        # 获取父容器的可用宽度
        parent = self.frm_tool.master
        row_width = parent.winfo_width()
        if row_width <= 0:
            # 如果父容器还未布局好，尝试用需求宽度
            row_width = parent.winfo_reqwidth()
        if row_width <= 0:
            return
        # 每个预览区的可用宽度约为 (总宽 - 间隔) / 2，再减去一点边距
        margin = 6  # 稍微减小边距，使预览区更大
        total_gap = 2  # 两个预览区之间的间隔
        square_size = (row_width - total_gap) // 2 - margin
        square_size = max(100, min(square_size, 600, max(100, parent.winfo_height() - 10)))  # 限制在 250-600 之间（增大）

        # 设置两个 LabelFrame 为正方形
        self.frm_tool.configure(width=square_size, height=square_size)
        self.frm_user.configure(width=square_size, height=square_size)

    def _on_mid_resize(self, _evt=None):
        # 先更新正方形尺寸
        self._resize_preview_squares()
        # 再重绘预览
        if self._orig_bgr is not None and self._resize_job is None:
            self._resize_job = self.after(120, self._redraw_preview)

    # -----------------------------------------------------------------
    # Canvas
    # -----------------------------------------------------------------

    def _bind_canvas(self):
        self.cv_orig.bind("<ButtonPress-1>", self._cv_press)
        self.cv_orig.bind("<B1-Motion>", self._cv_move)
        self.cv_orig.bind("<ButtonRelease-1>", self._cv_release)
        self.cv_orig.bind("<Motion>", self._cv_hover)

    def _tool(self):
        return self.var_canvas_tool.get()

    def _cv_to_orig(self, ex, ey):
        ox = int((ex - self._img_x) / max(self._scale, 1e-12))
        oy = int((ey - self._img_y) / max(self._scale, 1e-12))

        ox = min(max(ox, 0), max(0, self._ow - 1))
        oy = min(max(oy, 0), max(0, self._oh - 1))

        return ox, oy

    def _orig_to_cv(self, x, y):
        return (
            x * self._scale + self._img_x,
            y * self._scale + self._img_y
        )

    def _canvas_point_inside_image(self, ex, ey):
        """判断画布事件是否落在当前原图，而不是右侧/底部留白。"""
        if self._orig_bgr is None or self._ow <= 0 or self._oh <= 0:
            return False
        x0, y0 = float(self._img_x), float(self._img_y)
        x1 = x0 + float(self._ow) * float(self._scale)
        y1 = y0 + float(self._oh) * float(self._scale)
        return x0 <= float(ex) < x1 and y0 <= float(ey) < y1

    def _on_tool_change(self):
        self._drag = None
        self._measure_drag = None
        self._measure_creating = False
        self._measure_moved = False
        self._measure_preview_point = None
        self._m2_preview = None

        t = self._tool()

        if t == "roi":
            self._m1 = None
            self._m2 = None
            self._ang = []
        elif t == "dist":
            self._roi = None
            self._ang = []
        else:
            self._roi = None
            self._m1 = None
            self._m2 = None

        self._profile_span = []
        self._profile_pick = None
        self._profile_drag = None
        if (MATPLOTLIB_AVAILABLE and hasattr(self, "var_profile_mode")
                and self.var_profile_mode.get() == "直线剖面"):
            self._line_profile_data = None
            self._profile_meta = {"mode": "line"}
        self._redraw_marks()
        if MATPLOTLIB_AVAILABLE and hasattr(self, "var_profile_mode"):
            self._draw_line_profile()
            self._draw_profile_overlay()

    def _clear_canvas_marks(self):
        self._roi = None
        self._m1 = None
        self._m2 = None
        self._ang = []
        self._drag = None
        self._measure_drag = None
        self._measure_creating = False
        self._measure_moved = False
        self._measure_preview_point = None
        self._m2_preview = None
        self._line_profile_selected_x = None
        self._profile_span = []
        self._profile_pick = None
        self._profile_drag = None
        if (MATPLOTLIB_AVAILABLE and hasattr(self, "var_profile_mode")
                and self.var_profile_mode.get() == "直线剖面"):
            self._line_profile_data = None
            self._profile_meta = {"mode": "line"}
        self._redraw_marks()
        if MATPLOTLIB_AVAILABLE and hasattr(self, "var_profile_mode"):
            self._draw_line_profile()
            self._draw_profile_overlay()

    def _snap_point(self, base, target):
        bx, by = base
        tx, ty = target

        dx = tx - bx
        dy = ty - by

        if dx == 0 and dy == 0:
            return target, None

        angle = math.atan2(abs(dy), abs(dx))
        tol = math.radians(5.0)

        if angle < tol:
            return (tx, by), "H"

        if angle > math.pi / 2 - tol:
            return (bx, ty), "V"

        return target, None

    def _nearest_measure_endpoint(self, ex, ey):
        """返回鼠标附近的量测端点索引（0=A，1=B），否则返回 None。"""
        if self._m1 is None or self._m2 is None:
            return None

        points = (self._m1, self._m2)
        # 端点以原图坐标保存，命中半径用屏幕像素计算，避免缩放后难以拖动。
        radius = max(10.0, min(24.0, 9.0 + 5.0 * float(self._scale)))
        best = None
        best_d = radius * radius
        for i, (x, y) in enumerate(points):
            cx, cy = self._orig_to_cv(x, y)
            d = (float(ex) - cx) ** 2 + (float(ey) - cy) ** 2
            if d <= best_d:
                best = i
                best_d = d
        return best

    def _cv_press(self, e):
        if self.running or self._orig_bgr is None:
            return

        if not self._canvas_point_inside_image(e.x, e.y):
            return

        t = self._tool()

        if t == "roi":
            self._drag = (e.x, e.y)
            return

        p = self._cv_to_orig(e.x, e.y)

        if t == "dist":
            # 已有 A/B 时，优先命中端点并进入拖拽；点在其它位置才开始新量测。
            hit = self._nearest_measure_endpoint(e.x, e.y)
            if hit is not None:
                self._measure_drag = hit
                self._measure_creating = False
                self._measure_moved = False
                self._m2_preview = None
                return

            self._measure_drag = None
            if self._m1 is None:
                self._m1 = p
                self._m2 = None
                self._measure_creating = True
                self._measure_moved = False
                self._measure_preview_point = p
            elif self._m2 is None:
                self._m2, _ = self._snap_point(self._m1, p)
                self._measure_creating = False
                self._measure_preview_point = None
            else:
                self._m1 = p
                self._m2 = None
                self._measure_creating = True
                self._measure_moved = False
                self._measure_preview_point = p

            # 直线量测：A/B 一确定就立刻刷新灰度剖面（实时）
            self._refresh_line_profile_live()
            self._redraw_marks()
            return

        if t == "angle":
            if len(self._ang) >= 3:
                self._ang = [p]
            else:
                if self._ang:
                    p, _ = self._snap_point(self._ang[-1], p)
                self._ang.append(p)

            self._redraw_marks()

    def _cv_move(self, e):
        if self.running or self._orig_bgr is None:
            return

        t = self._tool()

        if t == "roi" and self._drag is not None:
            self._redraw_marks()

            x0, y0 = self._drag
            self.cv_orig.create_rectangle(
                x0, y0, e.x, e.y,
                outline="#ff3333",
                width=2,
                dash=(5, 3),
                tags="ovl"
            )
            return

        # 直线量测：A 已定、B 未定时，鼠标移动即实时刷新灰度剖面
        if t == "dist" and self._m1 is not None and self._m2 is None:
            p, _ = self._snap_point(self._m1, self._cv_to_orig(e.x, e.y))
            self._measure_moved = self._measure_creating
            self._measure_preview_point = p
            self._draw_dist_preview(p)
            return

        # 已确定 A/B 时拖动任一端点，图片、距离和灰度曲线同步更新。
        if t == "dist" and self._measure_drag is not None:
            p = self._cv_to_orig(e.x, e.y)
            if self._measure_drag == 0 and self._m2 is not None:
                p, _ = self._snap_point(self._m2, p)
                self._m1 = p
            elif self._measure_drag == 1 and self._m1 is not None:
                p, _ = self._snap_point(self._m1, p)
                self._m2 = p
            self._m2_preview = None
            self._refresh_profile_after_image_measure_change()
            self._redraw_marks()

    def _cv_hover(self, e):
        """单击 A 后无需按住鼠标，也持续预览 A→光标的灰度曲线。"""
        if (self.running or self._orig_bgr is None or self._tool() != "dist"
                or self._measure_creating or self._measure_drag is not None
                or self._m1 is None or self._m2 is not None):
            return
        if not self._canvas_point_inside_image(e.x, e.y):
            return
        p, _ = self._snap_point(self._m1, self._cv_to_orig(e.x, e.y))
        self._measure_preview_point = p
        self._draw_dist_preview(p)

    def _cv_release(self, e):
        if self._measure_creating:
            if self._measure_moved and self._m1 is not None:
                p, _ = self._snap_point(self._m1, self._cv_to_orig(e.x, e.y))
                if p != self._m1:
                    self._m2 = p
            self._measure_creating = False
            self._measure_moved = False
            self._measure_preview_point = None
            self._m2_preview = None
            self._refresh_profile_after_image_measure_change(activate_line=True)
            self._redraw_marks()
            return

        if self._measure_drag is not None:
            # Button release 也取一次最终坐标，避免松开时漏掉最后一个 motion 事件。
            if self._orig_bgr is not None and self._m1 is not None and self._m2 is not None:
                p = self._cv_to_orig(e.x, e.y)
                if self._measure_drag == 0:
                    p, _ = self._snap_point(self._m2, p)
                    self._m1 = p
                else:
                    p, _ = self._snap_point(self._m1, p)
                    self._m2 = p
                self._refresh_profile_after_image_measure_change()
                self._redraw_marks()
            self._measure_drag = None
            return

        if self._tool() != "roi" or self._drag is None:
            return

        x0, y0 = self._drag
        self._drag = None

        ax, ay = self._cv_to_orig(x0, y0)
        bx, by = self._cv_to_orig(e.x, e.y)

        if abs(bx - ax) < 3 and abs(by - ay) < 3:
            self._redraw_marks()
            return

        self._roi = (
            min(ax, bx),
            min(ay, by),
            max(ax, bx),
            max(ay, by),
        )

        self._redraw_marks()

    def _clear_overlays(self):
        self.cv_orig.delete("ovl")

    def _marker(self, x, y, fill, outline=None, radius=4):
        cx, cy = self._orig_to_cv(x, y)
        r = max(2, int(radius))

        self.cv_orig.create_oval(
            cx - r, cy - r,
            cx + r, cy + r,
            fill=fill,
            outline=outline or fill,
            tags="ovl"
        )

    def _redraw_marks(self):
        if self._orig_bgr is None:
            return

        self._clear_overlays()

        t = self._tool()

        if t == "roi" and self._roi is not None:
            x0, y0, x1, y1 = self._roi
            a = self._orig_to_cv(x0, y0)
            b = self._orig_to_cv(x1, y1)

            self.cv_orig.create_rectangle(
                a[0], a[1], b[0], b[1],
                outline="#ff0000",
                width=2,
                dash=(5, 3),
                tags="ovl"
            )

            self.lbl_roi_info.configure(
                text=f"ROI：({x0},{y0}) → ({x1},{y1})  "
                     f"{x1-x0+1}×{y1-y0+1}px",
                foreground="#cc0000"
            )

        elif t == "dist":
            self._draw_dist()

        elif t == "angle":
            self._draw_angle()

    def _refresh_profile_after_image_measure_change(self, activate_line=False):
        """图片 A/B 变化后刷新直线曲线，并保持 span 的相对位置。"""
        if not MATPLOTLIB_AVAILABLE or not hasattr(self, "var_profile_mode"):
            return

        if self.var_profile_mode.get() != "直线剖面":
            if activate_line:
                self.var_profile_mode.set("直线剖面")
            return

        old_n = len(self._line_profile_data) if self._line_profile_data is not None else 0
        span_f = [float(i) / max(old_n - 1, 1) for i in self._profile_span[:2]]
        pick_f = (float(self._profile_pick) / max(old_n - 1, 1)
                  if self._profile_pick is not None and old_n else None)

        self._update_profile_series()
        new_n = len(self._line_profile_data) if self._line_profile_data is not None else 0
        if new_n:
            self._profile_span = [
                int(min(max(round(f * (new_n - 1)), 0), new_n - 1)) for f in span_f
            ]
            if pick_f is not None:
                self._profile_pick = int(min(max(round(pick_f * (new_n - 1)), 0), new_n - 1))
        else:
            self._profile_span = []
            self._profile_pick = None
        self._draw_line_profile()
        self._draw_profile_overlay()

    def _refresh_line_profile_live(self):
        """新建直线量测时立即切到并刷新直线灰度曲线。"""
        self._refresh_profile_after_image_measure_change(activate_line=True)

    def _draw_dist_preview(self, target):
        """A 已定、鼠标移动中：画 A→光标 的临时连线与实时剖面。"""
        self._redraw_marks()

        ax, ay = self._m1
        bx, by = target

        a = self._orig_to_cv(ax, ay)
        b = self._orig_to_cv(bx, by)

        self.cv_orig.create_line(
            a[0], a[1], b[0], b[1],
            fill="#0066ff",
            width=2,
            dash=(4, 3),
            tags="ovl"
        )
        self._marker(bx, by, "#99ddff", "#33aaff")

        px = math.hypot(bx - ax, by - ay)
        nm = self._px_to_nm(px)
        txt = f"L={px:.2f}px"
        if nm is not None:
            txt += f" ≈ {nm:.3f}nm"

        self.lbl_roi_info.configure(
            text=f"预览 A({ax},{ay}) → ({bx},{by})：{txt}（松开/单击 B 定线）",
            foreground="#0077cc"
        )

        # 实时剖面：A → 当前光标。先保存 preview，再切换模式，避免
        # StringVar trace 在模式切换瞬间把曲线清空。
        if MATPLOTLIB_AVAILABLE and hasattr(self, "var_profile_mode"):
            self._m2_preview = (bx, by)
            try:
                if self.var_profile_mode.get() != "直线剖面":
                    self.var_profile_mode.set("直线剖面")
                self._update_profile_series(preview=(bx, by))
                self._draw_line_profile()
                self._draw_profile_overlay()
            finally:
                self._m2_preview = None

    def _draw_dist(self):
        if self._m1 is None:
            self.lbl_roi_info.configure(
                text="直线量测：依次单击 A、B 两点；接近水平/竖直自动吸附。",
                foreground="#555555"
            )
            return

        ax, ay = self._m1
        self._marker(ax, ay, "#33aaff", "#0066ff")

        if self._m2 is None:
            self.lbl_roi_info.configure(
                text="直线量测：A 已确定，请单击 B。",
                foreground="#aa5500"
            )
            return

        bx, by = self._m2

        a = self._orig_to_cv(ax, ay)
        b = self._orig_to_cv(bx, by)

        self.cv_orig.create_line(
            a[0], a[1], b[0], b[1],
            fill="#0066ff",
            width=2,
            tags="ovl"
        )
        self._marker(bx, by, "#33aaff", "#0066ff")

        px = math.hypot(bx - ax, by - ay)
        nm = self._px_to_nm(px)

        txt = f"L={px:.2f}px"
        if nm is not None:
            txt += f" ≈ {nm:.3f}nm"

        self.cv_orig.create_text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 - 10,
            text=txt,
            fill="#0044cc",
            font=("Microsoft YaHei UI", 9, "bold"),
            tags="ovl"
        )

        self.lbl_roi_info.configure(
            text=f"A({ax},{ay}) → B({bx},{by})：{txt}",
            foreground="#0055bb"
        )

        # 曲线刷新由图片端点的 press/motion/release 处理器显式触发；
        # 这里只负责绘制，避免每次重画标记时重复计算整条剖面。

    # -----------------------------------------------------------------
    # 灰度剖面：直线 / 横向行 / 纵向列 + 选点定位 + 曲线量测
    # -----------------------------------------------------------------

    def _on_profile_mode_change(self, *_args):
        self._profile_pick = None
        self._profile_span = []
        self._profile_drag = None
        self._profile_pos_job = None
        self._update_profile_series()
        self._draw_line_profile()
        self._draw_profile_overlay()

    def _on_profile_pos_change(self):
        self._profile_pos_job = None
        self._update_profile_series()
        self._draw_line_profile()
        self._draw_profile_overlay()

    def _on_profile_pos_edit(self, _event=None):
        """Spinbox 键盘输入/失焦时立即更新行列剖面。"""
        if hasattr(self, "_profile_pos_job") and self._profile_pos_job is not None:
            try:
                self.after_cancel(self._profile_pos_job)
            except Exception:
                pass
        self._profile_pos_job = self.after(60, self._on_profile_pos_change)

    def _update_profile_series(self, preview=None):
        """按当前模式重算灰度曲线，并记录 index → 图像坐标 的映射。"""
        if self._orig_bgr is None or not MATPLOTLIB_AVAILABLE:
            return

        mode = self.var_profile_mode.get()

        try:
            pos = int(float(self.var_profile_pos.get()))
        except Exception:
            pos = 0

        h, w = self._orig_bgr.shape[:2]

        if mode == "横向行剖面":
            y = min(max(pos, 0), h - 1)
            data = self._orig_bgr[y, :, 0].astype(np.float32)
            self._profile_meta = {"mode": "row", "pos": y}
            self.spn_profile_pos.configure(to=max(h - 1, 0))
            if str(self.var_profile_pos.get()) != str(y):
                self.var_profile_pos.set(str(y))

        elif mode == "纵向列剖面":
            x = min(max(pos, 0), w - 1)
            data = self._orig_bgr[:, x, 0].astype(np.float32)
            self._profile_meta = {"mode": "col", "pos": x}
            self.spn_profile_pos.configure(to=max(w - 1, 0))
            if str(self.var_profile_pos.get()) != str(x):
                self.var_profile_pos.set(str(x))

        else:
            # 直线剖面：优先用已定的 A/B；鼠标拖动预览时用 preview 端点
            if preview is None:
                preview = getattr(self, "_m2_preview", None)
            a = self._m1
            b = self._m2 if self._m2 is not None else preview

            if a is None or b is None:
                self._line_profile_data = None
                self._profile_meta = {"mode": "line"}
                return

            ax, ay = a
            bx, by = b

            n = max(2, int(round(math.hypot(bx - ax, by - ay))) + 1)
            xs = np.clip(np.rint(np.linspace(ax, bx, n)).astype(int), 0, w - 1)
            ys = np.clip(np.rint(np.linspace(ay, by, n)).astype(int), 0, h - 1)

            data = self._orig_bgr[ys, xs, 0].astype(np.float32)
            self._profile_meta = {"mode": "line", "a": (ax, ay), "b": (bx, by)}

        self._line_profile_data = data

    def _profile_index_to_xy(self, idx):
        """曲线上的第 idx 个点 → 原始图像坐标 (x, y)。"""
        meta = self._profile_meta or {}
        data = self._line_profile_data

        if data is None or len(data) == 0:
            return None

        idx = int(min(max(int(idx), 0), len(data) - 1))
        mode = meta.get("mode")

        if mode == "row":
            return float(idx), float(meta.get("pos", 0))

        if mode == "col":
            return float(meta.get("pos", 0)), float(idx)

        if mode == "line":
            a, b = meta.get("a"), meta.get("b")
            if not a or not b:
                return None
            t = idx / max(len(data) - 1, 1)
            return (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
            )

        return None

    def _profile_index_distance(self, i0, i1):
        """将曲线索引差换算为实际图像像素距离。

        行/列剖面每个样本相隔一个像素；直线剖面为了避免重复取整，
        采样索引按端点欧氏距离均匀铺开。
        """
        i0, i1 = int(i0), int(i1)
        if self._profile_meta.get("mode") == "line":
            a, b = self._profile_meta.get("a"), self._profile_meta.get("b")
            data = self._line_profile_data
            if a and b and data is not None and len(data) > 1:
                return abs(i1 - i0) * math.hypot(b[0] - a[0], b[1] - a[1]) / (len(data) - 1)
        return float(abs(i1 - i0))

    def _profile_title(self):
        meta = self._profile_meta or {}
        mode = meta.get("mode")

        if mode == "row":
            return f"横向行剖面 y={meta.get('pos', 0)}"

        if mode == "col":
            return f"纵向列剖面 x={meta.get('pos', 0)}"

        if mode == "line":
            a, b = meta.get("a"), meta.get("b")
            if a and b:
                return f"直线剖面 ({a[0]},{a[1]})→({b[0]},{b[1]})"

        return "灰度剖面"

    def _draw_line_profile(self):
        if not MATPLOTLIB_AVAILABLE:
            return

        ax = self._line_profile_ax
        ax.clear()

        # 轻量的卡片式绘图区：白底、低对比网格和统一蓝色曲线，
        # 在深色/浅色 Windows 主题下都保持清晰。
        ax.set_facecolor("#ffffff")
        ax.tick_params(axis="both", colors="#526274", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            spine.set_linewidth(0.8)

        data = self._line_profile_data

        if data is None or len(data) < 2:
            ax.set_title(self._profile_title(), fontsize=10)
            ax.set_xlabel("采样索引 index", fontsize=9)
            ax.set_ylabel("Gray", fontsize=9)
            ax.grid(True, alpha=0.25, color="#9aaabd", linewidth=0.6)
            self._line_profile_fig.tight_layout()
            self._line_profile_canvas.draw_idle()
            self._update_profile_info()
            return

        x = np.arange(len(data))
        ax.fill_between(x, data, np.min(data), color="#2f74c0", alpha=0.08)
        ax.plot(
            x,
            data,
            linewidth=1.35,
            color="#1f66ad",
            solid_capstyle="round",
            antialiased=True,
        )
        ax.set_xlabel("采样索引 index", fontsize=9)
        ax.set_ylabel("Gray", fontsize=9)
        ax.grid(True, alpha=0.25, color="#9aaabd", linewidth=0.6)

        title = self._profile_title()

        if self._profile_pick is not None:
            i = int(min(max(self._profile_pick, 0), len(data) - 1))
            ax.axvline(i, color="#00aa55", linestyle="--", linewidth=1)
            ax.plot([i], [data[i]], "o", color="#00aa55")
            ax.annotate(
                f"{float(data[i]):.0f}",
                (i, float(data[i])),
                fontsize=8, ha="center", va="bottom"
            )

        if len(self._profile_span) == 2:
            i0, i1 = sorted(int(v) for v in self._profile_span)
            i0 = int(min(max(i0, 0), len(data) - 1))
            i1 = int(min(max(i1, 0), len(data) - 1))

            ax.axvspan(i0, i1, color="#ff9900", alpha=0.18)
            ax.plot([i0, i1], [data[i0], data[i1]], "o", color="#ff7700")

            dpx = self._profile_index_distance(i0, i1)
            nm = self._px_to_nm(dpx)
            dgray = float(data[i1]) - float(data[i0])

            extra = f"  Δ={dpx:.3f}px"
            if nm is not None:
                extra += f" ≈ {nm:.2f}nm"
            extra += f"  Δgray={dgray:+.0f}"
            ax.set_title(title + extra, fontsize=10)
        else:
            ax.set_title(title, fontsize=10)

        self._line_profile_fig.tight_layout()
        self._line_profile_canvas.draw_idle()
        self._update_profile_info()

    def _update_profile_info(self):
        if not hasattr(self, "lbl_profile_info"):
            return

        data = self._line_profile_data

        if data is None or len(data) < 2:
            self.lbl_profile_info.configure(
                text=(
                    "曲线交互：切换横向行/纵向列查看灰度；"
                    "单击定位，Shift+单击/右键选两点，拖动端点同步图片测量。"
                )
            )
            return

        if self._profile_meta.get("mode") == "line" and self._profile_meta.get("a") and self._profile_meta.get("b"):
            total_px = math.hypot(self._profile_meta["b"][0] - self._profile_meta["a"][0],
                                  self._profile_meta["b"][1] - self._profile_meta["a"][1])
        else:
            total_px = max(0, len(data) - 1)
        parts = [self._profile_title(), f"长度 {total_px:.2f}px（采样点 {len(data)}）"]

        if self._profile_pick is not None:
            i = int(min(max(self._profile_pick, 0), len(data) - 1))
            xy = self._profile_index_to_xy(i)
            if xy:
                parts.append(
                    f"选点 idx={i} → 图({xy[0]:.0f},{xy[1]:.0f}) "
                    f"灰度={float(data[i]):.0f}"
                )

        if len(self._profile_span) == 2:
            i0, i1 = sorted(int(v) for v in self._profile_span)
            i0 = int(min(max(i0, 0), len(data) - 1))
            i1 = int(min(max(i1, 0), len(data) - 1))
            dpx = self._profile_index_distance(i0, i1)
            nm = self._px_to_nm(dpx)
            p0 = self._profile_index_to_xy(i0)
            p1 = self._profile_index_to_xy(i1)

            txt = f"量取 Δ={dpx:.3f}px"
            if nm is not None:
                txt += f" ≈ {nm:.3f}nm"
            if p0 and p1:
                txt += f"  图({p0[0]:.0f},{p0[1]:.0f})→({p1[0]:.0f},{p1[1]:.0f})"
            txt += f"  Δgray={float(data[i1]) - float(data[i0]):+.0f}"
            parts.append(txt)

        self.lbl_profile_info.configure(text="  |  ".join(parts))

    def _profile_event_index(self, event):
        """把 Matplotlib 事件转换为合法的曲线索引。"""
        data = self._line_profile_data
        if (
            data is None
            or len(data) == 0
            or event is None
            or event.inaxes != self._line_profile_ax
            or event.xdata is None
        ):
            return None
        return int(min(max(int(round(event.xdata)), 0), len(data) - 1))

    def _profile_marker_hit(self, event):
        """查找鼠标附近的曲线标记，返回 pick/span0/span1。"""
        data = self._line_profile_data
        if data is None or len(data) == 0 or event.x is None or event.y is None:
            return None

        candidates = []
        if self._profile_pick is not None:
            candidates.append(("pick", int(self._profile_pick)))
        for j, idx in enumerate(self._profile_span[:2]):
            candidates.append((f"span{j}", int(idx)))

        best = None
        best_d = 16.0 * 16.0
        for name, idx in candidates:
            idx = min(max(idx, 0), len(data) - 1)
            try:
                px, py = self._line_profile_ax.transData.transform(
                    (idx, float(data[idx]))
                )
            except Exception:
                continue
            d = (float(event.x) - px) ** 2 + (float(event.y) - py) ** 2
            if d <= best_d:
                best = name
                best_d = d
        return best

    def _sync_profile_span_to_canvas(self):
        """将曲线量测端点映射到原图覆盖层（不改写采样线 A/B）。"""
        if not self._profile_span:
            return

        # 横向/纵向剖面上的单点也可以直接定位图片；直线剖面需要两点
        # 才能建立一条新的测量线，否则会破坏正在显示的原曲线。
        if self._profile_meta.get("mode") == "line" and len(self._profile_span) < 2:
            return

        points = [self._profile_index_to_xy(i) for i in self._profile_span[:2]]
        points = [p for p in points if p is not None]
        if not points:
            return

        # span 是曲线上的独立量测端点；不要改写直线剖面的采样 A/B，
        # 否则每次 Shift/拖动都会裁短原曲线。调用方统一重绘，避免
        # 拖动过程中一次 motion 触发两次 Matplotlib draw。

    def _on_profile_click(self, event):
        if not MATPLOTLIB_AVAILABLE:
            return

        idx = self._profile_event_index(event)
        if idx is None:
            return

        # 先命中已有端点，随后 motion 事件即可拖动它。
        hit = self._profile_marker_hit(event)
        if event.button == 1 and hit is not None:
            self._profile_drag = hit
            return

        # Shift+单击 / 右键 = 量取线段端点；普通单击 = 选点并在原图定位
        if getattr(event, "key", None) == "shift" or event.button == 3:
            if len(self._profile_span) < 2:
                self._profile_span.append(idx)
            else:
                self._profile_span = [idx]
            self._sync_profile_span_to_canvas()
        else:
            self._profile_pick = idx

        self._draw_line_profile()
        self._draw_profile_overlay()

    def _on_profile_motion(self, event):
        """拖动曲线端点时实时刷新曲线和图片标记。"""
        if not MATPLOTLIB_AVAILABLE or self._profile_drag is None:
            return
        idx = self._profile_event_index(event)
        if idx is None:
            return

        kind = self._profile_drag
        if kind == "pick":
            self._profile_pick = idx
        elif kind == "span0":
            if len(self._profile_span) < 1:
                self._profile_span = [idx]
            else:
                self._profile_span[0] = idx
            self._sync_profile_span_to_canvas()
        elif kind == "span1":
            if len(self._profile_span) < 2:
                self._profile_span = (self._profile_span[:1] + [idx])
            else:
                self._profile_span[1] = idx
            self._sync_profile_span_to_canvas()

        self._draw_line_profile()
        self._draw_profile_overlay()

    def _on_profile_release(self, _event):
        if self._profile_drag is not None:
            self._profile_drag = None
            self._draw_line_profile()
            self._draw_profile_overlay()

    def _clear_profile_measure(self):
        self._profile_span = []
        self._profile_pick = None
        self._profile_drag = None
        # 曲线端点与图片 A/B 是同一份量测状态；清除曲线时同步移除
        # 原图连线，避免用户看到已经失效的旧距离。
        self._m1 = None
        self._m2 = None
        self._measure_drag = None
        self._measure_creating = False
        self._measure_moved = False
        self._measure_preview_point = None
        self._m2_preview = None
        if hasattr(self, "var_profile_mode") and self.var_profile_mode.get() == "直线剖面":
            self._update_profile_series()
        self._draw_line_profile()
        self._draw_profile_overlay()
        self._redraw_marks()

    def _profile_marker(self, x, y, color, text):
        cx, cy = self._orig_to_cv(x, y)

        self.cv_orig.create_oval(
            cx - 4, cy - 4, cx + 4, cy + 4,
            outline=color, width=2, tags="prof"
        )
        self.cv_orig.create_line(cx - 9, cy, cx + 9, cy, fill=color, tags="prof")
        self.cv_orig.create_line(cx, cy - 9, cx, cy + 9, fill=color, tags="prof")
        self.cv_orig.create_text(
            cx + 11, cy - 11,
            text=text, fill=color,
            font=("Microsoft YaHei UI", 8, "bold"),
            tags="prof"
        )

    def _draw_profile_overlay(self):
        """在原图上画出：剖面位置线、选点、量取线段（tag=prof）。"""
        if not hasattr(self, "cv_orig"):
            return

        self.cv_orig.delete("prof")

        if self._orig_bgr is None:
            return

        meta = self._profile_meta or {}
        mode = meta.get("mode")

        if mode == "row":
            y = meta.get("pos", 0)
            a = self._orig_to_cv(0, y)
            b = self._orig_to_cv(max(self._ow - 1, 0), y)
            self.cv_orig.create_line(
                a[0], a[1], b[0], b[1],
                fill="#00aa88", width=1, dash=(6, 3), tags="prof"
            )
        elif mode == "col":
            x = meta.get("pos", 0)
            a = self._orig_to_cv(x, 0)
            b = self._orig_to_cv(x, max(self._oh - 1, 0))
            self.cv_orig.create_line(
                a[0], a[1], b[0], b[1],
                fill="#00aa88", width=1, dash=(6, 3), tags="prof"
            )

        if self._profile_pick is not None:
            xy = self._profile_index_to_xy(self._profile_pick)
            if xy:
                self._profile_marker(xy[0], xy[1], "#00aa55", "P")

        for i, idx in enumerate(self._profile_span[:2]):
            xy = self._profile_index_to_xy(idx)
            if xy:
                self._profile_marker(
                    xy[0], xy[1], "#ff7700",
                    "M1" if i == 0 else "M2"
                )

        if len(self._profile_span) == 2:
            p0 = self._profile_index_to_xy(self._profile_span[0])
            p1 = self._profile_index_to_xy(self._profile_span[1])

            if p0 and p1:
                a = self._orig_to_cv(p0[0], p0[1])
                b = self._orig_to_cv(p1[0], p1[1])

                self.cv_orig.create_line(
                    a[0], a[1], b[0], b[1],
                    fill="#ff7700", width=2, tags="prof"
                )

                dpx = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                nm = self._px_to_nm(dpx)
                txt = f"{dpx:.1f}px"
                if nm is not None:
                    txt += f" ≈ {nm:.2f}nm"

                self.cv_orig.create_text(
                    (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 10,
                    text=txt, fill="#cc5500",
                    font=("Microsoft YaHei UI", 9, "bold"),
                    tags="prof"
                )

    def _angle_result(self):
        if len(self._ang) < 3:
            return None

        (ax, ay), (vx, vy), (cx, cy) = self._ang[:3]

        ux, uy = ax - vx, ay - vy
        wx, wy = cx - vx, cy - vy

        lu = math.hypot(ux, uy)
        lw = math.hypot(wx, wy)

        if lu <= 0 or lw <= 0:
            return None

        c = (ux * wx + uy * wy) / (lu * lw)
        c = max(-1.0, min(1.0, c))

        return math.degrees(math.acos(c)), lu, lw

    def _draw_angle(self):
        if not self._ang:
            self.lbl_roi_info.configure(
                text="角度量测：A 端点 → 顶点 V → C 端点。",
                foreground="#555555"
            )
            return

        a = self._ang[0]
        self._marker(*a, "#ffcc66", "#cc6600")

        if len(self._ang) >= 2:
            v = self._ang[1]

            ac = self._orig_to_cv(*a)
            vc = self._orig_to_cv(*v)

            self.cv_orig.create_line(
                ac[0], ac[1], vc[0], vc[1],
                fill="#cc6600",
                width=2,
                tags="ovl"
            )
            self._marker(*v, "#ff6666", "#cc0000", radius=5)

        if len(self._ang) >= 3:
            c = self._ang[2]
            vc = self._orig_to_cv(*self._ang[1])
            cc = self._orig_to_cv(*c)

            self.cv_orig.create_line(
                vc[0], vc[1], cc[0], cc[1],
                fill="#0088cc",
                width=2,
                tags="ovl"
            )
            self._marker(*c, "#66ccff", "#0066cc")

            result = self._angle_result()
            if result:
                deg, l1, l2 = result

                nm1 = self._px_to_nm(l1)
                nm2 = self._px_to_nm(l2)

                text = f"∠={deg:.2f}°  {l1:.1f}/{l2:.1f}px"
                if nm1 is not None and nm2 is not None:
                    text += f"  {nm1:.2f}/{nm2:.2f}nm"

                self.lbl_roi_info.configure(
                    text=text,
                    foreground="#cc0000"
                )

                self.cv_orig.create_text(
                    vc[0] + 32,
                    vc[1] - 16,
                    text=f"{deg:.1f}°",
                    fill="#cc0000",
                    font=("Microsoft YaHei UI", 9, "bold"),
                    tags="ovl"
                )

    def _px_to_nm(self, px):
        try:
            if not self._last_image_path:
                return None
            scale = self._display_scale
            return px * scale["nm"] / scale["pixels"]
        except Exception:
            return None

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    @staticmethod
    def _fmt_stat(desc, value):
        if value is None:
            return "-"

        try:
            v = float(value)
            if not math.isfinite(v):
                return "-"
        except Exception:
            return "-"

        if "角" in desc:
            return f"{v:.2f}"

        if "比值" in desc:
            return f"{v:.4f}"

        return f"{v:.3f}"

    def _populate_stats_tree(self, kind, data):
        blk = self._stat_blocks[kind]
        tree = blk["tree"]
        var = blk["var"]

        tree.delete(*tree.get_children())

        if not data or not data.get("cols"):
            var.set("没有识别到 PR 柱")
            return

        cols = data["cols"]
        descs = data["desc"]
        rows = data["rows"]
        means = data["means"]

        # 列顺序：全部柱平均 放最前，再 PR1..PRn
        col_ids = ["avg"] + [f"p{i}" for i in range(len(cols))]
        tree.configure(columns=col_ids)

        tree.heading("avg", text="全部柱平均")
        tree.column("avg", width=90, minwidth=80, anchor="e", stretch=False)

        for cid, label in zip(col_ids[1:], cols):
            tree.heading(cid, text=label)
            tree.column(cid, width=68, minwidth=60, anchor="e", stretch=False)

        tree.heading("#0", text="计算项")

        for i, desc in enumerate(descs):
            vals = [self._fmt_stat(desc, means[i])]
            vals += [self._fmt_stat(desc, v) for v in rows[i]]

            tree.insert(
                "",
                "end",
                text=desc,
                values=vals
            )

        var.set(
            f"{len(cols)} 根柱 · 亚像素边缘测量"
            + (f" · 已排除两端 {data.get('detection', {}).get('excluded_count', 0)} 根" if data.get("detection", {}).get("exclude_outermost") else "")
            + ("\n" + data.get("detection_warning", "") if data.get("detection_warning") else "")
        )

        morph = data["morphology"]
        tree.tag_configure("profile", foreground="#2463aa")
        tree.insert("", "end", text="形貌（可同时存在）",
                    values=[morphology_labels(morph["metrics"])] + [morphology_labels(c["metrics"]) for c in morph["per_column"]], tags=("profile",))

    # -----------------------------------------------------------------
    # 保存
    # -----------------------------------------------------------------

    def _save_preview_img(self, kind):
        bgr = self._tool_bgr if kind == "tool" else self._user_bgr

        if bgr is None or not self._last_img_name:
            messagebox.showwarning("提示", "没有可保存的预览")
            return

        stem = os.path.splitext(self._last_img_name)[0]
        tag = "tool" if kind == "tool" else "user"

        p = filedialog.asksaveasfilename(
            title="保存预览",
            initialfile=f"{stem}_{tag}_preview.png",
            defaultextension=".png",
            filetypes=[("PNG 图像", "*.png")]
        )

        if p:
            imwrite_unicode(p, bgr)
            self.var_status.set(f"已保存：{p}")

    def _save_stats_txt(self, kind):
        src = self._last_stat_src.get(kind)

        if not src or not os.path.isfile(src):
            messagebox.showwarning("提示", "没有可保存的统计表")
            return

        p = filedialog.asksaveasfilename(
            title="保存统计表",
            initialfile=os.path.basename(src),
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt")]
        )

        if p:
            shutil.copyfile(src, p)
            self.var_status.set(f"已保存：{p}")

    # -----------------------------------------------------------------
    # Help
    # -----------------------------------------------------------------

    def _show_pipeline(self):
        self._show_doc("识别流程", PIPELINE_HELP)

    def _show_params_help(self):
        lines = ["■ 左侧“识别参数”（改动后会自动防抖重识别）\n"]
        for attr, name, *_ in MAIN_PARAMS:
            lines.append(
                f"■ {name} [{attr}]\n"
                f"  默认：{getattr(core, attr)}\n"
                f"  {PARAM_DESC.get(attr, '')}\n"
            )

        lines.append("■ 柱体明暗\n亮柱（推荐）排除相反极性候选；暗柱需手动指定。\n"
                     "自动模式会比较两种极性，并明确提示人工复核。\n"
                     "候选分数用于排序，不是百分比准确率。\n"
                     "保存识别过程时会输出每一行和每一列的分页曲线及 points.csv。\n")

        lines.append("每图比例尺：在已分析文件列表双击图片设置，标定按图保存；修改后只重算该图的 nm 统计。\n")
        lines.append("方波或 PR 样本比较统一使用 Baseline 对比入口，首页只显示实际测量。\n")
        self._show_doc("参数说明", "\n".join(lines))

    def _show_about(self):
        self._show_doc("关于", ABOUT_HELP)

    def _show_doc(self, title, text):
        if self._help_window is not None and self._help_window.winfo_exists():
            self._help_window.select_topic(title, text)
        else:
            self._help_window = HelpWindow(self, title, text)

    # -----------------------------------------------------------------
    # Queue
    # -----------------------------------------------------------------

    def _resume_pending_work(self):
        if self.running:
            return
        if self._scale_dirty or self._pending_scale_edits or self._pending_scale_paths:
            self._scale_dirty = False
            try:
                self._flush_scale_edits()
            except Exception as exc:
                self.var_status.set("标定未保存：" + str(exc))
            if self.running:
                return
        if self._live_dirty:
            self._live_dirty = False
            self.after(80, self._run_live)

    def _drain_queue(self):
        try:
            while True:
                item = self.q.get_nowait()

                if not isinstance(item, tuple):
                    continue

                kind = item[0]

                if kind == "prog":
                    _, i, total, name = item
                    self.var_status.set(
                        f"正在识别：{name}  第 {i}/{total} 张"
                    )
                    self.pbar.configure(maximum=total, value=i)
                    self.lbl_pct.configure(
                        text=f"{int(i * 100 / max(total, 1))}%"
                    )

                elif kind == "img":
                    _, tag, orig, paint, name, payload = item
                    self._set_preview(
                        tag, orig, paint, name, payload
                    )

                elif kind == "rescaled":
                    payload = item[1]
                    self._cache_payload(payload)
                    if self._last_image_path and image_key(payload["path"]) == image_key(self._last_image_path):
                        self._set_preview(payload["tag"], payload["img"], payload["paint"],
                                          os.path.basename(payload["path"]), payload)

                elif kind == "done":
                    _, ok, total, fails = item
                    self._set_running(False)

                    self.pbar.configure(maximum=total, value=ok)
                    self.lbl_pct.configure(
                        text=f"{int(ok * 100 / max(total, 1))}%"
                    )

                    if fails:
                        self.var_status.set(
                            f"完成：成功 {ok}/{total}；"
                            + " | ".join(fails[:3])
                        )
                    else:
                        self.var_status.set(
                            f"处理完成：{ok}/{total} 张成功"
                        )

                    self._resume_pending_work()

                elif kind == "scale_done":
                    _, err = item
                    self._set_running(False)

                    if err:
                        self.var_status.set(
                            "比例尺重算失败：" + err.strip().splitlines()[-1]
                        )
                    else:
                        self.var_status.set(
                            "比例尺输出与统计已更新；未重新执行图像识别"
                        )

                    self._resume_pending_work()

        except queue.Empty:
            pass

        self._queue_job = self.after(80, self._drain_queue)

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

    def _open_baseline(self):
        if self._baseline_window is not None and self._baseline_window.winfo_exists():
            self._baseline_window.lift()
            return
        self._baseline_window = BaselineWindow(self)

    def _open_process(self):
        payloads = [p for p in self._analysis_cache.values() if p and p.get("process_dir")]
        if not payloads:
            messagebox.showinfo("识别过程", "请先分析一张图像；自动识别固定保存到 test，ROI 需勾选保存过程图。")
            return
        path = max((p["process_dir"] for p in payloads), key=os.path.getmtime)
        os.startfile(path)

    def _open_output(self):
        d = self.last_dir or core.INPUT_DIR

        if not os.path.isdir(d):
            messagebox.showwarning("提示", "输出目录不存在")
            return

        try:
            os.startfile(d)
            return
        except Exception:
            pass

        try:
            import subprocess
            if sys.platform == "darwin":
                subprocess.Popen(["open", d])
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            messagebox.showwarning(
                "提示",
                f"无法自动打开：{e}\n{d}"
            )

    def destroy(self):
        # 注销本窗口所有 after 回调，避免关闭后继续访问已销毁的 Tcl 控件。
        for job in self.tk.call("after", "info"):
            self.tk.call("after", "cancel", job)
        super().destroy()

    def _on_close(self):
        if not self.running:
            try:
                self._flush_scale_edits(refresh=False)
            except Exception as exc:
                messagebox.showerror("标定未保存", str(exc), parent=self)
                return
        if self.running:
            if not messagebox.askyesno(
                "确认",
                "正在处理，仍要退出吗？"
            ):
                return

        self.destroy()


# =====================================================================
# 自检 / 入口
# =====================================================================

def run_selftest():
    # 优先验证随工具提供的示例图；仍兼容旧环境中的 pr2.tif。
    if getattr(sys, "frozen", False):
        out_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    probe = os.environ.get("PR_PROBE_IMG")
    if not probe:
        probe = next(
            (
                name for name in ("pr3.tif", "pr2.tif")
                if os.path.isfile(name) or os.path.isfile(os.path.join(out_dir, name))
            ),
            "pr3.tif",
        )
    if not os.path.isabs(probe) and not os.path.isfile(probe):
        local_probe = os.path.join(out_dir, probe)
        if os.path.isfile(local_probe):
            probe = local_probe

    lines = [
        f"numpy={np.__version__}",
        f"opencv={cv2.__version__}",
        f"cwd={os.getcwd()}",
        f"probe={probe}",
    ]

    success = True
    try:
        if not os.path.isfile(probe):
            raise FileNotFoundError("自检需要原图，请设置 PR_PROBE_IMG")
        else:
            payload = process_image(probe)
            lines.append(
                f"process_image ok: n_col={len(payload['columns'])}, "
                f"detected={payload.get('detected_count')}, excluded={payload.get('excluded_count')}, "
                f"detection={payload.get('detection_mode', 'unknown')}, "
                f"variant={payload.get('detection_variant', 'unknown')}, "
                f"confidence={float(payload.get('detection_confidence', -1.0)):.3f}, "
                f"attempts={int(payload.get('detection_attempts', 0))}, "
                f"threshold={payload['threshold_mode']}, "
                f"cd_range={payload.get('cd_range_px')}"
            )
            process_dir = payload["process_dir"]
            if os.path.basename(process_dir) != "test":
                raise AssertionError("过程图未保存到 test 文件夹")
            with open(payload["process_manifest"], encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("status") != "complete":
                raise AssertionError("过程图导出未完成")
            for name, filename in manifest["files"].items():
                path = os.path.join(process_dir, filename)
                if not os.path.isfile(path):
                    raise AssertionError(f"缺少过程文件：{filename}")
                if name.endswith(".png") and imread_unicode(path) is None:
                    raise AssertionError(f"过程图无法读取：{filename}")
            required_steps = range(1, 11 if payload["columns"] else 9)
            for step in required_steps:
                if not any(name.startswith(f"step{step}_") for name in manifest["files"]):
                    raise AssertionError(f"缺少 step{step} 过程图")
            lines.append(f"process_images ok: folder={process_dir}, files={len(manifest['files'])}")
            lines.append(f"manifest={payload['process_manifest']}")
            if not MATPLOTLIB_AVAILABLE:
                raise AssertionError("缺少实时灰度曲线 / baseline 绘图依赖 Matplotlib")
            if payload["stats"]["morphology"]["included_count"]:
                sample = sample_morphology(comparison_sample(payload))
                baseline = square_baseline(sample["metrics"]["cd_mean_nm"], sample["metrics"]["height_nm"])
                report = compare_morphologies(baseline, [sample])
                paths = export_baseline_report(report, os.path.join(out_dir, "selftest_baseline"))
                if not os.path.isfile(paths["plot"]):
                    raise AssertionError("未生成 baseline 对比图")
                lines.append(f"baseline ok: metrics={len(report['rows'])}, complete_pillars={sample['included_count']}")
            probe_roi = os.environ.get("PR_PROBE_ROI")
            if probe_roi:
                roi = tuple(int(v) for v in probe_roi.split(","))
                if len(roi) != 4:
                    raise ValueError("PR_PROBE_ROI 应为 x0,y0,x1,y1")
                roi_payload = process_image_roi(probe, roi)
                expected = os.environ.get("PR_EXPECT_ROI_COLUMNS")
                if expected and len(roi_payload["columns"]) != int(expected):
                    raise AssertionError("ROI 柱数量不符合预期")
                lines.append(f"roi ok: roi={roi}, columns={len(roi_payload['columns'])}, process={roi_payload['process_dir']}")
    except Exception as e:
        success = False
        lines.append(f"FAILED: {e}")
        lines.append(traceback.format_exc())

    out = os.path.join(out_dir, "selftest_ok.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return success


def gui_main():
    if "--selftest" in sys.argv[1:]:
        raise SystemExit(0 if run_selftest() else 1)

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    App().mainloop()


if __name__ == "__main__":
    gui_main()
