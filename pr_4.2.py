# -*- coding: utf-8 -*-
"""
PR 柱子智能识别与测量工具 —— 单文件 GUI 版
============================================================

【推荐运行】
    python pr4_copy.py

【PyInstaller 单文件打包】
    PyInstaller --onefile --windowed pr4_copy.py
    （或双击 build_exe.bat，自动找 Python、装依赖、打包成 dist\\PR_Analyzer.exe）

------------------------------------------------------------
一、图像识别流程（本版重点：先看“逐像素横向灰度斜率”，再看亮度）
------------------------------------------------------------

0) 为什么要先看灰度斜率
   · 不同图（甚至同一批图）的整体亮暗、对比度差异很大，
     单靠一个固定二值阈值很难通用，图一换就要重新调。
   · 但有一件事是通用的：PR 柱的左右侧壁一定对应
     “沿 x 方向的灰度突变”。所以本版把“横向灰度斜率”放在最前面，
     用它来 (a) 选二值阈值、(b) 定位柱区、(c) 评价每行候选边界。

1) 读取与灰度归一化
   · 支持 8/16-bit、灰度/BGR/BGRA TIFF。
   · 非 8-bit 图像按 1%~99% 分位拉伸到 8-bit，降低极端亮点/暗点影响。
   · 使用 imdecode/fromfile 兼容 Windows 中文路径。

2) 对比度增强与降噪
   · CLAHE 做局部对比度增强。
   · 中值滤波去椒盐噪声。
   · 高斯滤波稳定灰度斜率与阈值分割。

3) “先看灰度变化斜率”
   · Sobel-X 得到 |dI/dx|：PR 左右侧壁通常产生明显横向灰度突变。
   · Sobel-Y 得到 |dI/dy|：辅助理解顶部/底部边缘。
   · 梯度按 99 分位归一化到 0~1，避免少数异常强边缘支配结果。

4) 自适应亮度分割 + “斜率选阈值”（本版核心）
   · 候选一：固定阈值（THRESHOLD ± THRESHOLD_TOL）
   · 候选二：Otsu
   · 候选三：局部自适应阈值（ADAPTIVE_BLOCK_SIZE / ADAPTIVE_C）
   · 按图像质量（边缘强度 / 对比度 / 直方图展宽）先做一次规则选择；
   · 再做“斜率选阈值”：在合理灰度区间扫描候选阈值 t，
     二值化后取横向边界像素，计算这些位置上的 Sobel-X 平均强度
     （edge_agreement_score）。边界越落在真正的灰度突变处，分值越高。
     只有当最优候选显著优于规则选择（GRADIENT_THRESHOLD_GAIN）时才替换，
     保证“智能化”不会越改越差。
   · 结果：图与图之间二值不一样时，也能自动落在比较合理的阈值上。
   · 会自动尝试亮/暗极性、增强、平滑、均衡化和局部对比度候选，
     对有限候选全部评分后选择全局最优结果，避免一次阈值就提前结束。

5) 形态学修整
   · 小核开运算去孤立噪声。
   · 内部“闭运算核”真正控制闭运算，不再被其它自动核偷偷覆盖；
     普通用户无需在左侧额外调节。

6) 全局柱体高度区间
   · 用整图 y 方向亮度投影寻找柱体主体平台，尽量避开底部基座/标尺。

7) 智能横向柱区定位
   · 亮度投影：柱体内部提供“实体证据”。
   · Sobel-X 梯度投影：柱子侧壁提供“边界证据”。
   · 两者归一化后按 GRADIENT_WEIGHT 融合为结构投影：
       structure = (1-w)*brightness + w*gradient
   · 对结构投影平滑后找柱间谷值，因此不只依赖二值图的绝对亮暗。

8) 单柱逐行侧壁追踪
   · 每一行仍从二值图找连续亮段。
   · 选择候选段时综合：
       a. 与上一行的重叠
       b. 中心位置连续性
       c. 宽度连续性
       d. 候选左右边界处的灰度斜率强度
   · 即：真实侧壁梯度越强，候选段越容易被保留。
   · 不再逐行调用无效 Canny，速度和逻辑都更合理。

9) 几何过滤
   · 最小柱高、最小平均宽、最小高宽比。
   · 整图模式可使用“柱顶应在上半区”的先验；ROI 模式关闭该限制，
     避免用户紧框后误删正常柱。

10) 测量与统计
   · 高度、上/中/下三段 CD、CD 比值、左右侧壁角、左右 LER 3σ。
   · 统计直接从原始浮点轮廓 × nm/px 计算，不再经过
     “写 nm txt → 四舍五入 → 重新读回”的量化链路。
   · 输出同时记录识别模式、候选版本、置信度（候选排序分数，非百分比）、
     尝试次数、用户先验范围/候选保护范围，以及完整主体柱、全部柱和边缘
     残柱的观测 CD，便于复核每次结果。
   · 用户可见的 Line CD 先验、观测范围和统计表 CD 统一按
     x_end-x_start（左右边界距离）计算；内部像素格数只用于几何过滤。
   · *_boundaries_nm.txt 仅作为导出结果，统计精度不依赖其显示小数位。

11) 纵向逐像素扫描 → W(y) 曲线（线宽随高度的变化）
   · 每根柱逐行追踪左右边界，得到 W(y)（= CD(y)）；
   · 所有柱的 W(y) 还会重采样到统一高度网格后求“平均 W(y)”，
     用于平均值与 Target 的对比。

12) W(y) 形貌归类（所有柱 + 平均值，与 Target 对比后归类）
    Profile            W(y) 特征
    -------------------------------------------------------
    Ideal / Square     基本恒定（理想方波）
    Footing            靠近底部突然变宽
    Undercut           靠近底部突然变窄
    Positive taper     从顶到底逐渐变宽
    Negative taper     从顶到底逐渐变窄
    Bowing             中间比上下更宽
    Necking            中间局部变窄
    T-top / Top flare  顶部局部变宽
    Asymmetric         左右边缘变化不对称

   判定标记：✅ 正常（≤ 0.5×容差） / ⚠️ 临界（0.5~1×容差） / ❌ 超差
    · 逐柱与平均值都给出 “PASS / FOOTING / UNDERCUT / T-TOP / TAPER+ /
      TAPER- / BOWING / NECKING / ASYM” 一行标记；
    · 统计表（xx_*_stats_nm.txt）最后两行就是：
        Profile  → 每根柱与平均值的归类（多个用 “+” 连接）
        Judge    → Profile/Target (%)；100=等宽，>100=Profile 偏宽，<100=Profile 偏窄
    · target_check.txt 里另有完整的“对象 × 类别”判定矩阵。

13) 智能后处理（纵向扫描带来的额外收益）
   · footing 台阶修剪：底边回到柱体主体，不被基座拉长
     （只剪“台阶式突增”，不会误剪渐变的上窄下宽）；
   · 粘连检测：相邻柱在共同高度上的间隙过小时提示“疑似粘连”；
   · 近柱合并：相邻两柱过近、或“结构投影在两柱之间没有明显下沉”
     时自动合并，纠正“把一根真实柱错拆成两根”的过切情况；
     调参建议：若仍过切，把 “近柱合并(px)” 调大（如 8~12），
     或把 “谷值比例” 调大（如 0.35~0.45）。

14) 标准 Target（理想方波）对比
   · 人工输入 Target CD / 容差(nm 与 %) / 目标高度；
   · 理想方波 = 上中下三段 CD 都等于 Target；
   · 输出 PASS/FAIL、Δvs Target、Profile/Target (%) Judge、方波相似度、
     片内均匀性 3σ、形貌分布，并自动保存 xx_*_target_check.txt。
     Judge=100 表示等宽，>100 表示 Profile 比 Target 宽，<100 表示偏窄。

15) GUI
   · 自动整图识别：_tool
   · 框选 ROI 识别：_user
   · 灰度剖面三种模式：直线 / 横向行 / 纵向列
       单击曲线 = 在原图定位该点；Shift+单击两点 = 量取线段（px 与 nm）
   · 直线量测 + 三点角度量测（水平/竖直自动吸附）
   · 参数变化后防抖自动重识别
   · 分析运行期间继续改参数，会在本轮结束后自动补跑最后一次，避免
     “GUI 参数已变、预览还是旧参数”的状态不一致。
   · 比例尺 / Target / 形貌参数变化只重算输出与判定，不重新做图像检测
     （有缓存时）。
   · 帮助菜单：识别流程 / 参数说明 / 统计公式 / 形貌判定 /
     Target 对比 / 灰度曲线与量测 / 输出文件 / 关于。

------------------------------------------------------------
二、重要说明
------------------------------------------------------------

· 线程环境变量只能限制 BLAS/OMP 线程数，不能修复 NumPy 二进制本身
  要求 X86_V2 而 CPU 不支持的问题。若遇到 X86_V2 baseline 报错，
  需要安装/打包面向目标旧 CPU 的 NumPy 构建。

· OpenCV 使用 BGR 颜色顺序。本文件中的 PR_COLORS 已按 BGR 正确填写。

· 本文件没有外部 .py 依赖。
"""

import os
import sys

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
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Optional

import numpy as np
import cv2

# scipy 为可选增强依赖：提供更先进的峰值查找（find_peaks）与
# 中值滤波（median_filter），缺失时自动回退到 OpenCV/numpy 实现。
try:
    from scipy.signal import find_peaks as _sp_find_peaks
    from scipy.ndimage import median_filter as _sp_median_filter
    SCIPY_AVAILABLE = True
except Exception:
    _sp_find_peaks = None
    _sp_median_filter = None
    SCIPY_AVAILABLE = False

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

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_GLOBS = ("*.tif", "*.tiff")
OUTPUT_SUBDIR = True
DEBUG_OUTPUT = True
# 识别过程测试图（中间阶段可视化）输出开关，GUI 调试模式同步打开
DEBUG_TEST_OUTPUT = True

# ---- 亮度分割 ----
THRESHOLD = 110
THRESHOLD_TOL = 0
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 2

# ---- 梯度 / 结构识别 ----
GRADIENT_WEIGHT = 0.35       # 结构投影中梯度证据权重 0~0.8
GRADIENT_PERCENTILE = 75.0   # 强梯度参考分位，仅用于调试/辅助
SMOOTH_KERNEL = 21
VALLEY_RATIO = 0.25
PROFILE_ACTIVE_RATIO = 0.10

# ---- 几何优先的自适应柱体识别 ----
# 旧版以全图二值图为主，遇到柱内高亮纹理时容易把一根柱切成数根窄条。
# 新路径先从“逐行归一化亮度的横向中位投影”定位实体，再用左右有符号
# 梯度逐行追踪边界；只有几何路径无法建立有效结构时才回退到旧二值流程。
GEOMETRIC_DETECTION_ENABLE = True
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
AUTO_RETRY_ENABLE = True
AUTO_RETRY_MAX = 5
AUTO_RETRY_MIN_CONFIDENCE = 4.8

# ---- 形态学 / 边界 ----
MORPH_CLOSE_K = 3
EDGE_MARGIN = 20

# ---- 柱子有效性 ----
MIN_COLUMN_HEIGHT = 20
MIN_AVG_WIDTH = 15
MIN_ASPECT_RATIO = 1.2
BOTTOM_OFFSET = 4
WINDOW_MARGIN = 3
GLOBAL_Y_TOLERANCE = 5
MAX_TRACE_GAP = 8

# ---- 追踪评分 ----
TRACE_CENTER_PENALTY = 0.20
TRACE_WIDTH_PENALTY = 0.10
TRACE_EDGE_REWARD = 0.45

# ---- 比例尺 ----
SCALE_PIXELS = 100.0
SCALE_NM = 50.0

# ---- 形貌（纵向逐像素扫描 → CD(y) 曲线）判定 ----
SHAPE_ENABLE = True
SHAPE_TOP_RATIO = 1.12        # 顶部相对中段展宽 ≥ 该比例 → T-top（蘑菇头）
SHAPE_FOOTING_RATIO = 1.12    # 底部相对中段展宽 ≥ 该比例 → Footing（底部脚宽）
SHAPE_WAIST_RATIO = 0.92      # 中段相对上下平均收缩 ≤ 该比例 → Undercut（腰部内凹）
SHAPE_TAPER_RATIO = 1.06      # 上→下单调收/放 ≥ 该比例 → Taper（梯形）
SHAPE_MIN_DELTA_NM = 1.0      # 绝对变化小于该值不判异常，避免噪声误判（nm）
SHAPE_SEG_FRAC = 0.15         # 顶部/底部展宽判定取上/下各 15% 高度
SHAPE_IRREGULAR_3SIG = 0.25   # CD 3σ / CD 均值 超过该值 → Irregular（形状异常）

# ---- 智能后处理（纵向逐像素扫描的额外作用）----
FOOTING_TRIM_ENABLE = True    # 底部因 footing/基座突然展宽时，把柱体底边上移到展宽起点
FOOTING_TRIM_RATIO = 1.6      # 底部宽度 ≥ 柱身中位宽 × 该倍数 → 判为 footing 台阶
FOOTING_TRIM_MIN_FRAC = 0.30  # 修剪后至少保留原高度的该比例，避免过度修剪
BRIDGE_GAP_PX = 2             # 相邻柱最小水平间隙 ≤ 该值 → 提示粘连/未分开
MERGE_GAP_PX = 4              # 相邻柱水平间隙 < 该值 → 自动合并成一根（防错拆）
MERGE_Y_OVERLAP_MIN = 4       # 合并要求 y 方向至少有该像素行重叠
MERGE_VALLEY_RATIO = 0.10     # 浅谷合并：相邻柱之间的亮度投影最小值 ≥ 该比例 × 两侧峰值平均 → 合并
MERGE_VALLEY_GAP_MAX = 60     # 浅谷合并仅在相邻柱水平间隙 ≤ 该值时考虑
MERGE_HEIGHT_RATIO_MAX = 1.5  # 合并还要求两柱高度比 ≤ 该值（避免错合上下错位的两根柱）
MERGE_BOTTOM_DIFF_MAX = 15    # 合并要求两柱底边 y 差 ≤ 该像素（避免错合高度不同的两根柱）
MERGE_DEEP_CRACK_RATIO = 0.05 # 深裂合并：谷深 ≤ 该比例 × 峰值平均，且其它条件满足 → 视为同一根柱被中缝错拆

# ---- W(y) 形貌归类（Profile）----
PROFILE_ASYM_RATIO = 0.45   # 左右边缘变化量差异占比 ≥ 该值 → Asymmetric
PROFILE_ASYM_MIN_NM = 2.0   # 左右边缘变化总量小于该值不判不对称（nm）
W_PROFILE_SAMPLES = 100     # 求“平均 W(y) 曲线”时的重采样点数

# ---- 灰度斜率驱动的二值阈值（图与图差异大时更稳）----
GRADIENT_THRESHOLD_ENABLE = True
GRADIENT_THRESHOLD_GAIN = 1.10   # 斜率候选需显著优于规则选择才切换
GRADIENT_THRESHOLD_STEPS = 24    # 候选灰度级数量

# ---- 标准 Target（理想方波）对比 ----
TARGET_ENABLE = False
TARGET_CD_NM = 60.0           # 目标线宽 CD（nm）
TARGET_TOL_NM = 3.0           # 绝对容差（nm）
TARGET_TOL_PCT = 5.0          # 相对容差（%，与绝对容差取大者）
TARGET_HEIGHT_NM = 0.0        # 目标高度（nm）；0 表示不判断高度

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


def detect_white_border_margin(binary, base_margin=0, max_margin=200):
    """自动检测图像四边连续的“近全白”边框宽度。

    很多 SEM 原图四周带白色边框/标尺，固定 EDGE_MARGIN 裁不干净时，
    会把白框误识别成柱子。这里按边逐行/列统计白色占比，连续 >90% 的
    行/列视为边框，返回应裁剪的宽度（与 base_margin 取大者）。
    """
    h, w = binary.shape[:2]
    b = binary > 0

    def side_width(line_fracs):
        run = 0
        for f in line_fracs:
            if float(f) > 0.90:
                run += 1
                if run >= max_margin:
                    break
            else:
                break
        return run

    top = side_width(b[:max_margin, :].mean(axis=1))
    bot = side_width(b[-max_margin:, :].mean(axis=1)[::-1])
    left = side_width(b[:, :max_margin].mean(axis=0))
    right = side_width(b[:, -max_margin:].mean(axis=0)[::-1])

    return max(int(base_margin), top, bot, left, right)


def edge_agreement_score(binary, grad_x):
    """二值图的横向边界与“横向灰度斜率”的吻合程度。

    图与图之间的绝对亮暗差异很大，但“柱壁处一定有横向灰度突变”这件事
    是通用的。因此用“边界像素上的 Sobel-X 平均强度”评价一个二值化好不好：
    边界越落在真正的灰度突变处，分值越高。
    """
    b = binary > 0
    if not b.any():
        return 0.0

    edges = np.zeros_like(b, dtype=bool)
    diff = b[:, 1:] != b[:, :-1]
    edges[:, 1:] |= diff
    edges[:, :-1] |= diff

    if not edges.any():
        return 0.0

    return float(np.mean(grad_x[edges]))


def estimate_threshold_by_gradient(blur, grad_x, steps=None):
    """逐像素横向灰度斜率驱动的阈值搜索。

    在合理灰度范围内扫描候选阈值，取“边界与横向灰度突变最吻合”的那个。
    返回 (best_threshold, best_score)；失败返回 (None, 0.0)。
    """
    steps = int(steps or GRADIENT_THRESHOLD_STEPS)

    lo_p, hi_p = np.percentile(blur, (20, 95))
    lo, hi = float(lo_p), float(hi_p)

    if hi - lo < 5:
        lo, hi = float(blur.min()), float(blur.max())

    if hi - lo < 3:
        return None, 0.0

    best_t, best_s = None, -1.0

    for t in np.linspace(lo + 1, hi - 1, max(4, steps)):
        cand = (blur > t).astype(np.uint8) * 255
        frac = _binary_fraction(cand)

        # 前景占比必须落在合理区间，避免全黑/全白拿到虚高分
        if frac < 0.005 or frac > 0.85:
            continue

        s = edge_agreement_score(cand, grad_x)
        if s > best_s:
            best_s, best_t = s, float(t)

    if best_t is None:
        return None, 0.0

    return best_t, best_s


# =====================================================================
# 预处理：亮度 + 灰度斜率
# =====================================================================

def preprocess_image(image_path):
    """
    返回:
      img_bgr : 原始灰度 8-bit BGR（显示/绘图）
      binary  : 自适应亮度分割后的二值图
      aux     : gray/enhanced/blur/grad_x/grad_y/quality/threshold_mode...
    """
    img_raw = imread_unicode(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise ValueError(f"无法读取图片：{image_path}")

    gray = to_gray8(img_raw)
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 三级滤波链：中值去椒盐 → 双边滤波保边去噪 → 高斯稳定梯度。
    # 双边滤波参数保持温和（sigmaColor=20），只抹平弱噪声、
    # 不改变柱壁弱边缘的相对强度（过强会把低衬度图的
    # binary/梯度证据洗掉，导致识别回退到错误路径）。
    denoise = cv2.medianBlur(enhanced, 3)
    try:
        denoise = cv2.bilateralFilter(denoise, 5, 20, 10)
    except Exception:
        pass
    blur = cv2.GaussianBlur(denoise, (5, 5), 0)

    # 灰度变化斜率
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_x = normalize_01(np.abs(gx))
    grad_y = normalize_01(np.abs(gy))

    quality = calculate_image_quality(blur)

    lower = int(np.clip(THRESHOLD - THRESHOLD_TOL, 0, 255))
    _, fixed = cv2.threshold(blur, lower, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    block = ensure_odd(ADAPTIVE_BLOCK_SIZE, 3, 151)
    adaptive = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        float(ADAPTIVE_C)
    )

    fixed_frac = _binary_fraction(fixed)
    otsu_frac = _binary_fraction(otsu)

    edge_strength = quality["edge_strength"]
    contrast = quality["contrast"]

    # 选择策略：
    # 高质量图 -> 固定阈值可解释性更好，但异常占比时回退 Otsu
    # 中质量图 -> Otsu
    # 低质量图 -> Otsu 为主体，自适应阈值与固定阈值共同补充
    if edge_strength >= 45 and contrast >= 65 and 0.01 <= fixed_frac <= 0.90:
        binary = fixed
        threshold_mode = f"fixed({lower})"
    elif edge_strength < 25 or contrast < 30:
        supplement = cv2.bitwise_and(adaptive, fixed)
        binary = cv2.bitwise_or(otsu, supplement)
        threshold_mode = "otsu + adaptive∩fixed"
    else:
        binary = otsu
        threshold_mode = "otsu"

    # 若仍出现极端全黑/全白，切换候选
    frac = _binary_fraction(binary)
    if frac < 0.003 or frac > 0.95:
        if 0.005 <= otsu_frac <= 0.95:
            binary = otsu
            threshold_mode = "otsu(fallback)"
        else:
            binary = adaptive
            threshold_mode = "adaptive(fallback)"

    # ------------------------------------------------------------------
    # 灰度斜率驱动的二值阈值：
    # 图与图之间的绝对亮暗差异很大，固定阈值很难通用；但“柱壁处必然出现
    # 横向灰度突变”是通用规律。这里扫描候选阈值，取边界与 Sobel-X 最吻合者，
    # 只有当它显著优于上面的规则选择时才替换，保证不会越改越差。
    # ------------------------------------------------------------------
    grad_t = None
    if GRADIENT_THRESHOLD_ENABLE:
        try:
            grad_t, grad_score = estimate_threshold_by_gradient(blur, grad_x)
        except Exception:
            grad_t, grad_score = None, 0.0

        if grad_t is not None:
            grad_bin = (blur > grad_t).astype(np.uint8) * 255
            rule_score = edge_agreement_score(binary, grad_x)

            if grad_score > max(rule_score * GRADIENT_THRESHOLD_GAIN, 1e-9):
                binary = grad_bin
                threshold_mode = f"gradient({grad_t:.0f})"

    # 开运算固定用小核，避免用户的大 close 核把细柱直接吃掉
    open_kernel = np.ones((3, 3), np.uint8)
    close_k = ensure_odd(MORPH_CLOSE_K, 1, 101)
    close_kernel = np.ones((close_k, close_k), np.uint8)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    if close_k > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # 边缘留白：自动检测四边白色边框，连同固定 EDGE_MARGIN 一起裁掉
    h, w = binary.shape[:2]
    m = detect_white_border_margin(binary, base_margin=EDGE_MARGIN)
    m = int(max(0, min(m, min(h, w) // 4)))
    if m > 0:
        binary[:m, :] = 0
        binary[-m:, :] = 0
        binary[:, :m] = 0
        binary[:, -m:] = 0

    aux = {
        "gray": gray,
        "enhanced": enhanced,
        "blur": blur,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "quality": quality,
        "threshold_mode": threshold_mode,
        "binary_fraction": _binary_fraction(binary),
        "_image_path": image_path,
    }
    return img_bgr, binary, aux


# =====================================================================
# 投影 / 柱区划分
# =====================================================================

def find_global_y_extent(binary):
    """
    用 y 方向亮度投影寻找柱体主体平台。
    返回 (g_top, g_bottom)，失败返回 (0,0)。
    """
    prof = np.sum(binary > 0, axis=1).astype(np.float32)
    prof = smooth_1d(prof, SMOOTH_KERNEL)
    maxv = float(prof.max()) if prof.size else 0.0
    if maxv <= 0:
        return 0, 0

    body = prof > maxv * 0.08
    segments = []
    s = None
    for y, b in enumerate(body):
        if b and s is None:
            s = y
        elif not b and s is not None:
            segments.append((s, y - 1))
            s = None
    if s is not None:
        segments.append((s, len(body) - 1))
    if not segments:
        return 0, 0

    main = max(segments, key=lambda p: p[1] - p[0] + 1)
    slice_prof = prof[main[0]:main[1] + 1]
    med = float(np.median(slice_prof))
    if med <= 0:
        return main

    platform = (prof >= 0.45 * med) & (prof <= 1.65 * med)

    best = None
    s = None
    for y, b in enumerate(platform):
        if b and s is None:
            s = y
        elif not b and s is not None:
            cand = (s, y - 1)
            if best is None or cand[1] - cand[0] > best[1] - best[0]:
                best = cand
            s = None
    if s is not None:
        cand = (s, len(platform) - 1)
        if best is None or cand[1] - cand[0] > best[1] - best[0]:
            best = cand

    return best if best is not None else main


def column_structure_profile(binary, grad_x, h, y_lo=None, y_hi=None):
    """
    横向结构投影:
        brightness evidence + x-gradient evidence

    亮度保证柱体内部不被当成“谷”；
    梯度增强不同图亮度变化时的侧壁识别稳定性。
    """
    if y_lo is None or y_hi is None or y_hi <= y_lo:
        y_lo = int(h * 0.10)
        y_hi = int(h * 0.80)

    y_lo = max(0, int(y_lo))
    y_hi = min(h - 1, int(y_hi))

    b = np.mean(binary[y_lo:y_hi + 1, :] > 0, axis=0).astype(np.float32)

    # 梯度证据分两部分：
    #   1) 连续梯度能量：保留一般侧壁信息；
    #   2) 强梯度支持：由 GUI 的 GRADIENT_PERCENTILE 控制，只强调最明显的灰度突变。
    gx_region = np.asarray(
        grad_x[y_lo:y_hi + 1, :],
        dtype=np.float32
    )

    nonzero = gx_region[gx_region > 1e-6]
    if nonzero.size:
        gp = float(np.clip(GRADIENT_PERCENTILE, 50.0, 99.9))
        strong_thr = float(np.percentile(nonzero, gp))
    else:
        strong_thr = 1.0

    denom = max(1e-6, 1.0 - strong_thr)
    strong_support = np.clip(
        (gx_region - strong_thr) / denom,
        0.0,
        1.0
    )

    g_energy = np.mean(gx_region, axis=0).astype(np.float32)
    g_strong = np.mean(strong_support, axis=0).astype(np.float32)

    # 强边缘占主要梯度证据，一般梯度用于避免过度稀疏。
    g = 0.40 * g_energy + 0.60 * g_strong

    b = normalize_01(smooth_1d(b, SMOOTH_KERNEL), percentile=100.0)
    g = normalize_01(smooth_1d(g, SMOOTH_KERNEL), percentile=99.0)

    w = float(np.clip(GRADIENT_WEIGHT, 0.0, 0.8))
    profile = (1.0 - w) * b + w * g

    m = min(max(0, int(EDGE_MARGIN)), len(profile) // 4)
    if m > 0:
        profile[:m] = 0
        profile[-m:] = 0

    return smooth_1d(profile, SMOOTH_KERNEL)


def _low_intervals(mask, min_len=3):
    out = []
    s = None
    for i, b in enumerate(mask):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if i - s >= min_len:
                out.append((s, i - 1))
            s = None
    if s is not None and len(mask) - s >= min_len:
        out.append((s, len(mask) - 1))
    return out


def find_valleys_optimized(profile, valley_ratio=None, min_gap=20):
    """
    谷值比例真正使用 GUI 的 VALLEY_RATIO。
    先找低结构区，再用局部极小值补充，最后合并过近谷。
    """
    if valley_ratio is None:
        valley_ratio = VALLEY_RATIO

    p = np.asarray(profile, dtype=np.float32)
    if p.size < 2:
        return []

    p = smooth_1d(p, 11)
    maxv = float(p.max())
    if maxv <= 1e-12:
        return []

    strict = float(np.clip(valley_ratio, 0.02, 0.90))
    relaxed = min(strict + 0.12, 0.95)

    valleys = _low_intervals(p < strict * maxv, min_len=3)

    # 局部极小值作为补充，但必须足够低
    d = np.gradient(p)
    for i in range(2, len(p) - 2):
        if d[i - 1] < 0 <= d[i + 1] and p[i] < relaxed * maxv:
            lo = i
            hi = i
            while lo > 0 and p[lo - 1] < relaxed * maxv:
                lo -= 1
            while hi < len(p) - 1 and p[hi + 1] < relaxed * maxv:
                hi += 1
            if hi - lo + 1 >= 3:
                valleys.append((lo, hi))

    if not valleys:
        valleys = _low_intervals(p < relaxed * maxv, min_len=3)

    if not valleys:
        return []

    valleys.sort(key=lambda v: (v[0], v[1]))
    merged = []
    for s, e in valleys:
        if not merged:
            merged.append([s, e])
            continue
        ps, pe = merged[-1]
        if s <= pe + max(1, int(min_gap)):
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])

    return [(int(s), int(e)) for s, e in merged]


def find_valleys(profile, valley_ratio=None, min_gap=20):
    return find_valleys_optimized(profile, valley_ratio, min_gap)


def valley_bottom(profile, valley):
    s, e = valley
    if e < s:
        return int(s)
    return int(s + np.argmin(profile[s:e + 1]))


def _longest_true_run(mask):
    best_s = 0
    best_len = 0
    s = None
    for i, b in enumerate(mask):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if i - s > best_len:
                best_s, best_len = s, i - s
            s = None
    if s is not None and len(mask) - s > best_len:
        best_s, best_len = s, len(mask) - s
    return best_s, best_len


def split_columns_by_valleys(profile, w):
    valleys = find_valleys(profile)
    bottoms = [valley_bottom(profile, v) for v in valleys]

    regions = []
    if bottoms:
        regions.append((0, bottoms[0]))
        for a, b in zip(bottoms[:-1], bottoms[1:]):
            regions.append((a, b))
        regions.append((bottoms[-1], w - 1))
    else:
        regions = [(0, w - 1)]

    global_max = float(np.max(profile)) if len(profile) else 0.0
    if global_max <= 0:
        return []

    cols = []
    for left, right in regions:
        left = max(0, int(left))
        right = min(w - 1, int(right))
        if right - left < 3:
            continue

        seg = np.asarray(profile[left:right + 1], dtype=np.float32)
        if seg.size == 0 or float(seg.max()) <= 0:
            continue

        # 旧版用 seg > 0，平滑后几乎整个区间都会 >0。
        # 改成相对活动阈值，真正裁到柱体主体。
        active_thr = max(
            global_max * float(PROFILE_ACTIVE_RATIO),
            float(seg.max()) * 0.12
        )
        active = seg >= active_thr
        rs, rlen = _longest_true_run(active)
        if rlen < 5:
            continue

        l2 = left + rs
        r2 = l2 + rlen - 1
        cx = (l2 + r2) // 2
        cols.append((l2, r2, cx))

    return cols


# =====================================================================
# 几何优先识别：抗柱内纹理、抗亮度漂移、支持边缘残柱
# =====================================================================

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


def _main_row_activity(gray):
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
    upper = [r for r in runs if r[0] < half_lim]
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
        [int(e) - int(s) for _, s, e in ranges],
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


def _detection_confidence(columns, gray=None, width_prior=None):
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

    count_fit = min(3.0, 0.55 * float(len(columns)))
    # binary 轮廓宽度先验（同图）：binary 路径的描边轮廓宽度通常
    # 比几何路径的实心段合并更精确（几何路径会把柱侧的衬底/纹理
    # 并进实心段，如 A689079 equalized 候选框宽 67px vs binary
    # 轮廓 48~57px）。仅用当前图 binary 结果作参考，避免跨图 CD
    # 差异误伤；候选典型宽显著更宽时扣分（背景被并进柱体）。
    width_prior_fit = 0.0
    if width_prior is not None and width_prior > 0:
        devs = np.abs(fit_widths - float(width_prior))
        over = float(np.mean(np.clip(
            devs - float(width_prior) * 0.12, 0.0, None)))
        # 超出先验 12% 的部分按比例扣分，平均每根柱最多扣 0.4
        width_prior_fit = -min(0.4, float(np.mean(
            np.clip(over / max(2.0, float(width_prior)), 0.0, 1.0))) * 1.2)
    return float(
        count_fit
        + 1.45 * width_fit
        + 1.20 * consistency
        + 0.85 * height_fit
        + 0.55 * gap_fit
        + 0.45 * edge_fit
        + contrast_fit
        + width_prior_fit
        - edge_penalty
    )


def _polarity_penalty(columns, orig_gray):
    """PR 常规极性检查：柱心灰度应亮于相邻沟心灰度（对照原图）。

    返回罚分：0=极性正常；>0 表示疑似极性反转（柱心反而更暗），
    即把暗 trench 误当成亮柱。必须用原始灰度而非候选自身图——
    inverted 候选在自身图上永远“柱亮沟暗”，只有对照原图才能
    发现极性被反转（如 A58）。
    """
    if orig_gray is None or not columns:
        return 0.0
    try:
        g = to_gray8(orig_gray)
        h, w = g.shape[:2]
        cols = sorted(columns, key=lambda c: float(c.get("center_x", 0)))
        col_vals = []
        for c in cols:
            l = int(np.clip(c.get("left", 0), 0, w - 1))
            r = int(np.clip(c.get("right", w - 1), 0, w - 1))
            wt = max(2, int(round(0.25 * (r - l + 1))))
            cx0 = int(np.clip(0.5 * (l + r) - wt, 0, w - 1))
            cx1 = int(np.clip(0.5 * (l + r) + wt + 1, 1, w))
            y0 = int(np.clip(c.get("y_top", 0), 0, h - 1))
            y1 = int(np.clip(c.get("y_bottom", h - 1), 0, h - 1))
            if y1 <= y0:
                continue
            col_vals.append(float(np.median(g[y0:y1 + 1, cx0:cx1])))
        if len(col_vals) < 2:
            return 0.0
        tr_vals = []
        for a, b in zip(cols[:-1], cols[1:]):
            ra = int(np.clip(a.get("right", 0), 0, w - 1))
            lb = int(np.clip(b.get("left", 0), 0, w - 1))
            mid = int(np.clip(0.5 * (ra + lb), 0, w - 1))
            y0 = int(np.clip(max(a.get("y_top", 0), b.get("y_top", 0)), 0, h - 1))
            y1 = int(np.clip(min(a.get("y_bottom", h - 1), b.get("y_bottom", h - 1)), 0, h - 1))
            if y1 <= y0:
                continue
            tr_vals.append(float(np.median(g[y0:y1 + 1, max(0, mid - 1):mid + 2])))
        if not tr_vals:
            return 0.0
        diff = float(np.median(col_vals)) - float(np.median(tr_vals))
        if diff < -3.0:
            # 柱心显著暗于沟心：极性反转，重罚让该候选出局
            return 2.5
        if diff < 0.0:
            return 0.6
        return 0.0
    except Exception:
        return 0.0


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


def _row_quality_map(binary):
    """按行内 run 结构区分柱体行/衬底行/刻度行（用于 y 投影过滤）。

    空心轮廓二值图中：
    - 柱体行：多条规则短 run（左/右边缘线），总占空比低；
    - 衬底行：单条超长 run（横向衬底把柱连成一片）；
    - 刻度行：少量不规则 run（总占空比低但 run 数/结构异常）。
    返回 1.0=柱体行 / 0.0=非柱体行 的 float32 向量。
    """
    h, w = binary.shape[:2]
    quality = np.zeros(h, dtype=np.float32)
    if h == 0 or w == 0:
        return quality
    mask = binary > 0
    # 每行 run 数量与最长 run 长度（用差分法向量化）
    padded = np.zeros((h, w + 2), dtype=np.uint8)
    padded[:, 1:-1] = mask.astype(np.uint8)
    starts = (padded[:, 1:-1] > 0) & (padded[:, :-2] == 0)
    ends = (padded[:, 1:-1] > 0) & (padded[:, 2:] == 0)
    n_runs = starts.sum(axis=1).astype(np.float32)
    # 每行最长 run：通过 start/end 对计算各 run 长度后取 max
    max_run = np.zeros(h, dtype=np.float32)
    for yy in range(h):
        st = np.where(starts[yy])[0]
        en = np.where(ends[yy])[0]
        if st.size:
            max_run[yy] = float((en - st + 1).max())
    occupancy = mask.sum(axis=1).astype(np.float32) / float(w)
    # 柱体行判据：多条 run（空心两条边缘）或中等长度实心段，
    # 且最长 run 不至于横向贯穿（衬底行特征）
    multi_run = (n_runs >= 2) & (max_run < w * 0.45)
    single_solid = (n_runs == 1) & (max_run >= 6) & (max_run < w * 0.45)
    quality[(multi_run | single_solid) & (occupancy > 0.001)] = 1.0
    return quality


def _pair_hollow_segments(segs, band_gray=None):
    """把空心轮廓的左右边缘线段解析配对成完整柱体区间。

    空心轮廓下每根柱产生左右两条细边缘线，段间距交替为
    「柱内宽（同柱左右线间） / 沟宽（相邻柱间）」。水平闭运算在
    柱内宽 > 沟宽时无法两全，改用假设检验配对：
    - 假设 A：从第 0 段开始两两配对；
    - 假设 B：首段视为边缘残柱，从第 1 段开始两两配对；
    - 假设 C：段是沟壁线（描边落在沟壁上，如 A55 沟壁梯度强于
      柱壁），build(0) 配对区间是沟而非柱，柱体 = 沟间间隙 +
      图像两端的边缘柱。
    用柱体宽度一致性 + 灰度对比（柱内亮于沟，PR 常规极性）评分，
    选高分假设。返回柱体区间 [(l, r)] 或 None。
    """
    n = len(segs)
    if n < 2:
        return None

    def build(offset):
        spans = []
        i = offset
        if offset == 1:
            spans.append((int(segs[0][0]), int(segs[0][1])))
        while i < n:
            if i + 1 < n:
                spans.append((int(segs[i][0]), int(segs[i + 1][1])))
                i += 2
            else:
                spans.append((int(segs[i][0]), int(segs[i][1])))
                i += 1
        return spans

    def signed_contrast(spans):
        # 柱内中心灰度 - 沟中心灰度（PR 图柱亮于沟，应为正）。
        # 用带内列中位数剖面代替逐列均值，抗噪且忽略描边白线本身。
        if band_gray is None or band_gray.size == 0:
            return 0.0
        w_img = band_gray.shape[1]
        col_prof = np.median(band_gray, axis=0).astype(np.float32)
        col_vals = []
        tr_vals = []
        for k, (l, r) in enumerate(spans):
            c = int(np.clip(0.5 * (l + r), 0, w_img - 1))
            col_vals.append(float(np.median(col_prof[max(0, c - 2):c + 3])))
            if k + 1 < len(spans):
                mid = int(np.clip(
                    0.5 * (spans[k][1] + spans[k + 1][0]), 0, w_img - 1))
                tr_vals.append(float(np.median(col_prof[max(0, mid - 2):mid + 3])))
        if not col_vals or not tr_vals:
            return 0.0
        return float(np.median(col_vals) - np.median(tr_vals))

    def build_complement():
        # 假设 C：段两两配对（build(0)）的区间是沟壁对围成的沟，
        # 柱体 = 沟与沟之间的间隙 + 图像两端的边缘残柱。
        # 适用于描边落在沟壁上的图（A55：沟壁梯度强于柱壁）。
        trenches = build(0)
        if not trenches:
            return None
        w_img = band_gray.shape[1] if band_gray is not None \
            else int(segs[-1][1]) + 1
        spans = []
        if trenches[0][0] >= 8:
            spans.append((0, int(trenches[0][0]) - 1))
        for k in range(len(trenches) - 1):
            l = int(trenches[k][1]) + 1
            r = int(trenches[k + 1][0]) - 1
            if r - l + 1 >= 8:
                spans.append((l, r))
        if w_img - 1 - int(trenches[-1][1]) >= 8:
            spans.append((int(trenches[-1][1]) + 1, int(w_img) - 1))
        return spans if len(spans) >= 2 else None

    best_spans = None
    best_score = -1e9
    for spans in (build(0), build(1) if n >= 3 else None, build_complement()):
        if not spans:
            continue
        ws = np.array([r - l + 1 for l, r in spans], dtype=np.float32)
        med_all = float(np.median(ws))
        main = ws[ws >= 0.35 * med_all]
        if main.size < max(2, len(spans) - 1):
            continue
        med = float(np.median(main))
        if med < 8:
            continue
        mad = float(np.median(np.abs(main - med)))
        consist = -mad / max(2.0, med * 0.15)
        contrast = signed_contrast(spans)
        # 极性硬约束：柱心必须亮于沟心（PR 常规极性）。contrast 显著
        # 为负说明该假设把暗沟配成了柱体（如 A55），直接丢弃该假设；
        # 全部假设都被丢弃时返回 None，binary 路径让位给几何路径。
        if contrast < -3.0:
            continue
        # 灰度对比是柱/沟归属的强证据（柱亮沟暗为正），主导评分；
        # 宽度一致性作为次级平滑约束。
        score = 0.015 * contrast + consist
        if score > best_score:
            best_score = score
            best_spans = spans
    if best_spans is None or best_score < -0.8:
        return None
    return best_spans


def _refine_columns_with_gradient(columns, gray, gx_arr=None, gy_arr=None):
    """把描边边缘吸附到 |grad_x| 峰值，做边缘精修（亚像素级对齐）。

    对每根柱每一行的左/右边界，在 ±snap 邻域内找 |grad_x| 最大值
    位置吸附（梯度显著时才移动），抑制二值化阈值导致的半像素级
    偏差。返回 (精修后 columns 列表, 移动像素数)。
    """
    if not columns:
        return columns, 0
    g = None
    try:
        g = to_gray8(gray) if gray is not None else None
    except Exception:
        g = None
    if g is None:
        return columns, 0
    if gx_arr is None:
        gx_arr = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    if gy_arr is None:
        gy_arr = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    h, w = g.shape[:2]
    try:
        med_w = float(np.median([_column_width_px(c) for c in columns]))
    except Exception:
        med_w = 20.0
    snap = max(2, int(round(0.06 * max(4.0, med_w))))
    moved_total = 0
    refined = []
    for col in columns:
        ranges = list(col.get("ranges") or [])
        if not ranges:
            refined.append(col)
            continue
        new_ranges = []
        for (yy, s, e) in ranges:
            yy_i = int(np.clip(int(yy), 0, h - 1))
            s2, e2 = int(s), int(e)
            lo = max(0, s2 - snap)
            hi = min(w - 2, s2 + snap)
            if hi > lo:
                seg = gx_arr[yy_i, lo:hi + 1]
                pk = int(np.argmax(seg))
                # 峰值需显著强于当前位置（1.8 倍）才吸附，避免逐行
                # 追逐噪声峰导致边缘摆动
                if float(seg[pk]) > max(0.15, 1.8 * float(gx_arr[yy_i, s2])):
                    if lo + pk != s2:
                        moved_total += 1
                    s2 = lo + pk
            lo2 = max(0, e2 - snap)
            hi2 = min(w - 1, e2 + snap)
            if hi2 > lo2:
                seg2 = gx_arr[yy_i, lo2:hi2 + 1]
                pk2 = int(np.argmax(seg2))
                if float(seg2[pk2]) > max(0.15, 1.8 * float(gx_arr[yy_i, e2])):
                    if lo2 + pk2 != e2:
                        moved_total += 1
                    e2 = lo2 + pk2
            if e2 >= s2:
                new_ranges.append((yy_i, s2, e2))
        if len(new_ranges) < max(8, MIN_COLUMN_HEIGHT * 0.6):
            refined.append(col)
            continue
        # 吸附后形状先验拟合：柱侧壁在 SEM 图上是直线/缓变曲线，
        # 梯度吸附会忠实跟随弯曲的梯度峰（binary 边缘犬牙交错），
        # 引入中低频摆动。对左/右边界做逐段线性回归（窗口 15，
        # 步长 5 重叠）拟合，侧壁形状先验可彻底消除锯齿且保留
        # 真实锥度/台阶；与原边缘差异超过 5px 的行视为真实结构
        # （台阶/悬凸），回贴原值保护。
        ys = [r[0] for r in new_ranges]
        lefts = np.array([float(r[1]) for r in new_ranges], dtype=np.float64)
        rights = np.array([float(r[2]) for r in new_ranges], dtype=np.float64)

        def _seg_fit(a, win=15, step=5):
            n = a.size
            if n < win:
                return a.copy()
            out = a.copy()
            yy = np.arange(n, dtype=np.float64)
            for st in range(0, n - win + 1, step):
                sl = slice(st, st + win)
                p = np.polyfit(yy[sl], a[sl], 1)
                fit = np.polyval(p, yy[sl])
                # 只替换偏离回归线 ≤5px 的点（保护真实台阶/悬凸）
                m = np.abs(a[sl] - fit) <= 5.0
                out[sl][m] = fit[m]
            return out

        sm_l = _seg_fit(lefts)
        sm_r = _seg_fit(rights)
        new_ranges = [(yy, int(round(l)) if abs(l - l0) <= 5 else int(l0),
                       int(round(r)) if abs(r - r0) <= 5 else int(r0))
                      for yy, l, r, l0, r0 in zip(ys, sm_l, sm_r, lefts, rights)]
        col2 = dict(col)
        col2["ranges"] = new_ranges
        col2["left"] = int(min(s for _, s, _ in new_ranges))
        col2["right"] = int(max(e for _, _, e in new_ranges))
        col2["y_top"] = int(new_ranges[0][0])
        col2["y_bottom"] = int(new_ranges[-1][0])
        col2["contour"] = build_contour(new_ranges)
        refined.append(col2)
    return refined, moved_total


def _write_debug_test_images(image_path, dbg):
    """输出识别过程测试图（中间阶段可视化），便于人工检查算法行为。

    写入与源图同名的结果目录：
    - dbg1_binary.png       二值图 + y 主体段横线
    - dbg2_yprofile.png     行质量过滤后的 y 投影与阈值
    - dbg3_xprofile.png     x 占空比投影 / 梯度融合证据与阈值
    - dbg4_edges.png        追踪到的原始边缘点（左绿右红）
    - dbg5_gradsnap.png     |grad_x| 热图 + 精修后描边轮廓
    - dbg6_result.png       原图 + 最终柱体描边
    - iter_log.txt          迭代次数 / 边缘移动像素数 / 配对结果
    """
    if not image_path or not isinstance(dbg, dict):
        return
    img = dbg.get("img")
    if img is None:
        return
    stem, out_dir = _output_dir_for(image_path)
    h, w = img.shape[:2]

    # 1. 二值图 + y 主体段
    try:
        src_bin = dbg.get("dbg_binary")
        if src_bin is None or src_bin.shape[:2] != (h, w):
            src_bin = np.zeros((h, w), np.uint8)
        vis1 = cv2.cvtColor(src_bin, cv2.COLOR_GRAY2BGR)
    except Exception:
        vis1 = np.zeros((h, w, 3), dtype=np.uint8)
    final = dbg.get("final_columns") or []
    if final:
        y_top = min(int(c["y_top"]) for c in final)
        y_bot = max(int(c["y_bottom"]) for c in final)
        cv2.line(vis1, (0, y_top), (w - 1, y_top), (0, 0, 255), 1)
        cv2.line(vis1, (0, y_bot), (w - 1, y_bot), (0, 0, 255), 1)
        # 标注最终识别结果：每根柱的左右边界竖线（黄）+ 编号 + 柱宽
        for c in final:
            lx, rx = int(c["left"]), int(c["right"])
            cv2.line(vis1, (lx, y_top), (lx, y_bot), (0, 255, 255), 1)
            cv2.line(vis1, (rx, y_top), (rx, y_bot), (0, 255, 255), 1)
            cv2.putText(vis1, f"{c.get('label', '')}({rx - lx + 1}px)",
                        (lx, max(10, y_top - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    ok1 = imwrite_unicode(os.path.join(out_dir, f"{stem}_dbg1_binary.png"), vis1)
    if not ok1:
        print("[dbg] dbg1 write failed")
    # 2. y 投影曲线
    vis2 = np.zeros((240, w, 3), dtype=np.uint8)
    y_prof = dbg.get("y_prof")
    if y_prof is not None and y_prof.size:
        ymax = float(y_prof.max()) or 1.0
        pts = [(int(i), 239 - int(np.clip(y_prof[i] / ymax, 0, 1) * 230))
               for i in range(0, len(y_prof), max(1, len(y_prof) // w))]
        for p in pts:
            cv2.circle(vis2, p, 1, (0, 255, 0), -1)
    y_thr = dbg.get("y_thr")
    if y_thr is not None:
        yy = 239 - int(np.clip(y_thr, 0, 1) * 230)
        cv2.line(vis2, (0, yy), (w - 1, yy), (0, 0, 255), 1)
    # 标注最终识别结果：每根柱在 y 投影图上的区间（顶部青色横线 + 编号）
    for c in final:
        yy = 239 - int(np.clip(1.0, 0, 1) * 230)
        lx = max(0, int(c["left"]))
        rx = min(w - 1, int(c["right"]))
        cv2.line(vis2, (lx, yy - 6), (rx, yy - 6), (255, 255, 0), 2)
        cv2.putText(vis2, str(c.get("label", "")),
                    (lx, max(10, yy - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    imwrite_unicode(os.path.join(out_dir, f"{stem}_dbg2_yprofile.png"), vis2)

    # 3. x 投影 / 融合证据
    vis3 = np.zeros((240, w, 3), dtype=np.uint8)
    x_proj = dbg.get("x_proj")
    fuse = dbg.get("fuse_xproj")
    if fuse is not None and fuse.size:
        src = fuse
    elif x_proj is not None and x_proj.size:
        src = x_proj
    else:
        src = None
    if src is not None:
        xmax = float(src.max()) or 1.0
        for i in range(w):
            if i < src.size:
                yy = 239 - int(np.clip(src[i] / xmax, 0, 1) * 230)
                cv2.circle(vis3, (i, yy), 1, (255, 200, 0), -1)
    thr = dbg.get("threshold")
    if thr is not None and src is not None:
        yy = 239 - int(np.clip(thr, 0, 1) * 230)
        cv2.line(vis3, (0, yy), (w - 1, yy), (0, 0, 255), 1)
    # 标注最终识别结果：每根柱的 x 区间（顶部青色横线 + 编号），
    # 直观对照 x 投影峰与最终柱归属是否一致
    for c in final:
        lx = max(0, int(c["left"]))
        rx = min(w - 1, int(c["right"]))
        cv2.line(vis3, (lx, 6), (rx, 6), (255, 255, 0), 2)
        cv2.putText(vis3, str(c.get("label", "")),
                    (lx, max(10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
    imwrite_unicode(os.path.join(out_dir, f"{stem}_dbg3_xprofile.png"), vis3)

    # 4. 追踪到的原始边缘点 + 逐层扫描精确点标注：
    #    白/黑点 = binary 描边还原（左绿右红）；
    #    亮点   = enhanced 灰度曲线上由 grad_x 找到的精确边缘点；
    #    品红点 = 由 grad_y 找到的垂直边缘点。
    vis4 = cv2.cvtColor(dbg.get("dbg_binary", np.zeros((h, w), np.uint8)),
                        cv2.COLOR_GRAY2BGR)
    for ci, c in enumerate(final):
        color = PR_COLORS[ci % len(PR_COLORS)]
        for (yy, s, e) in c.get("ranges", [])[:: max(1, len(c["ranges"]) // 60 or 1)]:
            cv2.circle(vis4, (int(s), int(yy)), 1, (0, 255, 0), -1)
            cv2.circle(vis4, (int(e), int(yy)), 1, (0, 0, 255), -1)
    pts_h = dbg.get("scan_points_h") or []
    pts_v = dbg.get("scan_points_v") or []
    for px, py in pts_h:
        cv2.circle(vis4, (int(px), int(py)), 1, (255, 255, 255), -1)
    for px, py in pts_v:
        cv2.circle(vis4, (int(px), int(py)), 1, (255, 0, 255), -1)
    cv2.putText(vis4, "white=grad_x pts  magenta=grad_y pts",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    # 标注最终识别结果：每根柱的黄色包络框（顶/底横线 + 左右竖线）+ 编号
    if final:
        y_top_all = min(int(c["y_top"]) for c in final)
        y_bot_all = max(int(c["y_bottom"]) for c in final)
        for c in final:
            lx, rx = int(c["left"]), int(c["right"])
            cv2.rectangle(vis4, (lx, y_top_all), (rx, y_bot_all),
                          (0, 255, 255), 1)
            cv2.putText(vis4, str(c.get("label", "")),
                        (lx, max(10, y_top_all - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    imwrite_unicode(os.path.join(out_dir, f"{stem}_dbg4_edges.png"), vis4)

    # 5. |grad_x| 热图 + 最终描边
    g = dbg.get("gray")
    if g is not None:
        gx_arr = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        gx8 = np.clip(gx_arr / (float(gx_arr.max()) or 1.0) * 255.0,
                      0, 255).astype(np.uint8)
        vis5 = cv2.applyColorMap(gx8, cv2.COLORMAP_JET)
    else:
        vis5 = np.zeros((h, w, 3), dtype=np.uint8)
    overlay = img.copy()
    for c in final:
        pts = c.get("contour")
        if pts is not None:
            cv2.polylines(overlay, [pts.astype(np.int32)], True, (0, 255, 255), 1)
    vis5 = cv2.addWeighted(vis5, 0.75, overlay, 0.25, 0)
    # 标注最终识别结果：每根柱的黄色边界框 + 编号 + 柱宽（与 dbg6 一致）
    for c in final:
        lx, rx = int(c["left"]), int(c["right"])
        yt, yb = int(c["y_top"]), int(c["y_bottom"])
        cv2.rectangle(vis5, (lx, yt), (rx, yb), (0, 255, 255), 1)
        cv2.putText(vis5, f"{c.get('label', '')}({rx - lx + 1}px)",
                    (lx, max(12, yt - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    imwrite_unicode(os.path.join(out_dir, f"{stem}_dbg5_gradsnap.png"), vis5)

    # 6. 原图 + 最终描边 + 识别结果标注（边界框/编号/柱宽/模式）
    vis6 = img.copy()
    for c in final:
        pts = c.get("contour")
        lx, rx = int(c["left"]), int(c["right"])
        yt, yb = int(c["y_top"]), int(c["y_bottom"])
        if pts is not None:
            cv2.polylines(vis6, [pts.astype(np.int32)], True, (0, 255, 255), 1)
        cv2.rectangle(vis6, (lx, yt), (rx, yb), (0, 255, 255), 1)
        cv2.putText(vis6, f"{c.get('label', '')}({rx - lx + 1}px)",
                    (lx, max(12, yt - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    mode = dbg.get("detection_mode")
    if mode:
        cv2.putText(vis6, f"mode={mode}  cols={len(final)}",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1)
    imwrite_unicode(os.path.join(out_dir, f"{stem}_dbg6_result.png"), vis6)

    # 7. enhanced 图逐行/逐列灰度曲线：上为 enhanced 图（含采样线），
    #    中为采样行的灰度曲线瀑布图（grad_x 精确点标红），
    #    下为采样列的灰度曲线瀑布图（grad_y 精确点标红）。
    try:
        if g is not None:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(g)
            region = dbg.get("scan_region") or (0, h - 1)
            ry0, ry1 = int(region[0]), int(region[1])
            n_slot = 7
            slot_h = 30
            wf_h = n_slot * slot_h
            vis7 = np.vstack([
                cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                np.zeros((wf_h * 2 + 8, w, 3), dtype=np.uint8),
            ])
            # 采样行：在扫描区域内均匀取 n_slot 行
            ys_sample = np.linspace(max(1, ry0), max(1, ry1), n_slot).astype(int) \
                if ry1 > ry0 else [max(1, min(h - 2, ry0))]
            pts_by_row = {}
            for px, py in pts_h:
                pts_by_row.setdefault(int(py), []).append(int(px))
            for si, sy in enumerate(ys_sample):
                sy = int(np.clip(sy, 0, h - 1))
                base = h + si * slot_h
                cv2.line(vis7, (0, sy), (w - 1, sy), (255, 0, 255), 1)
                curve = enhanced[sy, :].astype(np.float32)
                cmax = float(curve.max()) or 1.0
                for xx in range(w):
                    yy = base + slot_h - 2 - int(curve[xx] / cmax * (slot_h - 6))
                    vis7[yy, xx] = (0, 255, 0)
                for px in pts_by_row.get(sy, []):
                    yy = base + slot_h - 2 - int(curve[px] / cmax * (slot_h - 6))
                    cv2.circle(vis7, (px, yy), 2, (0, 0, 255), -1)
            # 采样列：在边缘点覆盖范围内均匀取 n_slot 列
            all_px = [px for px, _ in pts_h] or list(range(0, w, max(1, w // 8)))
            xs_sample = np.linspace(int(np.min(all_px)), int(np.max(all_px)),
                                    n_slot).astype(int) if all_px else []
            pts_by_col = {}
            for px, py in pts_v:
                pts_by_col.setdefault(int(px), []).append(int(py))
            for si, sx in enumerate(xs_sample):
                sx = int(np.clip(sx, 0, w - 1))
                base = h + wf_h + 8 + si * slot_h
                cv2.line(vis7, (sx, 0), (sx, h - 1), (255, 0, 255), 1)
                ry0c = int(np.clip(ry0, 0, h - 2))
                ry1c = int(np.clip(ry1, ry0c + 1, h - 1))
                curve = enhanced[ry0c:ry1c + 1, sx].astype(np.float32)
                if curve.size < 2:
                    continue
                cmax = float(curve.max()) or 1.0
                for k in range(curve.size):
                    xx = sx
                    yy = base + slot_h - 2 - int(curve[k] / cmax * (slot_h - 6))
                    if 0 <= yy < vis7.shape[0]:
                        vis7[yy, xx] = (255, 200, 0)
                for py in pts_by_col.get(sx, []):
                    if ry0c <= py <= ry1c:
                        k = py - ry0c
                        yy = base + slot_h - 2 - int(curve[k] / cmax * (slot_h - 6))
                        cv2.circle(vis7, (sx, yy), 2, (0, 0, 255), -1)
            cv2.putText(vis7, "row curves (green) / col curves (yellow), red=precise pts",
                        (8, vis7.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
            # 标注最终识别结果：enhanced 图顶部每根柱的边界竖线（黄）
            # + 编号，直观对照灰度曲线峰（亮柱）与最终选区是否对齐
            for c in final:
                lx, rx = int(c["left"]), int(c["right"])
                cv2.line(vis7, (lx, 0), (lx, h - 1), (0, 255, 255), 1)
                cv2.line(vis7, (rx, 0), (rx, h - 1), (0, 255, 255), 1)
                cv2.putText(vis7, str(c.get("label", "")),
                            (lx, max(10, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            imwrite_unicode(os.path.join(out_dir, f"{stem}_dbg7_curves.png"), vis7)
    except Exception as e:
        print("[dbg] dbg7 write failed: %s" % e)

    # 7. 迭代日志
    try:
        lines = [
            "iters=%s" % dbg.get("iters"),
            "moved_px=%s" % dbg.get("moved_px"),
            "pair_spans=%s" % dbg.get("pair_spans"),
            "y_segs_thr=%s" % dbg.get("y_thr"),
            "x_threshold=%s" % dbg.get("threshold"),
            "close_w=%s" % dbg.get("close_w"),
            "columns=%d" % len(final),
            "scan_points_h=%d" % len(dbg.get("scan_points_h") or []),
            "scan_points_v=%d" % len(dbg.get("scan_points_v") or []),
            "scan_region=%s" % (dbg.get("scan_region"),),
            "scipy=%s" % SCIPY_AVAILABLE,
        ]
        with open(os.path.join(out_dir, f"{stem}_iter_log.txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print("[iter_log] write failed: %s" % e)


def _iterative_refine_columns(columns, binary, gray,
                              max_iters=3):
    """用前一轮识别统计作为先验，迭代重修柱体边界，直到收敛。

    每轮：
    1. 从当前柱体集合统计典型 CD（中位宽度）与典型间距；
    2. 用 CD 先验重跑 binary 路径（prior_cd_override），
       修正空心合并/配对阈值和边缘追踪跳变限制；
    3. 新旧结果逐柱比较中心位置：稳定（全部中心移动 ≤ 1px）
       即收敛，提前结束。
    返回 (最优 columns, 迭代次数)。
    """
    if not columns:
        return columns, 0
    best = columns
    prev_centers = [0.5 * (c["left"] + c["right"]) for c in columns]
    prev_cd = float(np.median([_column_width_px(c) for c in columns]))
    used = 0
    for it in range(1, max(1, int(max_iters)) + 1):
        used = it
        try:
            new_cols, _ = _detect_columns_from_binary(
                binary, gray, roi_mode=False,
                prior_cd_override=prev_cd)
        except Exception:
            break
        if not new_cols or len(new_cols) != len(best):
            break
        new_centers = [0.5 * (c["left"] + c["right"]) for c in new_cols]
        max_shift = float(np.max(np.abs(
            np.asarray(new_centers) - np.asarray(prev_centers))))
        new_cd = float(np.median([_column_width_px(c) for c in new_cols]))
        if max_shift <= 1.0 and abs(new_cd - prev_cd) <= 1.0:
            best = new_cols
            prev_centers = new_centers
            prev_cd = new_cd
            break
        # 评分择优：只有新结果置信度不低于旧结果时才接受
        old_score = _detection_confidence(best, gray)
        new_score = _detection_confidence(new_cols, gray)
        if new_score >= old_score - 0.05:
            best = new_cols
            prev_centers = new_centers
            prev_cd = new_cd
        else:
            break
    return best, used


def _detect_columns_from_binary(binary, gray, roi_mode=False, debug=None,
                                prior_cd_override=None):
    """从 binary 图中利用轮廓查找进行柱识别（辅助路径）。
    
    当自动识别遇到困难时（如二值化边界不连续、柱内有纹理），
    尝试用形态学闭运算连接断点，再用轮廓查找提取柱体。
    适用于你的 binary.png 中柱体边界清晰但存在断裂的情况。
    
    优化点：
    1. 增大闭运算核尺寸以连接更宽的断裂
    2. 添加轮廓面积过滤，避免过小/过大误检
    3. 使用轮廓近似减少锯齿
    4. 改进逐行边界追踪精度
    """
    try:
        h, w = binary.shape[:2]
        g = to_gray8(gray)
        dbg = debug if isinstance(debug, dict) else None
        if dbg is not None:
            dbg["dbg_binary"] = binary.copy()
            dbg["gray"] = g
            dbg.setdefault("y_prof", None)
            dbg.setdefault("row_ok", None)
            dbg.setdefault("y_thr", None)
            dbg.setdefault("x_proj", None)
            dbg.setdefault("threshold", None)
            dbg.setdefault("y0", 0)
            dbg.setdefault("y1", 0)
            dbg.setdefault("fuse_xproj", None)
            dbg.setdefault("close_w", 0)
            dbg.setdefault("pair_spans", None)
            dbg.setdefault("moved_px", 0)

        # binary 自身的 y 投影找主体行：柱体（含空心轮廓）在投影上
        # 明显高于背景/刻度。对投影做 Otsu 分割，选最长主体段。
        # PR profile 位于图像上半部：先用行质量图剔除横向衬底行
        # （单条超长 run 会把 x 投影连成一片），再在上半部内选主体段，
        # 避免选中底部刻度区。
        row_ok = _row_quality_map(binary)
        y_prof = np.mean(binary > 0, axis=1).astype(np.float32) * row_ok
        y_prof = smooth_1d(y_prof, 5)
        if y_prof.size == 0 or float(y_prof.max()) <= 1e-6:
            return None, None
        if dbg is not None:
            dbg["y_prof"] = y_prof.copy()
            dbg["row_ok"] = row_ok.copy()
            dbg["y_thr"] = None
        y_scaled = np.clip(np.round(y_prof * 255.0), 0, 255).astype(np.uint8)
        y_otsu, _ = cv2.threshold(y_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        y_thr = float(np.clip(float(y_otsu) / 255.0, 0.10, 0.60))
        if dbg is not None:
            dbg["y_thr"] = float(y_thr)
        y_segs = _run_segments(y_prof >= y_thr, min_len=4)
        if not y_segs:
            return None, None
        # 空心轮廓（gradient 阈值）下柱体只有上下边缘两条短带，
        # 需要把相近 y 段合并成完整柱体区间再选最长段。
        # 上半部约束：PR profile 主体在图像上部，优先在上半部内
        # 合并选段；上半部没有有效段时才回退全图最长段。
        half_y = int(h * 0.55)
        y_segs_scope = [seg for seg in y_segs if seg[0] <= half_y] or y_segs
        merged_y = [list(y_segs_scope[0])]
        for s, e in y_segs_scope[1:]:
            if s - merged_y[-1][1] - 1 <= int(h * 0.45):
                merged_y[-1][1] = int(e)
            else:
                merged_y.append([int(s), int(e)])
        y_seg = max([(int(s), int(e)) for s, e in merged_y], key=lambda p: p[1] - p[0] + 1)
        y_top, y_bottom = int(y_seg[0]), int(y_seg[1])
        if y_bottom - y_top + 1 < max(MIN_COLUMN_HEIGHT, 12):
            return None, None

        # 主体行内只取上部 80%（底部 20% 常有衬底把柱连成一片），
        # 对 binary 做 x 方向占空比投影，再用 Otsu 阈值分离柱体/背景
        y0 = max(0, int(y_top))
        y1 = max(y0 + 1, int(y_top) + int(round((y_bottom - y_top + 1) * 0.80)))
        y1 = min(y1, int(y_bottom), h - 1)
        band = binary[y0:y1 + 1]
        x_proj = np.mean(band > 0, axis=0).astype(np.float32)
        if x_proj.size == 0 or float(x_proj.max()) <= 1e-6:
            return None, None
        x_proj = smooth_1d(x_proj, 5)

        scaled = np.clip(np.round(x_proj * 255.0), 0, 255).astype(np.uint8)
        otsu_t, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(np.clip(float(otsu_t) / 255.0, 0.12, 0.72))

        segs = _run_segments(x_proj >= threshold, min_len=6)
        if not segs:
            return None, None
        if dbg is not None:
            dbg["x_proj"] = x_proj.copy()
            dbg["threshold"] = float(threshold)
            dbg["y0"] = int(y0)
            dbg["y1"] = int(y1)

        # 柱内细小缺口用保守合并消除（间距 ≤ 0.2×典型柱宽）
        widths = np.array([e - s + 1 for s, e in segs], dtype=np.float32)
        typical = float(np.median(widths)) if widths.size else 5.0
        merge_gap = max(4, int(round(0.20 * typical)))
        merged = []
        for s, e in segs:
            if merged and s - merged[-1][1] - 1 <= merge_gap:
                merged[-1][1] = int(e)
            else:
                merged.append([int(s), int(e)])
        segs = [(int(s), int(e)) for s, e in merged]
        if not segs:
            return None, None

        # 空心轮廓检测（gradient 阈值下柱体只剩左右两条细边缘线）：
        # 典型段宽远小于 Line CD 先验时，用水平闭运算把左右边缘线
        # 合并回完整柱体，避免把同一根柱拆成两条细线。
        # 先验不可用时（GUI 未启动 / Line CD=0），用段宽与段间距的关系
        # 判断：细线对的间距（柱宽）远大于线宽本身 → 视为空心轮廓。
        try:
            if prior_cd_override is not None and float(prior_cd_override) > 0:
                prior_cd = float(prior_cd_override)
            else:
                hint = _auto_cd_range()
                prior_cd = float(hint[0]) if hint else 0.0
        except Exception:
            prior_cd = 0.0
        hollow_like = False
        if prior_cd > 0:
            hollow_like = typical < prior_cd * 0.55
        elif len(segs) >= 3:
            # 细线对：相邻段中心距（≈柱宽+沟宽）通常 > 3×段宽
            centers_s = [0.5 * (s + e) for s, e in segs]
            gaps_s = np.diff(np.asarray(centers_s, dtype=np.float32))
            if gaps_s.size and float(np.median(gaps_s)) > typical * 3.0:
                hollow_like = True
        max_jump_local = None
        if hollow_like and len(segs) >= 2:
            # grad_x 边缘证据融合：细线缺失（描边不完整）时，梯度
            # 列剖面仍保留完整边缘峰，把 binary 证据与梯度证据取
            # max 后重新分段，补齐缺失柱的边缘线。
            gx_arr = None
            if gray is not None:
                try:
                    gx_arr = np.abs(cv2.Sobel(
                        to_gray8(gray), cv2.CV_32F, 1, 0, ksize=3))
                except Exception:
                    gx_arr = None
            if gx_arr is not None and gx_arr.shape == binary.shape[:2]:
                gx_prof = smooth_1d(
                    np.mean(gx_arr[y0:y1 + 1], axis=0).astype(np.float32), 5)
                gx_n = normalize_01(gx_prof, 99.0)
                gx_ev = np.clip((gx_n - 0.25) / 0.35, 0.0, 1.0)
                fuse = np.maximum(x_proj, gx_ev)
                segsF = _run_segments(fuse >= threshold, min_len=6)
                if len(segsF) > len(segs):
                    wF = np.array([e - s + 1 for s, e in segsF], dtype=np.float32)
                    if float(np.median(wF)) <= typical * 1.6:
                        # 融合后段宽仍在细线量级（未把相邻柱连起来），
                        # 且柱数增加 → 采纳融合结果
                        segs = [(int(s), int(e)) for s, e in segsF]
                        widths = np.array([e - s + 1 for s, e in segs], dtype=np.float32)
                        typical = float(np.median(widths))
            close_w = max(5, int(round((prior_cd if prior_cd > 0 else float(np.median(gaps_s)) * 0.5) * 1.2)) | 1)
            # 闭运算核不得超过最小段间距（否则会把相邻柱连成一片）
            close_w = min(close_w, max(5, int(round(0.9 * float(np.min(gaps_s)))) | 1))
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, 1))
            closed2 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
            band2 = closed2[y0:y1 + 1]
            x_proj2 = smooth_1d(
                np.mean(band2 > 0, axis=0).astype(np.float32), 5)
            segs2 = _run_segments(x_proj2 >= threshold, min_len=6)
            if dbg is not None:
                dbg["fuse_xproj"] = fuse.copy()
                dbg["segs_fused"] = list(segs)
                dbg["close_w"] = int(close_w)
                dbg["dbg_closed"] = closed2.copy()
            if segs2:
                w2 = np.array([e - s + 1 for s, e in segs2], dtype=np.float32)
                if float(np.median(w2)) > typical * 1.8:
                    segs = [(int(s), int(e)) for s, e in segs2]
                    binary = closed2  # 逐行扫描同样在合并后的掩码上
                elif float(np.median(w2)) < typical * 1.5:
                    # 合并后段宽没有显著增长（如孤立中心线，不是左右
                    # 边缘对），尝试解析配对：把左右边缘线两两配成
                    # 完整柱体；配对失败才放弃 binary 路径。
                    band_gray = g[y0:y1 + 1]
                    spans = _pair_hollow_segments(segs, band_gray)
                    if dbg is not None:
                        dbg["pair_spans"] = list(spans) if spans else None
                    if not spans:
                        return None, None
                    segs = [(int(l), int(r)) for l, r in spans]
                    widths = np.array([e - s + 1 for s, e in segs], dtype=np.float32)
                    typical = float(np.median(widths))
                    max_jump_local = max(6, int(round(0.35 * max(6.0, typical))))

        columns = []
        # 行间最大跳变：允许真实边缘的缓慢倾斜，抑制噪声野点
        # （配对后 typical 为完整柱宽，跳变限制相应放宽）
        max_jump = max(6, int(round(0.35 * max(6.0, typical))))
        if max_jump_local is not None:
            max_jump = max_jump_local
        for seg_l, seg_r in segs:
            # 逐行扫描描边像素：把行内二值点分成连续簇。
            # 多簇 → 左/右边缘取首/末簇中心（空心轮廓的两条边线）；
            # 单簇宽段 → 实心行（整段即柱体）；
            # 单簇细线 → 待归属，按邻近有效边缘就近归类
            raw_left = {}
            raw_right = {}
            pending = []
            min_solid = max(4, int(round(typical * 0.6)))
            for yy in range(max(0, int(y_top)), min(int(y_bottom), h - 1) + 1):
                row = binary[yy, seg_l:seg_r + 1]
                xs = np.where(row > 0)[0]
                if xs.size == 0:
                    continue
                splits = np.where(np.diff(xs) > 1)[0]
                clusters = np.split(xs, splits + 1)
                if len(clusters) >= 2:
                    raw_left[yy] = int(round(seg_l + float(np.mean(clusters[0]))))
                    raw_right[yy] = int(round(seg_l + float(np.mean(clusters[-1]))))
                else:
                    c0 = clusters[0]
                    if int(c0[-1] - c0[0] + 1) >= min_solid:
                        raw_left[yy] = int(seg_l + c0[0])
                        raw_right[yy] = int(seg_l + c0[-1])
                    else:
                        pending.append(
                            (yy, int(seg_l + int(round(0.5 * (c0[0] + c0[-1]))))))
            # 细线归属：与上方最近的有效左/右边缘比较，就近归类
            for yy, cx in pending:
                ref_l = ref_r = None
                for dy in range(1, 8):
                    if ref_l is None and (yy - dy) in raw_left:
                        ref_l = raw_left[yy - dy]
                    if ref_r is None and (yy - dy) in raw_right:
                        ref_r = raw_right[yy - dy]
                    if ref_l is not None and ref_r is not None:
                        break
                if ref_l is None and ref_r is None:
                    continue
                if ref_r is None or (
                        ref_l is not None
                        and abs(cx - ref_l) <= abs(cx - ref_r)):
                    raw_left[yy] = cx
                else:
                    raw_right[yy] = cx
            if len(raw_left) < max(8, MIN_COLUMN_HEIGHT * 0.6):
                continue

            y_lo = int(min(raw_left))
            y_hi = int(max(raw_left))
            if y_hi - y_lo + 1 < max(12, MIN_COLUMN_HEIGHT * 0.75):
                continue

            # 轮廓线连续性约束 + 平滑处理：野点剔除、断裂行插值、
            # 滑动中值 + 均值平滑，得到连续平滑的左/右边缘线
            left_line = _trace_edge_line(raw_left, y_lo, y_hi, max_jump)
            right_line = _trace_edge_line(raw_right, y_lo, y_hi, max_jump)

            ranges = []
            for i, yy in enumerate(range(y_lo, y_hi + 1)):
                lv = float(left_line[i])
                rv = float(right_line[i])
                if np.isfinite(lv) and np.isfinite(rv) and rv >= lv:
                    ranges.append((yy, int(round(lv)), int(round(rv))))
            if len(ranges) < max(8, MIN_COLUMN_HEIGHT * 0.6):
                continue

            widths_c = np.array([e - s + 1 for _, s, e in ranges], dtype=np.float32)
            if float(np.mean(widths_c)) < max(3.0, MIN_AVG_WIDTH * 0.45):
                continue

            columns.append({
                "left": int(min(s for _, s, _ in ranges)),
                "right": int(max(e for _, _, e in ranges)),
                "center_x": int(round((seg_l + seg_r) * 0.5)),
                "y_top": int(ranges[0][0]),
                "y_bottom": int(ranges[-1][0]),
                "y_proj": y_prof,
                "ranges": ranges,
                "contour": build_contour(ranges),
            })
        if not columns:
            return None, None

        # 逐层扫描优化：在 PR profile 轮廓区域内双向扫描——
        # 先对逐横行灰度曲线找点（grad_x 水平边缘），再对逐列灰度
        # 曲线找点（grad_y 垂直边缘），合并后用轮廓连续性约束。
        profile_columns, profile_stats = _scan_contour_profile(
            binary, g, int(columns[0]["y_top"]), int(columns[-1]["y_bottom"]),
            max_jump=0.35, debug=dbg)
        # 逐层扫描边缘点覆盖充分时才采纳（避免退化结果覆盖原轮廓）。
        # 防退化检查：扫描柱数不得少于基准柱数，且扫描单柱典型宽度
        # 不得异常大于基准（否则说明把相邻柱连成一片了）。
        scan_ok = False
        if profile_columns and profile_stats:
            scan_ok = profile_stats.get("edge_points", 0) > len(columns) * 0.8
            if scan_ok and len(profile_columns) < len(columns):
                scan_ok = False
            if scan_ok:
                base_w = float(np.median([
                    max(1, _column_width_px(c)) for c in columns]))
                scan_w = float(np.median([
                    max(1, _column_width_px(c)) for c in profile_columns]))
                if scan_w > base_w * 1.8:
                    scan_ok = False
        if scan_ok:
            columns = profile_columns

        # grad_x/grad_y 边缘精修：把描边边缘吸附到梯度峰值
        columns, moved = _refine_columns_with_gradient(columns, g)
        if dbg is not None:
            dbg["moved_px"] = int(moved)
            dbg["columns"] = [
                {"left": int(c["left"]), "right": int(c["right"]),
                 "y_top": int(c["y_top"]), "y_bottom": int(c["y_bottom"]),
                 "ranges": list(c["ranges"])}
                for c in columns]

        # 输出连续编号与颜色；x 剖面用主体段占空比投影（供综合图）
        for idx, col in enumerate(columns, 1):
            col["label"] = idx
            col["color"] = PR_COLORS[(idx - 1) % len(PR_COLORS)]
        return columns, x_proj
    except Exception:
        return None, None


def _scan_contour_profile(binary, gray, y_top, y_bottom, max_jump=0.35,
                          debug=None):
    """在轮廓区域内逐层扫描，通过变化率找到精确的 PR profile 边界。

    策略（对应用户要求的三层证据链）：
    1. binary 轮廓描边已给出柱体大致范围（顶部边界 + 底部谷底连线）；
    2. 在该范围内对 enhanced 图逐行（y 固定）/逐列（x 固定）提取灰度曲线；
    3. 用 grad_x / grad_y 在曲线上找精确边缘点（优先 scipy.find_peaks），
       并应用轮廓连续性约束 + 中值平滑抑制锯齿。

    返回：(columns, profile_stats)
    """
    try:
        h, w = binary.shape[:2]
        g = to_gray8(gray)
        y_lo, y_hi = max(0, int(y_top)), min(int(y_bottom), h - 1)

        # 提取轮廓区域内的梯度剖面
        gx_arr = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        gy_arr = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

        # 区域统一阈值：用整个轮廓区域的梯度分布做基准，避免
        # 柱间隙纯噪声行用自身 85 百分位选出伪边缘点，把 x 直方
        # 图的间隙填满导致多柱连成一片。行级阈值取两者较大者。
        region_gx = gx_arr[y_lo:y_hi + 1]
        region_nz = region_gx[region_gx > 0]
        region_thr = float(np.percentile(region_nz, 90)) * 0.55 \
            if region_nz.size else 0.0

        # ========== 第一步：对逐横行（y 固定）的灰度曲线找点 ==========
        # 在每一行内，找 grad_x 的局部峰值（水平边缘）
        horizontal_edges = []  # [(y, [(x, strength)], ...), ...]

        for yy in range(y_lo, y_hi + 1):
            row_gx = gx_arr[yy, :]

            # 中值滤波抑制椒盐野点，再高斯平滑稳定峰形
            gx_smooth = smooth_1d(
                _median_filter_1d(row_gx.astype(np.float32), 3), 3)

            # 行内自适应阈值与区域统一阈值取较大者（全零行直接跳过）
            nz_row = gx_smooth[gx_smooth > 0]
            if nz_row.size == 0:
                continue
            threshold = max(float(np.percentile(nz_row, 85)) * 0.6,
                            region_thr)

            # scipy.find_peaks 找局部峰值（回退纯 numpy）
            cand = _find_peaks_1d(gx_smooth, min_distance=5, max_count=8)
            selected = [(px, val) for px, val in cand
                        if 10 <= px < w - 10 and val > threshold]
            if selected:
                horizontal_edges.append((yy, selected))
        
        # ========== 第二步：对逐列（x 固定）的灰度曲线找点 ==========
        # 在每一列内，找 grad_y 的局部峰值（垂直边缘）
        vertical_edges = []  # [(x, [(y, strength)], ...), ...]
        
        # 优化：只在 horizontal_edges 覆盖的 x 范围内扫描
        if horizontal_edges:
            all_x = [px for _, peaks in horizontal_edges for px, _ in peaks]
            x_min, x_max = int(np.percentile(all_x, 5)), int(np.percentile(all_x, 95))
        else:
            x_min, x_max = 0, w - 1
        
        for xx in range(x_min, x_max + 1):
            col_gy = gy_arr[y_lo:y_hi + 1, xx]

            if col_gy.size == 0:
                continue

            # 中值滤波 + 高斯平滑稳定峰形
            gy_smooth = smooth_1d(
                _median_filter_1d(col_gy.astype(np.float32), 3), 3)

            # 自适应阈值（全零列直接跳过）
            nz_col = gy_smooth[gy_smooth > 0]
            if nz_col.size == 0:
                continue
            threshold = float(np.percentile(nz_col, 85)) * 0.6

            # scipy.find_peaks 找垂直边缘峰（回退纯 numpy），
            # 边缘点必须落在轮廓区域内（y_lo ~ y_hi）
            cand = _find_peaks_1d(gy_smooth, min_distance=5, max_count=3)
            selected = [(py + y_lo, val) for py, val in cand
                        if 5 <= py < len(gy_smooth) - 5
                        and val > threshold]
            if selected:
                vertical_edges.append((xx, selected))
        
        # ========== 第三步：合并横纵两个方向的边缘点 ==========
        # 构建更密集的边缘点云
        edge_data = []
        
        # 先加入所有水平边缘点
        for yy, peaks in horizontal_edges:
            edge_data.append((yy, peaks))
        
        # 再加入垂直边缘点（如果该 y 行还没有数据）
        vertical_by_y = {}
        for xx, peaks in vertical_edges:
            for yy, val in peaks:
                if yy not in vertical_by_y:
                    vertical_by_y[yy] = []
                vertical_by_y[yy].append((xx, val))
        
        # 合并：对每个有垂直边缘的 y 行，补充到 edge_data
        for yy, peaks in vertical_by_y.items():
            if not any(y == yy for y, _ in edge_data):
                edge_data.append((yy, peaks))
            else:
                # 合并到现有行的峰值列表
                for i, (y, peaks_orig) in enumerate(edge_data):
                    if y == yy:
                        # 去重：过滤距离 < 10px 的重复点
                        all_peaks = peaks_orig + [(px, val) for px, val in peaks 
                                                  if not any(abs(px - p[0]) < 10 for p in peaks_orig)]
                        all_peaks.sort(key=lambda p: p[0])
                        edge_data[i] = (y, all_peaks)
                        break
        
        if len(edge_data) < max(8, (y_hi - y_lo + 1) * 0.4):
            return None, None

        # 检测到的精确边缘点存入 debug，供 dbg4 在图上标注
        if debug is not None:
            pts_h = [(px, yy) for yy, peaks in horizontal_edges
                     for px, _ in peaks]
            pts_v = [(px, yy) for xx, peaks in vertical_edges
                     for _, yy in peaks]
            debug["scan_points_h"] = pts_h
            debug["scan_points_v"] = pts_v
            debug["scan_region"] = (int(y_lo), int(y_hi))
        
        # 聚类分析：识别 x 方向的多组边缘
        # 收集所有边缘点的 x 坐标
        all_x = [px for _, peaks in edge_data for px, _ in peaks]
        if not all_x:
            return None, None
        
        # 用 x 投影分段：找低谷把 x 轴分成多段
        x_hist = np.zeros(w, dtype=np.int32)
        for x in all_x:
            if 0 <= x < w:
                x_hist[x] += 1
        
        # 找低谷（hist < 3 的连续段）
        x_segs = []
        seg_start = None
        for x in range(w):
            if x_hist[x] >= 3:
                if seg_start is None:
                    seg_start = x
            else:
                if seg_start is not None:
                    x_segs.append((seg_start, x - 1))
                    seg_start = None
        if seg_start is not None:
            x_segs.append((seg_start, w - 1))
        
        # 合并相邻段（间距 ≤ 10px）
        merged_xsegs = []
        for seg in x_segs:
            if merged_xsegs and seg[0] - merged_xsegs[-1][1] - 1 <= 10:
                merged_xsegs[-1] = (merged_xsegs[-1][0], seg[1])
            else:
                merged_xsegs.append(seg)
        
        # 对每段进行边缘追踪
        columns = []
        total_edge_points = 0
        
        for seg_l, seg_r in merged_xsegs:
            if seg_r - seg_l + 1 < 15:
                continue
            
            # 收集该段内的边缘点
            left_cands = []
            right_cands = []
            
            for yy, peaks in edge_data:
                # 只选在该段内的峰值
                seg_peaks = [(px, val) for px, val in peaks if seg_l <= px <= seg_r]
                if len(seg_peaks) >= 2:
                    seg_peaks.sort(key=lambda p: p[0])
                    left_cands.append((yy, seg_peaks[0][0]))  # 左边缘
                    right_cands.append((yy, seg_peaks[-1][0]))  # 右边缘
                    total_edge_points += 1
            
            if len(left_cands) < max(8, (y_hi - y_lo + 1) * 0.5):
                continue
            
            # 边缘线追踪 + 平滑
            left_line = _trace_edge_line_smooth(left_cands, y_lo, y_hi, max_jump)
            right_line = _trace_edge_line_smooth(right_cands, y_lo, y_hi, max_jump)
            
            # 构建柱子段
            valid_rows = []
            for i, yy in enumerate(range(y_lo, y_hi + 1)):
                lv = left_line[i] if i < len(left_line) else None
                rv = right_line[i] if i < len(right_line) else None
                if lv is not None and rv is not None and rv > lv:
                    valid_rows.append((yy, int(lv), int(rv)))
            
            if len(valid_rows) < max(8, (y_hi - y_lo + 1) * 0.5):
                continue
            
            # 计算柱子属性
            widths = [e - s for _, s, e in valid_rows]
            typical_w = float(np.median(widths)) if widths else 0
            
            if typical_w < 10:
                continue
            
            # 行宽 y 剖面（供综合结果图 PR 面板绘制）
            y_proj_full = np.zeros(h, dtype=np.float32)
            for vy, vs, ve in valid_rows:
                y_proj_full[vy] = float(ve - vs + 1)

            columns.append({
                "left": int(min(s for _, s, _ in valid_rows)),
                "right": int(max(e for _, _, e in valid_rows)),
                "center_x": int(round(sum(s for _, s, _ in valid_rows) / len(valid_rows))),
                "y_top": int(valid_rows[0][0]),
                "y_bottom": int(valid_rows[-1][0]),
                "width": int(typical_w),
                "ranges": valid_rows,
                "y_proj": y_proj_full,
                "contour": build_contour(valid_rows),
            })
        
        if not columns:
            return None, None
        
        profile_stats = {
            "n_rows": sum(len(c.get("ranges", [])) for c in columns),
            "typical_width": float(np.median([c.get("width", 0) for c in columns])) if columns else 0,
            "edge_points": total_edge_points,
            "n_h_points": sum(len(p) for _, p in horizontal_edges),
            "n_v_points": sum(len(p) for _, p in vertical_edges),
            "scipy": bool(SCIPY_AVAILABLE),
        }
        
        return columns, profile_stats
        
    except Exception:
        return None, None


def _trace_edge_line_smooth(edge_list, y_lo, y_hi, max_jump_factor=0.35):
    """平滑处理边缘线：野点剔除 + 断裂插值 + 滑动平滑。
    
    edge_list: [(y, x), ...] 按 y 排序的候选边缘点
    max_jump_factor: 行间最大跳变因子（相对于典型间距）
    """
    try:
        if not edge_list:
            return [0.0] * (y_hi - y_lo + 1)
        
        # 按 y 排序
        edge_list = sorted(edge_list, key=lambda p: p[0])
        
        # 计算典型 y 间距
        y_vals = [p[0] for p in edge_list]
        if len(y_vals) >= 2:
            typical_dy = float(np.median(np.diff(y_vals)))
        else:
            typical_dy = 1.0
        
        max_jump = max_jump_factor * typical_dy if typical_dy > 0 else 5.0
        
        # 构建完整 y 序列，缺失处插值
        full_line = {}
        for yy, x in edge_list:
            if np.isfinite(x):
                full_line[yy] = x
        
        # 野点剔除 + 插值
        cleaned = {}
        y_sorted = sorted(full_line.keys())
        
        # 计算典型 x 值和跳变阈值
        if len(y_sorted) >= 2:
            x_vals = [full_line[yy] for yy in y_sorted]
            typical_x = float(np.median(x_vals))
            jumps = np.abs(np.diff(x_vals))
            max_allowed_jump = float(np.percentile(jumps, 90)) * 1.5
        else:
            max_allowed_jump = 10.0
        
        for i, yy in enumerate(y_sorted):
            x = full_line[yy]
            if i > 0:
                prev_y = y_sorted[i - 1]
                jump = abs(x - full_line[prev_y])
                if jump > max_allowed_jump:
                    # 野点，用前后平均插值
                    continue
            cleaned[yy] = x
        
        # 线性插值填充断裂
        if len(cleaned) < (y_hi - y_lo + 1) * 0.7:
            y_min, y_max = min(cleaned.keys()), max(cleaned.keys())
            for yy in range(y_min, y_max + 1):
                if yy not in cleaned:
                    # 找前后有效值
                    prev_y = next((k for k in reversed(range(y_min, yy)) if k in cleaned), None)
                    next_y = next((k for k in range(yy, y_max + 1) if k in cleaned), None)
                    if prev_y is not None and next_y is not None:
                        ratio = (yy - prev_y) / float(next_y - prev_y)
                        cleaned[yy] = cleaned[prev_y] + ratio * (cleaned[next_y] - cleaned[prev_y])
                    elif prev_y is not None:
                        cleaned[yy] = cleaned[prev_y]
                    elif next_y is not None:
                        cleaned[yy] = cleaned[next_y]
        
        # 滑动平滑：中值 + 均值组合
        smoothed = []
        median_win = 5
        mean_win = 3
        
        for i, yy in enumerate(range(y_lo, y_hi + 1)):
            if yy in cleaned:
                # 中值滤波去锯齿
                lo_y = max(y_lo, yy - median_win)
                hi_y = min(y_hi, yy + median_win)
                win_vals = [cleaned[y] for y in range(lo_y, hi_y + 1) if y in cleaned]
                if win_vals:
                    median_val = float(np.median(win_vals))
                    # 均值平滑
                    lo_y = max(y_lo, yy - mean_win // 2)
                    hi_y = min(y_hi, yy + mean_win // 2)
                    win_vals = [cleaned[y] for y in range(lo_y, hi_y + 1) if y in cleaned]
                    if win_vals:
                        mean_val = float(np.mean(win_vals))
                        # 组合：中值 + 0.3*均值修正
                        smoothed.append(median_val + 0.3 * (mean_val - median_val))
                    else:
                        smoothed.append(median_val)
                else:
                    smoothed.append(float(cleaned[yy]))
            else:
                # 缺失行，用前值填充
                smoothed.append(float(smoothed[-1]) if smoothed else 0.0)
        
        return smoothed
        
    except Exception:
        return [0.0] * (y_hi - y_lo + 1)


def _find_peaks_1d(sig, min_distance=5, max_count=8):
    """一维信号峰值查找（优先 scipy.signal.find_peaks，回退纯 numpy）。

    返回 [(pos, value), ...]，按位置升序。max_count<=0 表示不限数量。
    """
    sig = np.asarray(sig, dtype=np.float32)
    if sig.size < 3:
        return []
    if _sp_find_peaks is not None:
        kwargs = {}
        if min_distance and min_distance > 1:
            kwargs["distance"] = int(min_distance)
        pk, _ = _sp_find_peaks(sig, **kwargs)
        peaks = [(int(p), float(sig[p])) for p in pk]
    else:
        # 回退实现：高于两侧 5 邻域的局部极大值
        peaks = []
        for i in range(2, sig.size - 2):
            v = sig[i]
            if (v > sig[i - 1] and v >= sig[i + 1]
                    and v >= sig[i - 2] and v >= sig[i + 2]):
                peaks.append((i, float(v)))
        # 就近去重（等价 min_distance 语义）
        dedup = []
        for p, v in peaks:
            if not dedup or p - dedup[-1][0] >= max(1, int(min_distance)):
                dedup.append((p, v))
            elif v > dedup[-1][1]:
                dedup[-1] = (p, v)
        peaks = dedup
    if max_count and max_count > 0 and len(peaks) > int(max_count):
        peaks.sort(key=lambda t: -t[1])
        peaks = peaks[:int(max_count)]
        peaks.sort(key=lambda t: t[0])
    return peaks


def _median_filter_1d(sig, size=5):
    """一维中值滤波（优先 scipy.ndimage.median_filter，回退 numpy 滑窗）。"""
    sig = np.asarray(sig, dtype=np.float32)
    if sig.size == 0 or size <= 1:
        return sig.copy()
    size = int(min(size, sig.size if sig.size % 2 == 1 else sig.size - 1))
    if size <= 1:
        return sig.copy()
    if _sp_median_filter is not None:
        return _sp_median_filter(sig, size=int(size), mode="nearest")
    pad = size // 2
    padded = np.pad(sig, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    return np.median(windows, axis=1).astype(np.float32)


def _trace_edge_line(raw_map, y_lo, y_hi, max_jump, med_win=5, avg_win=3):
    """追踪边缘线：连续性约束 + 平滑处理。"""
    try:
        if not raw_map:
            return []
        
        y_vals = sorted(raw_map.keys())
        if not y_vals:
            return []
        
        # 计算典型 x 值和最大跳变
        # max_jump 调用方传入时已是像素值（如 0.35×典型柱宽），
        # 直接使用；不要乘以 typical（否则阈值放大数百倍，
        # 连续性约束失效，binary 边缘的犬牙交错被原样保留）。
        typical = float(np.median([raw_map[yy] for yy in y_vals]))
        max_jump_px = float(max_jump) if max_jump > 0 else 10.0
        
        # 野点剔除：行间跳变超过阈值，保留偏离中点更远的一端
        cleaned = {}
        for i, yy in enumerate(y_vals):
            x = raw_map[yy]
            if i == 0:
                cleaned[yy] = x
                continue
            
            prev_y = y_vals[i - 1]
            prev_x = cleaned[prev_y]
            jump = abs(x - prev_x)
            
            if jump > max_jump_px:
                # 野点，用前值或中点插值
                mid = (prev_x + typical) * 0.5
                cleaned[yy] = prev_x if abs(prev_x - mid) < abs(x - mid) else x
            else:
                cleaned[yy] = x
        
        # 断裂线性插值
        y_sorted = sorted(cleaned.keys())
        if y_sorted:
            for yy in range(y_sorted[0], y_sorted[-1] + 1):
                if yy not in cleaned:
                    prev_y = next(k for k in reversed(y_sorted) if k < yy)
                    next_y = next(k for k in y_sorted if k > yy)
                    ratio = (yy - prev_y) / float(next_y - prev_y)
                    cleaned[yy] = cleaned[prev_y] + ratio * (cleaned[next_y] - cleaned[prev_y])
        
        # 平滑：中值 + 均值组合
        smoothed = []
        for i, yy in enumerate(range(y_lo, y_hi + 1)):
            if yy in cleaned:
                lo = max(y_lo, yy - med_win)
                hi = min(y_hi, yy + med_win)
                win = [cleaned[y] for y in range(lo, hi + 1) if y in cleaned]
                if win:
                    mval = float(np.median(win))
                    lo = max(y_lo, yy - avg_win // 2)
                    hi = min(y_hi, yy + avg_win // 2)
                    win = [cleaned[y] for y in range(lo, hi + 1) if y in cleaned]
                    if win:
                        aval = float(np.mean(win))
                        smoothed.append(mval + 0.3 * (aval - mval))
                    else:
                        smoothed.append(mval)
                else:
                    smoothed.append(float(cleaned[yy]))
            else:
                smoothed.append(float(smoothed[-1]) if smoothed else 0.0)
        
        return smoothed
    except Exception:
        return [0.0] * (y_hi - y_lo + 1)


def _geometric_gray_candidates(gray, aux=None):
    """生成有限数量的灰度候选，供失败时自动重试。"""
    base = to_gray8(gray)
    out = []

    def add(name, image):
        if image is None:
            return
        arr = to_gray8(image)
        if arr.shape != base.shape:
            return
        # 避免完全重复的候选造成无意义的重复计算。
        if any(np.array_equal(arr, old) for _, old in out):
            return
        out.append((name, arr))

    add("raw", base)
    # 同时保留反相候选，覆盖“PR 柱比背景更暗”的图。候选排序会结合
    # 柱数、CD 先验和边缘一致性自动选择极性；正常亮柱图不会被反相结果
    # 替换。把它放在第二位，确保默认的前两轮就能完成极性判断。
    add("inverted", cv2.bitwise_not(base))
    if isinstance(aux, dict):
        add("enhanced", aux.get("enhanced"))
        add("blur", aux.get("blur"))
    # 这两个候选用于补充曝光偏移或柱壁很暗的图；候选仍由统一评分
    # 决定是否胜出，同时不改变原图输出。
    add("equalized", cv2.equalizeHist(base))
    local = cv2.GaussianBlur(base, (0, 0), 1.2)
    add("local_contrast", cv2.addWeighted(base, 1.45, local, -0.45, 0))
    return out


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
        y_top, y_bottom, row_activity = _main_row_activity(g)
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

def bright_segments(row, lo, hi):
    lo = max(0, int(lo))
    hi = min(len(row) - 1, int(hi))
    if hi < lo:
        return []

    mask = (row[lo:hi + 1] == 255).astype(np.int8)
    diff = np.diff(np.pad(mask, (1, 1), mode="constant"))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return [(lo + int(s), lo + int(e)) for s, e in zip(starts, ends)]


def seg_overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def find_column_y_extent(binary, left, right):
    region = binary[:, left:right + 1]
    row_prof = np.sum(region > 0, axis=1).astype(np.float32)
    row_prof_smooth = smooth_1d(row_prof, SMOOTH_KERNEL)

    maxv = float(row_prof_smooth.max()) if row_prof_smooth.size else 0.0
    if maxv <= 0:
        return 0, 0, row_prof_smooth

    # 阈值取 0.08：顶部往往比主体暗，用更低的门槛把顶部弱信号也纳入
    body = row_prof_smooth > maxv * 0.08
    segs = []
    s = None
    for y, b in enumerate(body):
        if b and s is None:
            s = y
        elif not b and s is not None:
            segs.append((s, y - 1))
            s = None
    if s is not None:
        segs.append((s, len(body) - 1))

    if not segs:
        return 0, 0, row_prof_smooth

    # 合并间隔很小的相邻段（断行/顶帽），避免柱子顶部被切掉
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= MAX_TRACE_GAP + 1:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    main = max(merged, key=lambda p: p[1] - p[0] + 1)

    # y_top 向上延伸到主段上方与之相邻的段（顶部帽/断行）
    y_top = main[0]
    for s, e in merged:
        if e < main[0] and main[0] - e <= MAX_TRACE_GAP + 1:
            y_top = min(y_top, s)

    # 进一步：若 y_top 上方几行仍有微弱信号（>3% 峰值），也纳入柱顶
    # 用于捕捉顶部因对比度低而被二值化切掉的区域
    if y_top > 0 and maxv > 1e-12:
        weak_thr = maxv * 0.03
        new_top = y_top
        for y in range(y_top - 1, max(-1, y_top - 12), -1):
            if row_prof_smooth[y] >= weak_thr:
                new_top = y
            else:
                break
        y_top = new_top

    y_bot_raw = main[1]

    med = float(np.median(row_prof_smooth[y_top:y_bot_raw + 1]))
    if med <= 0:
        return y_top, y_bot_raw, row_prof_smooth

    thr = med * 1.35

    base_top = None
    for y in range(y_bot_raw, y_top - 1, -1):
        if row_prof_smooth[y] >= thr:
            base_top = y
        else:
            break

    if base_top is not None and base_top < y_bot_raw:
        y_bottom = max(y_top, base_top - 1)
    else:
        y_bottom = y_bot_raw

    return int(y_top), int(y_bottom), row_prof_smooth


def _edge_strength_at(grad_row, x, radius=2):
    lo = max(0, int(x) - radius)
    hi = min(len(grad_row) - 1, int(x) + radius)
    if hi < lo:
        return 0.0
    return float(np.mean(grad_row[lo:hi + 1]))


def trace_column_ranges(binary, grad_x, left, right, cx, y_top, y_bottom):
    """
    逐行从下向上追踪。
    候选段评分 = 连通重叠 + 侧壁斜率证据 - 中心跳变 - 宽度跳变。
    """
    ranges = []
    prev_s, prev_e = int(left), int(right)
    gap_count = 0

    for y in range(int(y_bottom), int(y_top) - 1, -1):
        row = binary[y, :]
        grow = grad_x[y, :]

        wl = max(int(left), prev_s - int(WINDOW_MARGIN))
        wr = min(int(right), prev_e + int(WINDOW_MARGIN))

        segs = bright_segments(row, wl, wr)
        if not segs:
            gap_count += 1
            if gap_count > MAX_TRACE_GAP:
                break
            continue

        gap_count = 0
        prev_center = (prev_s + prev_e) * 0.5
        prev_width = max(1, prev_e - prev_s + 1)

        def score_segment(seg):
            s, e = seg
            width = e - s + 1
            center = (s + e) * 0.5

            overlap = seg_overlap(seg, (prev_s, prev_e))
            center_shift = abs(center - prev_center)
            width_shift = abs(width - prev_width)

            # 候选段自己的左右边界梯度，不再是“所有候选共享同一个惩罚”
            edge_strength = (
                _edge_strength_at(grow, s) +
                _edge_strength_at(grow, e)
            ) * 0.5

            # 首几行若窗口还很宽，轻微偏好靠近初始柱中心的段
            center_anchor = abs(center - cx)

            score = (
                2.0 * overlap
                - TRACE_CENTER_PENALTY * center_shift
                - TRACE_WIDTH_PENALTY * width_shift
                - 0.03 * center_anchor
                + TRACE_EDGE_REWARD * edge_strength * prev_width
            )
            return score

        s, e = max(segs, key=score_segment)
        prev_s, prev_e = int(s), int(e)

        if e - s + 1 >= 5:
            ranges.append((int(y), int(s), int(e)))

    ranges.reverse()
    return ranges


def filter_ranges_by_width(ranges, sigma=2.5):
    if len(ranges) < 3:
        return ranges

    widths = np.array([e - s + 1 for _, s, e in ranges], dtype=np.float64)
    med = float(np.median(widths))
    mad = float(np.median(np.abs(widths - med)))
    robust_sigma = max(1.0, 1.4826 * mad)

    lo_thr = max(3.0, med - sigma * robust_sigma)
    hi_thr = med + sigma * robust_sigma

    filtered = [
        r for r, w in zip(ranges, widths)
        if lo_thr <= w <= hi_thr
    ]
    if not filtered:
        return ranges

    # 允许小孔/断行，gap <= MAX_TRACE_GAP 仍视作同一主体
    groups = []
    cur = []
    for r in filtered:
        if not cur or r[0] - cur[-1][0] <= MAX_TRACE_GAP + 1:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    if cur:
        groups.append(cur)

    return max(groups, key=lambda g: (g[-1][0] - g[0][0] + 1, len(g)))


def build_contour(ranges):
    if not ranges:
        return np.empty((0, 1, 2), dtype=np.int32)

    ys = np.array([r[0] for r in ranges], dtype=np.int32)
    xs = np.array([r[1] for r in ranges], dtype=np.int32)
    xe = np.array([r[2] for r in ranges], dtype=np.int32)

    left = np.column_stack((xs, ys))
    right = np.column_stack((xe, ys))[::-1]
    return np.vstack([left, right]).astype(np.int32).reshape((-1, 1, 2))


def detect_one_column(
    binary,
    grad_x,
    left,
    right,
    cx,
    h,
    g_top=0,
    g_bottom=0,
    require_top_half=True,
):
    y_top, y_bottom, y_proj = find_column_y_extent(binary, left, right)

    if require_top_half and y_top >= int(h * 0.5):
        return None

    if g_bottom > 0 and y_bottom > g_bottom + GLOBAL_Y_TOLERANCE:
        y_bottom = g_bottom + GLOBAL_Y_TOLERANCE

    if y_bottom < y_top:
        return None

    ranges = trace_column_ranges(
        binary, grad_x,
        left, right, cx,
        y_top, y_bottom
    )
    ranges = filter_ranges_by_width(ranges)

    if ranges and BOTTOM_OFFSET > 0:
        last_y, s, e = ranges[-1]
        for dy in range(1, int(BOTTOM_OFFSET) + 1):
            yy = last_y + dy
            if yy >= h:
                break
            ranges.append((yy, s, e))

    if len(ranges) < MIN_COLUMN_HEIGHT:
        return None

    widths = np.array([e - s + 1 for _, s, e in ranges], dtype=np.float64)
    avg_width = float(np.mean(widths))
    if avg_width < MIN_AVG_WIDTH:
        return None

    y_min = min(r[0] for r in ranges)
    y_max = max(r[0] for r in ranges)
    height = y_max - y_min + 1

    if height < avg_width * MIN_ASPECT_RATIO:
        return None

    return {
        "left": int(left),
        "right": int(right),
        "center_x": int(cx),
        "y_top": int(y_top),
        "y_bottom": int(y_max),
        "y_proj": y_proj,
        "ranges": ranges,
        "contour": build_contour(ranges),
    }


def _detect_columns(img, binary, aux, roi_mode=False):
    h, w = binary.shape[:2]

    # 几何优先路径：先识别整根实体，再追踪两侧物理边缘。
    # 这一步不使用 preprocess_image 生成的固定二值图，因此柱内亮纹理不会
    # 被拆成窄柱；失败时继续走下面的兼容旧流程。
    if GEOMETRIC_DETECTION_ENABLE:
        gray = aux.get("gray") if isinstance(aux, dict) else None
        if gray is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 第一轮使用原始灰度；若候选数量/宽度一致性不够，再自动尝试少量
        # 对比度版本。每轮都用同一个 CD 先验评分，选分数最高的一轮，
        # 避免单一阈值把整张图锁死在错误结果上。
        variants = _geometric_gray_candidates(gray, aux)
        if not AUTO_RETRY_ENABLE:
            variants = variants[:1]
        try:
            max_attempts = max(1, int(round(float(AUTO_RETRY_MAX))))
        except Exception:
            max_attempts = 1
        if not AUTO_RETRY_ENABLE:
            max_attempts = 1
        max_attempts = min(max_attempts, len(variants))

        best_columns = None
        best_profile = None
        best_gray = None
        best_score = -1e9
        best_variant = "none"
        attempts = 0
        # 原图主体 y 范围：作为各增强候选的搜索带软约束，防止
        # CLAHE 放大基底纹理后柱体追踪穿到基底里（如 A55）。
        hint_extent = _main_row_activity(gray)[:2]
        
        # 新增：先尝试从 binary 图轮廓识别（辅助路径）。
        # binary 路径的轮廓直接描在二值化边缘线上（含连续性约束与
        # 平滑处理），边缘证据更直接；与几何候选比较时给予小幅
        # 证据加成。只影响 binary 与几何路径之间的优先级，
        # 不影响几何候选之间的排序，最终报告的置信度仍是原始分。
        BINARY_CONTOUR_BONUS = 0.30
        _dbg = {} if DEBUG_TEST_OUTPUT else None
        binary_columns, binary_profile = _detect_columns_from_binary(
            binary, gray, roi_mode, debug=_dbg)
        binary_iters = 0
        bin_width_prior = None
        if binary_columns:
            # binary 路径的 y 覆盖范围（含描边带）比原图行活动度更
            # 准确（A55 活动度被颗粒基底压到 y_bottom=87，实际柱脚
            # 在 218）。优先用它更新 hint，让几何候选也搜到完整柱高。
            b_top = min(int(c["y_top"]) for c in binary_columns)
            b_bot = max(int(c["y_bottom"]) for c in binary_columns)
            if b_bot - b_top + 1 > hint_extent[1] - hint_extent[0] + 1:
                hint_extent = (b_top, b_bot)
            # 同图 binary 宽度先验：描边轮廓宽度最接近真实 CD，
            # 用于压分几何候选把背景并进柱体的过宽框。
            _bw = [_column_width_px(c) for c in binary_columns]
            _bw = [v for v in _bw if np.isfinite(v) and v > 0]
            if len(_bw) >= 2:
                bin_width_prior = float(np.median(_bw))
            # 多迭代优化：用前轮统计（典型 CD）作为先验重修，直到收敛
            binary_columns, binary_iters = _iterative_refine_columns(
                binary_columns, binary, gray, max_iters=3)
            score = (_detection_confidence(binary_columns, gray,
                                           width_prior=bin_width_prior)
                     + BINARY_CONTOUR_BONUS
                     - _polarity_penalty(binary_columns, gray))
            if score > best_score:
                best_score = score
                best_columns = binary_columns
                best_profile = binary_profile
                best_gray = gray
                best_variant = "binary_contour"
                attempts = 1
        if _dbg is not None:
            _dbg["iters"] = int(binary_iters)
            _dbg["final_columns"] = binary_columns
            _dbg["img"] = img
            try:
                _write_debug_test_images(
                    aux.get("_image_path") if isinstance(aux, dict) else None,
                    _dbg)
            except Exception:
                pass
        
        for variant_name, variant_gray in variants[:max_attempts]:
            attempts += 1
            candidate_columns, candidate_profile = _detect_columns_geometric(
                variant_gray, roi_mode=roi_mode, y_hint=hint_extent
            )
            if not candidate_columns:
                continue

            candidate_columns = _apply_cd_hint(candidate_columns, w)
            if not candidate_columns:
                continue

            # 每个候选都先经过同一套 footing 修剪/粘连标记/近柱合并，
            # 再计算置信度。否则一个候选可能在评分时看似有 5 根柱，
            # 后处理却把它合并成 1 根，导致“评分结果”和最终输出不一致。
            candidate_columns = smart_refine_columns(candidate_columns)
            candidate_columns, _ = merge_close_columns(
                candidate_columns,
                structure_profile=candidate_profile,
                shallow_profile=_gray_profile_for_merge(variant_gray),
            )
            if not candidate_columns:
                continue

            # 过滤后重新编号，保证输出 PR1…PRn 连续。
            for idx, col in enumerate(candidate_columns, 1):
                col["label"] = idx
                col["color"] = PR_COLORS[(idx - 1) % len(PR_COLORS)]

            # 极性罚分对照原始灰度：inverted 候选在自身图上永远
            # “柱亮沟暗”，只有对照原图才能发现柱/沟极性被反转
            # （如 A58，暗 trench 被反相成亮柱）。
            score = (_detection_confidence(candidate_columns, variant_gray,
                                           width_prior=bin_width_prior)
                     - _polarity_penalty(candidate_columns, gray))
            if score > best_score:
                best_columns = candidate_columns
                best_profile = candidate_profile
                best_gray = variant_gray
                best_score = float(score)
                best_variant = variant_name

            # 不在 raw/inverted 后提前退出：ROI 中原图候选可能只覆盖
            # 一小段主体，而 blur/equalized 候选才恢复完整高度。所有在
            # AUTO_RETRY_MAX 内的有限候选都必须评分，最后保留全局最优。

        if best_columns:
            # 候选在评分前已经完成统一后处理；这里仅重新整理颜色和
            # 最终置信度，确保输出编号连续且分数反映最终轮廓。
            for idx, col in enumerate(best_columns, 1):
                col["label"] = idx
                col["color"] = PR_COLORS[(idx - 1) % len(PR_COLORS)]
            best_score = _detection_confidence(best_columns, best_gray)
            if isinstance(aux, dict):
                # 根据识别路径设置模式标签
                if best_variant == "binary_contour":
                    aux["detection_mode"] = "binary_contour"
                else:
                    aux["detection_mode"] = "geometric"
                aux["detection_variant"] = best_variant
                aux["detection_confidence"] = float(best_score)
                aux["detection_attempts"] = int(attempts)
                try:
                    floor = float(AUTO_RETRY_MIN_CONFIDENCE)
                except Exception:
                    floor = 4.8
                aux["detection_warning"] = (
                    "置信度偏低，请检查灰度曲线或适当调整 Line CD 先验"
                    if float(best_score) < floor else ""
                )
                hint = _auto_cd_range()
                aux["cd_hint_px"] = (
                    {"estimate": float(hint[0]), "tolerance": float(hint[1])}
                    if hint is not None else None
                )
                aux["cd_range_px"] = _reported_cd_range(best_columns, image_width=w)
                aux["cd_observed_px"] = _observed_cd_range(best_columns, image_width=w)
                if not aux.get("detection_warning"):
                    hint_info = _auto_cd_range()
                    observed_info = aux.get("cd_observed_px") or {}
                    if hint_info is not None and observed_info.get("center") is not None:
                        h_center, h_tol = hint_info
                        deviation = abs(float(observed_info["center"]) - h_center)
                        if deviation > max(2.0 * h_tol, 3.0):
                            aux["detection_warning"] = (
                                f"实际观测 CD 中位 {float(observed_info['center']):.1f}px "
                                f"明显偏离先验 {h_center:.1f}±{h_tol:.1f}px"
                            )
            return best_columns, best_profile

    g_top, g_bottom = find_global_y_extent(binary)
    if isinstance(aux, dict):
        aux["detection_mode"] = "binary_legacy"
        aux["detection_variant"] = "legacy"
        aux["detection_confidence"] = -1.0
        aux["detection_attempts"] = 0
        aux["detection_warning"] = "几何路径未建立有效候选，已回退传统二值识别"
        hint = _auto_cd_range()
        aux["cd_hint_px"] = (
            {"estimate": float(hint[0]), "tolerance": float(hint[1])}
            if hint is not None else None
        )
        aux["cd_range_px"] = None
        aux["cd_observed_px"] = _observed_cd_range([], image_width=w)

    profile = column_structure_profile(
        binary,
        aux["grad_x"],
        h,
        g_top,
        g_bottom
    )

    col_defs = split_columns_by_valleys(profile, w)

    columns = []
    for left, right, cx in col_defs:
        col = detect_one_column(
            binary,
            aux["grad_x"],
            left, right, cx,
            h,
            g_top, g_bottom,
            require_top_half=not roi_mode,
        )
        if col is None:
            continue

        col["label"] = len(columns) + 1
        col["color"] = PR_COLORS[(col["label"] - 1) % len(PR_COLORS)]
        columns.append(col)

    # 纵向逐像素扫描驱动的智能后处理（footing 修剪 / 粘连检测 / 近柱合并）
    columns = smart_refine_columns(columns)

    # 近柱合并：使用纯亮度投影判断浅谷（不夹带梯度假谷）
    bright_prof = _brightness_profile(binary, h, g_top, g_bottom)
    columns, _ = merge_close_columns(
        columns,
        structure_profile=profile,
        shallow_profile=bright_prof,
    )

    return columns, profile


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
        for xx in range(last[1] + xoff, last[2] + xoff + 1, 6):
            cv2.line(
                main,
                (xx, bot),
                (min(xx + 3, last[2] + xoff), bot),
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
            return str(int(v))
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


def _edge_ler(rows, right_side=False):
    if len(rows) < 3:
        return None

    ys = np.array([r[0] for r in rows], dtype=np.float64)
    xs = np.array(
        [r[2] if right_side else r[1] for r in rows],
        dtype=np.float64
    )

    b, a = np.polyfit(ys, xs, 1)
    residual = xs - (a + b * ys)
    return 3.0 * float(np.std(residual))


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

    out["ler_left"] = _edge_ler(rows, False)
    out["ler_right"] = _edge_ler(rows, True)

    return out


def column_cd_profile(rows):
    """纵向逐像素扫描结果 → CD(y) 曲线。

    rows: [(y, x_start, x_end), ...]（按 y 升序），单位随输入（nm 或 px）。
    返回 (ys, cds)。
    """
    if not rows:
        return np.array([]), np.array([])

    ys = np.array([float(r[0]) for r in rows], dtype=np.float64)
    cds = np.array([float(r[2]) - float(r[1]) for r in rows], dtype=np.float64)
    return ys, cds


def analyze_shape_metrics(rows):
    """从 CD(y) 曲线提取形貌特征量（纵向逐像素扫描的核心价值）。

    除了分柱，纵向扫描还能给出：
      · 柱体是否上下等宽（CD 均匀性 3σ）
      · 顶部/底部是否展宽（T-top / Footing）
      · 腰部是否内凹（Undercut）
      · 底部展宽台阶的高度与宽度（footing step）
      · 侧壁弯曲量 bow（CD(y) 相对直线的最大偏差）
    """
    keys = (
        "cd_min", "cd_max", "cd_mean", "cd_std", "cd_range",
        "cd_uniformity", "cd_top_seg", "cd_mid_seg", "cd_bot_seg",
        "widen_top", "widen_bot", "waist_ratio", "taper_ratio",
        "footing_step_nm", "footing_height_nm", "bow_nm",
    )
    out = {k: None for k in keys}

    ys, cds = column_cd_profile(rows)
    if len(cds) < 4:
        return out

    out["cd_min"] = float(np.min(cds))
    out["cd_max"] = float(np.max(cds))
    out["cd_mean"] = float(np.mean(cds))
    out["cd_std"] = float(np.std(cds))
    out["cd_range"] = float(np.max(cds) - np.min(cds))

    if out["cd_mean"] > 1e-12:
        out["cd_uniformity"] = 3.0 * out["cd_std"] / out["cd_mean"]

    n = len(cds)
    k = max(1, int(round(n * SHAPE_SEG_FRAC)))

    top_seg = cds[:k]
    bot_seg = cds[-k:]
    mid_seg = cds[k:n - k] if (n - 2 * k) >= 3 else cds

    out["cd_top_seg"] = float(np.mean(top_seg))
    out["cd_bot_seg"] = float(np.mean(bot_seg))
    out["cd_mid_seg"] = float(np.mean(mid_seg))

    mid = out["cd_mid_seg"]
    if mid > 1e-12:
        out["widen_top"] = out["cd_top_seg"] / mid
        out["widen_bot"] = out["cd_bot_seg"] / mid
        out["taper_ratio"] = out["cd_top_seg"] / out["cd_bot_seg"] if out["cd_bot_seg"] > 1e-12 else None

    shoulders = 0.5 * (out["cd_top_seg"] + out["cd_bot_seg"])
    if shoulders > 1e-12:
        out["waist_ratio"] = mid / shoulders

    # 底部展宽台阶：自下往上找到第一个“回到柱身宽度”的行
    body = float(np.median(cds[:max(1, int(n * 0.6))]))
    if body > 1e-12:
        cut = None
        for i in range(n - 1, -1, -1):
            if cds[i] <= body * FOOTING_TRIM_RATIO:
                cut = i
                break
        if cut is not None and cut < n - 1:
            out["footing_step_nm"] = float(cds[-1] - cds[cut])
            out["footing_height_nm"] = float(ys[-1] - ys[cut])

    # 弯曲量：CD(y) 对 y 做线性拟合后的最大残差
    yn = ys - ys.mean()
    denom = float(np.sum(yn ** 2))
    if denom > 1e-15:
        b = float(np.sum(yn * (cds - cds.mean())) / denom)
        a = float(cds.mean() - b * ys.mean())
        out["bow_nm"] = float(np.max(np.abs(cds - (a + b * ys))))

    return out


def classify_shape(metrics):
    """按柱体自身几何判定形貌（不依赖 Target）。

    返回 (形貌标签, 判定依据文字)。
    """
    if not metrics or metrics.get("cd_mean") is None:
        return "Unknown", ""

    top = metrics.get("cd_top_seg")
    mid = metrics.get("cd_mid_seg")
    bot = metrics.get("cd_bot_seg")

    if not (top and mid and bot):
        return "Unknown", ""

    labels = []
    reasons = []

    if metrics.get("widen_top") is not None:
        if metrics["widen_top"] >= SHAPE_TOP_RATIO and (top - mid) >= SHAPE_MIN_DELTA_NM:
            labels.append("T-top")
            reasons.append(f"顶部比中段宽 {top - mid:.2f}nm")

    if metrics.get("waist_ratio") is not None:
        shoulders = 0.5 * (top + bot)
        if metrics["waist_ratio"] <= SHAPE_WAIST_RATIO and (shoulders - mid) >= SHAPE_MIN_DELTA_NM:
            labels.append("Undercut")
            reasons.append(f"腰部比肩部窄 {shoulders - mid:.2f}nm")

    if metrics.get("widen_bot") is not None:
        if metrics["widen_bot"] >= SHAPE_FOOTING_RATIO and (bot - mid) >= SHAPE_MIN_DELTA_NM:
            labels.append("Footing")
            reasons.append(f"底部比中段宽 {bot - mid:.2f}nm")

    if labels:
        return "+".join(labels), "；".join(reasons)

    if metrics.get("cd_uniformity") is not None and metrics["cd_uniformity"] > SHAPE_IRREGULAR_3SIG:
        return "Irregular", f"CD 3σ/均值 = {metrics['cd_uniformity']:.3f}"

    tr = metrics.get("taper_ratio")
    if tr is not None:
        if tr >= SHAPE_TAPER_RATIO:
            return "Taper(上宽下窄)", f"上/下 = {tr:.3f}"
        if tr <= 1.0 / SHAPE_TAPER_RATIO:
            return "Taper(上窄下宽)", f"上/下 = {tr:.3f}"

    return "Vertical(正常)", "上/中/下三段基本一致"


PROFILE_CLASSES = [
    ("SQUARE",  "Ideal / Square",     "W(y) 基本恒定（理想方波）"),
    ("FOOTING", "Footing",            "靠近底部突然变宽"),
    ("UNDERCUT", "Undercut",          "靠近底部突然变窄"),
    ("TAPER_P", "Positive taper",     "从顶到底逐渐变宽"),
    ("TAPER_N", "Negative taper",     "从顶到底逐渐变窄"),
    ("BOWING",  "Bowing",             "中间比上下更宽"),
    ("NECKING", "Necking",            "中间局部变窄"),
    ("T_TOP",   "T-top / Top flare",  "顶部局部变宽"),
    ("ASYM",    "Asymmetric",         "左右边缘变化不对称"),
]

PROFILE_CODE_NAME = {c[0]: c[1] for c in PROFILE_CLASSES}
PROFILE_NAME_CODE = {c[1]: c[0] for c in PROFILE_CLASSES}

# 判定表列顺序（PASS 为总判定，其余为各类形貌）
JUDGE_ITEMS = [
    "PASS", "FOOTING", "UNDERCUT", "T_TOP",
    "TAPER_P", "TAPER_N", "BOWING", "NECKING", "ASYM",
]

JUDGE_HEAD = {
    "PASS": "PASS", "FOOTING": "FOOTING", "UNDERCUT": "UNDERCUT",
    "T_TOP": "T-TOP", "TAPER_P": "TAPER+", "TAPER_N": "TAPER-",
    "BOWING": "BOWING", "NECKING": "NECKING", "ASYM": "ASYM",
}

def resample_w_profile(rows, n=W_PROFILE_SAMPLES):
    """把某根柱的 W(y) 重采样到归一化高度网格，便于求平均曲线。"""
    ys, cds = column_cd_profile(rows)
    if len(ys) < 2:
        return None

    span = float(ys[-1] - ys[0])
    t = (ys - ys[0]) / (span if span > 1e-9 else 1.0)
    grid = np.linspace(0.0, 1.0, n)
    return np.interp(grid, t, cds)


def average_w_profile(list_of_rows, n=W_PROFILE_SAMPLES):
    """所有柱的平均 W(y) 曲线（用于“平均值 vs Target”的归类）。"""
    curves = []
    for rows in list_of_rows:
        w = resample_w_profile(rows, n)
        if w is not None:
            curves.append(w)

    if not curves:
        return None

    return np.mean(np.array(curves, dtype=np.float64), axis=0)


def w_profile_features(w):
    """从（归一化高度上的）W(y) 曲线提取形貌特征量。"""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    if n < 5:
        return None

    k = max(1, int(round(n * SHAPE_SEG_FRAC)))
    kb = max(1, int(round(n * 0.10)))

    top = float(np.mean(w[:k]))
    bot = float(np.mean(w[-k:]))
    shoulders = 0.5 * (top + bot)

    mid_core_start = int(round(n * 0.35))
    mid_core_end = int(round(n * 0.65))
    mid_core = float(np.mean(w[mid_core_start:max(mid_core_end, mid_core_start + 1)]))

    body = float(np.median(w[:max(2, int(round(n * 0.85)))]))

    # 靠近底部的“突然”变化：最下 10% 与紧邻其上的一段比较
    seg_bot = float(np.mean(w[-kb:]))
    seg_above = float(np.mean(w[max(0, n - 6 * kb):max(1, n - kb)]))
    bottom_step = seg_bot - seg_above

    # 靠近顶部的“局部”变化
    seg_top = float(np.mean(w[:kb]))
    seg_below = float(np.mean(w[min(n - 1, kb):min(n, 6 * kb)]))
    top_step = seg_top - seg_below

    trend = bot - top                 # >0 上窄下宽；<0 上宽下窄
    bow = mid_core - shoulders        # >0 中间鼓；<0 中间缩

    return {
        "w": w,
        "cd_mean": float(np.mean(w)),
        "top": top,
        "mid": mid_core,
        "bot": bot,
        "shoulders": shoulders,
        "bottom_step": bottom_step,
        "top_step": top_step,
        "trend": trend,
        "bow": bow,
        "min": float(np.min(w)),
        "max": float(np.max(w)),
        "uniformity": float(3.0 * np.std(w) / float(np.mean(w))) if np.mean(w) > 1e-12 else None,
        "body": body,
    }


def column_asymmetry(rows):
    """左右边缘变化是否不对称。

    返回 (asym_ratio, total_move_nm)：
      ratio 越接近 1 表示变化几乎全发生在单侧（越不对称）。
    """
    if len(rows) < 4:
        return 0.0, 0.0

    left = np.array([float(r[1]) for r in rows], dtype=np.float64)
    right = np.array([float(r[2]) for r in rows], dtype=np.float64)

    dl = abs(float(left[-1] - left[0]))
    dr = abs(float(right[-1] - right[0]))
    total = dl + dr

    if total <= 1e-9:
        return 0.0, 0.0

    return float(abs(dl - dr) / total), float(total)


def profile_target_ratio_score(w, target_cd):
    """返回 ``Profile / Target CD × 100`` 的方向性评分。

    ``W(y)`` 是沿归一化高度采样的线宽曲线，因此曲线面积除以采样
    高度等价于平均 Profile CD。使用有方向的比值能保留“过宽/过窄”
    信息：100 表示平均宽度等于 Target，超过 100 表示 Profile 比
    Target 宽，小于 100 表示 Profile 比 Target 窄。该分数没有人为
    的 100 上限，只有在 Target CD 有效且曲线包含足够多的有限值时才返回。
    """
    if w is None or target_cd is None or len(w) < 5:
        return None

    try:
        target = float(target_cd)
    except Exception:
        return None
    if not math.isfinite(target) or target <= 1e-12:
        return None

    arr = np.asarray(w, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return None

    # 轮廓宽度理论上不应为负；过滤异常值可以避免单个坏点把评分
    # 拉到无意义的负数，同时保留真实的 >100 过宽结果。
    arr = np.maximum(arr, 0.0)
    return 100.0 * float(np.mean(arr)) / target


def classify_and_judge(feat, asym_ratio, asym_nm, target_cd=None, tol=None,
                       use_target=False):
    """按 W(y) 特征归类 + 计算方向性的 Profile/Target (%) 得分。

    返回 (label, score)：
      label : 该柱的唯一形貌（取所有形貌里偏离最严重的那一项；
             若无明显缺陷则为 "SQUARE"）
      score : 平均 Profile CD / Target CD × 100；100 表示等宽，
             >100 表示 Profile 偏宽，<100 表示 Profile 偏窄；
             未启用 Target 或无法计算时为 None
    """
    if feat is None:
        return "SQUARE", None

    cd_mean = feat["cd_mean"]

    if tol is None or tol <= 0:
        if use_target and target_cd:
            tol = target_tolerance(target_cd)
        else:
            tol = max(float(SHAPE_MIN_DELTA_NM), 0.03 * float(cd_mean))

    # 各项“缺陷方向上的偏离量”（nm，>0 表示存在该类缺陷）
    devs = {
        "FOOTING": max(0.0, feat["bottom_step"]),
        "UNDERCUT": max(0.0, -feat["bottom_step"]),
        "T_TOP": max(0.0, feat["top_step"]),
        "BOWING": max(0.0, feat["bow"]),
        "NECKING": max(0.0, -feat["bow"]),
        "TAPER_P": max(0.0, feat["trend"]) if feat["bottom_step"] <= tol else 0.0,
        "TAPER_N": max(0.0, -feat["trend"]) if feat["bottom_step"] >= -tol else 0.0,
        "ASYM": asym_nm if (asym_ratio >= PROFILE_ASYM_RATIO
                            and asym_nm >= PROFILE_ASYM_MIN_NM) else 0.0,
    }

    # 一根柱子只取一项最严重的形貌；没有就 SQUARE
    best_key = max(devs, key=lambda k: devs.get(k, 0.0))
    if devs.get(best_key, 0.0) <= 0.5 * tol:
        label = "SQUARE"
    else:
        label = best_key

    score = None
    if use_target and target_cd and feat.get("w") is not None:
        score = profile_target_ratio_score(feat["w"], float(target_cd))

    return label, score


def _trim_footing_ranges(ranges):
    """底部因 footing/基座突然展宽时，把柱体底边上移到展宽起点。

    纵向逐像素扫描的又一个作用：让“柱体主体”与“底部脚宽”分开，
    避免基座把柱底拉长、污染高度与底部 CD。
    返回 (新 ranges, 是否发生修剪)
    """
    if not FOOTING_TRIM_ENABLE or len(ranges) < 8:
        return ranges, False

    cds = np.array([float(e) - float(s) for _, s, e in ranges], dtype=np.float64)
    n = len(cds)
    body = float(np.median(cds[:max(1, int(n * 0.6))]))
    if body <= 1e-12:
        return ranges, False

    cut = None
    for i in range(n - 1, -1, -1):
        if cds[i] <= body * FOOTING_TRIM_RATIO:
            cut = i
            break

    if cut is None or cut >= n - 1:
        return ranges, False

    if (cut + 1) < len(ranges) * FOOTING_TRIM_MIN_FRAC:
        return ranges, False

    # 必须是“台阶式”突增才修剪：
    # 台阶处单行跳变应占总增量较大比例，避免把渐变的上窄下宽(Taper)误剪。
    total = float(cds[-1] - cds[cut])
    step = float(cds[cut + 1] - cds[cut])

    if total < body * 0.20:
        return ranges, False

    if total > 1e-12 and step < total * 0.40:
        return ranges, False

    return ranges[:cut + 1], True


def _brightness_profile(binary, h, y_lo=None, y_hi=None):
    """只基于二值前景的“亮度证据”横向投影（不含梯度）。

    用作“近柱合并”浅谷检查的依据：避免梯度投影在柱壁之间
    形成的“假谷”干扰合并判断。
    """
    if y_lo is None or y_hi is None or y_hi <= y_lo:
        y_lo = int(h * 0.10)
        y_hi = int(h * 0.80)

    y_lo = max(0, int(y_lo))
    y_hi = min(h - 1, int(y_hi))

    b = np.mean(binary[y_lo:y_hi + 1, :] > 0, axis=0).astype(np.float32)
    b = normalize_01(smooth_1d(b, SMOOTH_KERNEL), percentile=100.0)
    return b


def _gray_profile_for_merge(gray):
    """从候选灰度图生成用于近柱合并的亮度投影。

    几何路径可能使用反相/增强候选，不能直接复用原始二值图；统一对
    候选灰度在主体高度内求横向均值，并归一化为与旧版投影相同的尺度。
    """
    try:
        g = to_gray8(gray)
        h, w = g.shape[:2]
        if h <= 0 or w <= 0:
            return None
        y_top, y_bottom, _ = _main_row_activity(g)
        y_top = max(0, min(int(y_top), h - 1))
        y_bottom = max(y_top + 1, min(int(y_bottom), h - 1))
        prof = np.mean(g[y_top:y_bottom + 1, :], axis=0).astype(np.float32)
        return normalize_01(smooth_1d(prof, SMOOTH_KERNEL), percentile=100.0)
    except Exception:
        return None


def merge_close_columns(columns, max_gap=None, y_overlap_min=None,
                        structure_profile=None, shallow_ratio=None, shallow_gap_max=None,
                        shallow_profile=None, height_ratio_max=None,
                        bottom_diff_max=None, deep_crack_ratio=None):
    """相邻柱体过近、浅谷、或“同一根柱被中缝错拆成两根”时自动合并。

    合并条件（满足任一即可）：
      ① 几何条件：gap < max_gap 且 y 方向有 y_overlap_min 像素行重叠；
      ② 浅谷条件：shallow_profile（默认纯亮度投影）中两柱间最小值
                    ≥ 浅谷比 × 两柱峰值平均，且 gap ≤ shallow_gap_max；
      ③ 深裂条件：两柱间亮度投影谷深 ≤ deep_crack_ratio × 峰值平均，
                    且 gap ≤ shallow_gap_max、高度比 ≤ height_ratio_max、
                    底边 y 差 ≤ bottom_diff_max —— 用于把一根被中间
                    竖缝/弱区错拆成两根的柱子重新合并。

    返回 (new_columns, merged_count)
    """
    if max_gap is None:
        max_gap = MERGE_GAP_PX
    if y_overlap_min is None:
        y_overlap_min = MERGE_Y_OVERLAP_MIN
    if shallow_ratio is None:
        shallow_ratio = MERGE_VALLEY_RATIO
    if shallow_gap_max is None:
        shallow_gap_max = MERGE_VALLEY_GAP_MAX
    if height_ratio_max is None:
        height_ratio_max = MERGE_HEIGHT_RATIO_MAX
    if bottom_diff_max is None:
        bottom_diff_max = MERGE_BOTTOM_DIFF_MAX
    if deep_crack_ratio is None:
        deep_crack_ratio = MERGE_DEEP_CRACK_RATIO
    if len(columns) < 2:
        return columns, 0

    # 默认用纯亮度投影判断浅谷/深裂（不夹带梯度投影的“假谷”）
    prof = shallow_profile if shallow_profile is not None else structure_profile
    if prof is not None and hasattr(prof, "ndim") and prof.ndim > 1:
        prof = np.mean(prof, axis=0)

    def _col_peak(col):
        if prof is None:
            return 0.0
        a = max(0, min(int(col["left"]), len(prof) - 1))
        b = max(a + 1, min(int(col["right"]) + 1, len(prof)))
        return float(np.max(prof[a:b])) if b > a else 0.0

    def _basic_ok(a, b):
        """高度比与底边差检查，浅谷/深裂都需要。"""
        ha = max(1, int(a["y_bottom"]) - int(a["y_top"]))
        hb = max(1, int(b["y_bottom"]) - int(b["y_top"]))
        if max(ha, hb) / min(ha, hb) > height_ratio_max:
            return False
        if abs(int(a["y_bottom"]) - int(b["y_bottom"])) > bottom_diff_max:
            return False
        return True

    def _pair_width(a, b):
        """返回两候选中较窄的逐行 CD，作为裂缝/浅谷的尺度约束。"""
        wa = _column_width_px(a)
        wb = _column_width_px(b)
        vals = [float(v) for v in (wa, wb)
                if math.isfinite(float(v)) and float(v) > 0]
        return min(vals) if vals else 1.0

    def _shallow_gap_limit(a, b):
        """浅谷合并只允许发生在相对柱宽较小的间隙内。

        原先固定允许到 60 px；当真实柱间距落在该范围内时，平滑后的
        背景基线会被误判成“浅谷”，把整排独立柱连成一根。动态尺度仍
        能覆盖同一实体被纹理/弱缝分开的情况，但不会把半个柱宽以外的
        正常间距吞掉。
        """
        width = _pair_width(a, b)
        return min(float(shallow_gap_max), max(4.0, min(24.0, 0.30 * width)))

    def _deep_gap_limit(a, b):
        """深裂只接受真正窄缝；它不是浅谷规则的 60 px 别名。"""
        width = _pair_width(a, b)
        return min(float(shallow_gap_max), max(2.0, min(12.0, 0.18 * width)))

    def _valley_ratio(a, b):
        """返回两柱之间亮度投影谷值 / 两侧峰值平均（None 表示无法计算）。"""
        if prof is None:
            return None
        gap = int(b["left"]) - int(a["right"])
        if gap <= 0:
            return None
        x0 = max(0, int(a["right"]))
        x1 = min(len(prof), int(b["left"]) + 1)
        if x1 <= x0:
            return None
        valley = float(np.min(prof[x0:x1]))
        peak_a = _col_peak(a)
        peak_b = _col_peak(b)
        if peak_a + peak_b <= 1e-12:
            return None
        avg_peak = 0.5 * (peak_a + peak_b)
        return valley / avg_peak if avg_peak > 1e-12 else None

    def _shallow(a, b):
        if prof is None or shallow_ratio <= 0:
            return False
        gap = int(b["left"]) - int(a["right"])
        if gap <= 0 or gap > _shallow_gap_limit(a, b):
            return False
        if not _basic_ok(a, b):
            return False
        ratio = _valley_ratio(a, b)
        return ratio is not None and ratio >= shallow_ratio

    def _deep_crack(a, b):
        """深而窄的谷 + 高度/底边对齐 → 同一根柱的中缝被错拆。

        注意：两柱之间完全黑暗（valley=0）通常是真正的间隙，不是中缝；
        这里要求 gap 内有少量亮度（>0）才视为“中缝”。
        """
        if prof is None or deep_crack_ratio <= 0:
            return False
        gap = int(b["left"]) - int(a["right"])
        if gap <= 0 or gap > _deep_gap_limit(a, b):
            return False
        if not _basic_ok(a, b):
            return False
        ratio = _valley_ratio(a, b)
        if ratio is None:
            return False
        # 完全黑暗的 gap 不合并（那是真实间隙）
        if ratio <= 1e-6:
            return False
        return ratio <= deep_crack_ratio

    merged_count = 0
    changed = True
    while changed:
        changed = False
        new = []
        i = 0
        while i < len(columns):
            if i + 1 < len(columns):
                a = columns[i]
                b = columns[i + 1]
                gap = int(b["left"]) - int(a["right"])
                y_top_a, y_bot_a = int(a["y_top"]), int(a["y_bottom"])
                y_top_b, y_bot_b = int(b["y_top"]), int(b["y_bottom"])
                y_overlap = min(y_bot_a, y_bot_b) - max(y_top_a, y_top_b)

                geom_merge = gap < max_gap and y_overlap >= y_overlap_min
                shallow_merge = _shallow(a, b)
                crack_merge = _deep_crack(a, b)

                if geom_merge or shallow_merge or crack_merge:
                    # 合并：同一 y 上的两个区间必须取并集；旧实现只保留
                    # 先出现的区间，导致合并后的 bbox 跨越两柱而 ranges/
                    # contour 仍只有左柱，后续 CD/Judge 会被静默污染。
                    by_y = {}
                    for yy, ss, ee in a["ranges"] + b["ranges"]:
                        yy = int(yy)
                        ss, ee = int(min(ss, ee)), int(max(ss, ee))
                        if yy in by_y:
                            by_y[yy][0] = min(by_y[yy][0], ss)
                            by_y[yy][1] = max(by_y[yy][1], ee)
                        else:
                            by_y[yy] = [ss, ee]
                    ranges = [
                        (yy, se[0], se[1])
                        for yy, se in sorted(by_y.items())
                    ]

                    merged = dict(a)
                    merged["ranges"] = ranges
                    merged["left"] = min(int(a["left"]), int(b["left"]))
                    merged["right"] = max(int(a["right"]), int(b["right"]))
                    merged["y_top"] = min(y_top_a, y_top_b)
                    merged["y_bottom"] = max(y_bot_a, y_bot_b)
                    merged["center_x"] = (merged["left"] + merged["right"]) // 2
                    merged["contour"] = build_contour(ranges)
                    merged["merged"] = True
                    merged["merged_count"] = (a.get("merged_count", 1) + 1)
                    merged["bridge_warning"] = False
                    new.append(merged)
                    i += 2
                    changed = True
                    merged_count += 1
                    continue

            new.append(columns[i])
            i += 1
        columns = new

    # 重新分配 label / color
    for idx, c in enumerate(columns):
        c["label"] = idx + 1
        c["color"] = PR_COLORS[(c["label"] - 1) % len(PR_COLORS)]

    return columns, merged_count


def smart_refine_columns(columns):
    """基于纵向逐像素扫描的智能后处理。

    1) footing 台阶修剪：底边回到柱体主体；
    2) 粘连检测：相邻柱在重叠高度上的水平间隙过小 → bridge_warning。
    """
    for col in columns:
        ranges, trimmed = _trim_footing_ranges(col["ranges"])
        if trimmed:
            col["ranges"] = ranges
            col["y_bottom"] = int(ranges[-1][0])
            col["contour"] = build_contour(ranges)
        col["footing_trimmed"] = bool(trimmed)
        col["bridge_warning"] = False

    for a, b in zip(columns, columns[1:]):
        ra = {y: (s, e) for y, s, e in a["ranges"]}
        rb = {y: (s, e) for y, s, e in b["ranges"]}
        common = sorted(set(ra) & set(rb))
        if not common:
            continue
        gap = min(rb[y][0] - ra[y][1] for y in common)
        if gap <= BRIDGE_GAP_PX:
            a["bridge_warning"] = True
            b["bridge_warning"] = True

    return columns


def target_tolerance(target_cd=None):
    cd = float(TARGET_CD_NM if target_cd is None else target_cd)
    return max(float(TARGET_TOL_NM), abs(cd) * float(TARGET_TOL_PCT) / 100.0)


def build_target_report(data, target=None):
    """把每根柱与“理想方波”（CD(y) 恒等于 Target）对比。

    判定思路：
      · 理想方波：上/中/下三段 CD 都等于 Target；
      · 中段明显细于 Target → Undercut（腰部被钻进去）；
      · 底部明显宽于 Target → Footing（底部站裙/脚宽）；
      · 顶部明显宽于 Target → T-top（蘑菇头）；
      · 三段都在容差内 → Normal，判定 PASS。
    未启用 Target 时，只输出按自身几何判定的形貌，不做 PASS/FAIL。
    """
    if target is None:
        target = {
            "enable": bool(TARGET_ENABLE),
            "cd": float(TARGET_CD_NM),
            "tol_nm": float(TARGET_TOL_NM),
            "tol_pct": float(TARGET_TOL_PCT),
            "height": float(TARGET_HEIGHT_NM),
        }

    use = bool(target.get("enable"))
    cd_t = float(target.get("cd", TARGET_CD_NM))
    tol = max(
        float(target.get("tol_nm", TARGET_TOL_NM)),
        abs(cd_t) * float(target.get("tol_pct", TARGET_TOL_PCT)) / 100.0,
    )
    h_t = float(target.get("height", 0.0) or 0.0)

    report_rows = []
    shape_counts = {}
    profile_counts = {}
    n_pass = 0
    n_fail = 0
    deltas = []
    cd_means = []
    judge_scores = []
    similarities = []

    cols = data.get("cols", [])
    per_col = data.get("per_col", [])
    shapes = data.get("shapes", [])
    reasons = data.get("shape_reasons", [])

    feats = data.get("profile_feat", [])
    asyms = data.get("profile_asym", [])
    stored_labels = data.get("profile_labels", [])

    for i, label in enumerate(cols):
        st = per_col[i] if i < len(per_col) else {}
        geo_shape = shapes[i] if i < len(shapes) else "Unknown"
        geo_why = reasons[i] if i < len(reasons) else ""

        # W(y) 单形貌：启用 Target 时按 Target 判定，否则用识别时的自身几何归类
        ratio, move = asyms[i] if i < len(asyms) else (0.0, 0.0)
        feat = feats[i] if i < len(feats) else None

        if use:
            prof_label, prof_score = classify_and_judge(
                feat, ratio, move, cd_t, tol, use_target=True
            )
        else:
            prof_label = (
                stored_labels[i] if i < len(stored_labels) else "SQUARE"
            )
            _, prof_score = classify_and_judge(
                feat, ratio, move, None, None, use_target=False
            )

        if prof_score is not None:
            try:
                if math.isfinite(float(prof_score)):
                    judge_scores.append(float(prof_score))
            except Exception:
                pass

        cd_top = st.get("cd_top")
        cd_mid = st.get("cd_mid")
        cd_bot = st.get("cd_bot")
        height = st.get("height")

        vals = [v for v in (cd_top, cd_mid, cd_bot)
                if isinstance(v, (int, float)) and math.isfinite(float(v))]
        cd_mean = sum(vals) / len(vals) if vals else None

        delta = None
        judge = "—"
        shape = geo_shape
        why = geo_why
        similarity = None

        if cd_mean is not None:
            cd_means.append(cd_mean)

        if use and cd_mean is not None:
            delta = cd_mean - cd_t
            deltas.append(abs(delta))

            devs = [abs(v - cd_t) for v in vals] if vals else []
            if devs and cd_t > 1e-12:
                similarity = max(0.0, 1.0 - (sum(devs) / len(devs)) / cd_t)
                if math.isfinite(float(similarity)):
                    similarities.append(float(similarity))

            flags = []
            if cd_mid is not None and (cd_t - cd_mid) > tol:
                flags.append("Undercut")
            if cd_bot is not None and (cd_bot - cd_t) > tol:
                flags.append("Footing")
            if cd_top is not None and (cd_top - cd_t) > tol:
                flags.append("T-top")

            shape = "+".join(flags) if flags else "Normal"
            why = f"相对 Target {cd_t:g}nm：上{_signed(cd_top, cd_t)} / " \
                  f"中{_signed(cd_mid, cd_t)} / 下{_signed(cd_bot, cd_t)} nm"

            ok = all(
                (v is None) or (abs(v - cd_t) <= tol)
                for v in vals
            )
            if h_t > 0 and height is not None:
                ok = ok and abs(float(height) - h_t) <= max(tol, h_t * 0.05)

            judge = "PASS" if ok else "FAIL"
            if ok:
                n_pass += 1
            else:
                n_fail += 1

        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        profile_counts[prof_label] = profile_counts.get(prof_label, 0) + 1

        report_rows.append({
            "label": label,
            "cd_top": cd_top,
            "cd_mid": cd_mid,
            "cd_bot": cd_bot,
            "cd_mean": cd_mean,
            "height": height,
            "delta": delta,
            "judge": judge,
            "shape": shape,
            "geo": geo_shape,
            "profile": prof_label,
            "score": prof_score,
            # 明确的别名，方便外部脚本读取而不必猜测 score 的单位。
            "profile_target_pct": prof_score,
            "profile_min": (float(feat.get("min"))
                            if isinstance(feat, dict)
                            and feat.get("min") is not None else None),
            "profile_max": (float(feat.get("max"))
                            if isinstance(feat, dict)
                            and feat.get("max") is not None else None),
            "profile_uniformity": (float(feat.get("uniformity"))
                                   if isinstance(feat, dict)
                                   and feat.get("uniformity") is not None else None),
            "similarity": similarity,
            "reason": why,
        })

    summary = {
        "n": len(cols),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pass_rate": (n_pass / len(cols)) if cols else None,
        "max_abs_delta": max(deltas) if deltas else None,
        "mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "similarity_mean": (sum(similarities) / len(similarities)
                            if similarities else None),
        "cd_mean_avg": (sum(cd_means) / len(cd_means)) if cd_means else None,
        "cd_uniformity": None,
        "shape_counts": shape_counts,
        "profile_counts": profile_counts,
        "judge_score_mean": (float(np.mean(judge_scores))
                              if judge_scores else None),
        "judge_score_min": (min(judge_scores) if judge_scores else None),
        "judge_score_max": (max(judge_scores) if judge_scores else None),
        "profile_wider_count": (sum(v > 100.0 + 1e-9 for v in judge_scores)
                                 if use else 0),
        "profile_narrower_count": (sum(v < 100.0 - 1e-9 for v in judge_scores)
                                    if use else 0),
        "profile_equal_count": (sum(abs(v - 100.0) <= 1e-9 for v in judge_scores)
                                 if use else 0),
    }

    if len(cd_means) >= 2 and summary["cd_mean_avg"]:
        m = summary["cd_mean_avg"]
        sd = float(np.std(np.array(cd_means, dtype=np.float64)))
        summary["cd_uniformity"] = 3.0 * sd / m if m > 1e-12 else None

    # 平均值曲线（所有柱的平均 W(y)）与 Target 的对比
    avg_info = data.get("avg_profile", {}) or {}
    avg_feat = avg_info.get("feat")
    a_ratio, a_move = avg_info.get("asym", (0.0, 0.0))

    if use:
        avg_label, avg_score = classify_and_judge(
            avg_feat, a_ratio, a_move, cd_t, tol, use_target=True
        )
    else:
        avg_label = avg_info.get("label", "SQUARE")
        _, avg_score = classify_and_judge(
            avg_feat, a_ratio, a_move, None, None, use_target=False
        )

    avg_w = data.get("avg_w")
    avg_similarity = None
    if use and avg_w is not None and len(avg_w) and cd_t > 1e-12:
        _aw = np.asarray(avg_w, dtype=np.float64)
        _aw = _aw[np.isfinite(_aw)]
        if len(_aw):
            avg_similarity = max(
                0.0,
                1.0 - float(np.mean(np.abs(_aw - cd_t))) / cd_t,
            )
    avg_row = {
        "label": "平均",
        "profile": avg_label,
        "score": avg_score,
        "profile_target_pct": avg_score,
        "cd_mean": (float(np.mean(avg_w)) if avg_w is not None
                    and len(avg_w) else None),
        "delta": (float(np.mean(avg_w)) - cd_t
                  if (avg_w is not None and len(avg_w) and use) else None),
        "similarity": avg_similarity,
        "profile_min": (float(np.min(avg_w)) if avg_w is not None and len(avg_w)
                        else None),
        "profile_max": (float(np.max(avg_w)) if avg_w is not None and len(avg_w)
                        else None),
    }

    return {
        "target": target,
        "tol": tol,
        "rows": report_rows,
        "summary": summary,
        "avg": avg_row,
    }


def _signed(value, ref):
    if value is None:
        return "-"
    return f"{value - ref:+.2f}"


def write_target_file(path, report):
    """导出 Target 方波对比 / 形貌判定表。

    Judge 分数采用方向性的 ``Profile/Target (%)``：100 表示平均
    Profile CD 与 Target 相等，超过 100 表示 Profile 偏宽，低于 100
    表示 Profile 偏窄。
    """
    target = report.get("target", {})
    s = report.get("summary", {})
    tol = report.get("tol", 0.0)

    def g(v, nd=3):
        if v is None:
            return "-"
        try:
            f = float(v)
        except Exception:
            return "-"
        return f"{f:.{nd}f}" if math.isfinite(f) else "-"

    head = [
        "标准 Target（理想方波）对比与形貌判定",
        "",
        f"启用状态      : {'启用' if target.get('enable') else '未启用（仅输出自身几何形貌）'}",
        f"目标 CD       : {float(target.get('cd', 0)):.3f} nm",
        f"容差          : ±{tol:.3f} nm "
        f"(绝对值 {float(target.get('tol_nm', 0)):g} 与 "
        f"相对 {float(target.get('tol_pct', 0)):g}% 取大者)",
        f"目标高度      : {'不判断' if not float(target.get('height', 0) or 0) else format(float(target['height']), '.3f') + ' nm'}",
        "Judge 分数定义 : Profile/Target = mean(W(y)) / Target CD × 100 (%)",
        "Judge 分数解读 : 100=等宽；>100=Profile 比 Target 宽；<100=Profile 比 Target 窄",
        ("Judge 状态      : 未启用 Target，逐柱分数显示 '-'"
         if not target.get("enable") else
        "Judge 状态      : 已启用，分数按每根柱的 W(y) 曲线计算"),
        "",
        "—— 汇总 ——",
        f"柱子数量      : {s.get('n', 0)}",
    ]

    meta_rows = _detection_meta_rows(report.get("detection"))
    if meta_rows:
        head += ["", "—— 识别参数与候选 ——"]
        head.extend(
            f"{row[0]} : {row[1]}" for row in meta_rows
            if row and any(str(v) for v in row)
        )

    if target.get("enable"):
        rate = s.get("pass_rate")
        head += [
            f"PASS / FAIL   : {s.get('n_pass', 0)} / {s.get('n_fail', 0)}"
            + (f"  (通过率 {rate * 100:.1f}%)" if rate is not None else ""),
            f"最大 |ΔCD|    : {g(s.get('max_abs_delta'))} nm",
            f"平均 |ΔCD|    : {g(s.get('mean_abs_delta'))} nm",
            f"方波相似度均值 : {g(s.get('similarity_mean'), 4)} (0~1，越接近 1 越像)",
            f"Judge 分数范围 : {g(s.get('judge_score_min'), 2)} ~ {g(s.get('judge_score_max'), 2)} %",
            f"Profile > Target: {s.get('profile_wider_count', 0)} 根；"
            f"Profile < Target: {s.get('profile_narrower_count', 0)} 根；"
            f"约等于: {s.get('profile_equal_count', 0)} 根",
        ]

    head += [
        f"CD 均值(逐柱三段平均) : {g(s.get('cd_mean_avg'))} nm",
        f"平均 W(y) CD       : {g((report.get('avg') or {}).get('cd_mean'))} nm",
        f"CD 片内均匀性 : {g(s.get('cd_uniformity'))} (3σ/均值，越小越一致)",
    ]

    counts = s.get("shape_counts", {}) or {}
    if counts:
        head.append("形貌分布      : " + "，".join(
            f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        ))

    head += ["", "—— 逐柱判定 ——"]

    # 顺序：对象(全部柱平均在最前) + CD上中下 + 平均CD + Δ + 判定 + Profile + 得分
    rows = [[
        "对象", "CD上(nm)", "CD中(nm)", "CD下(nm)", "平均CD(nm)",
        "Δvs Target(nm)", "判定", "Profile(归类)",
        "Profile/Target (%)", "方波相似度", "Profile范围(nm)", "自身几何"
    ]]

    avg = report.get("avg")
    if avg:
        rows.append([
            "全部柱平均",
            "-", "-", "-",
            _num(avg.get("cd_mean")),
            _num(avg.get("delta"), signed=True),
            "—",
            avg.get("profile", "SQUARE"),
            _num(avg.get("score"), signed=False),
            _num(avg.get("similarity"), signed=False),
            (f"{g(avg.get('profile_min'), 2)}~{g(avg.get('profile_max'), 2)}"
             if avg.get("profile_min") is not None and avg.get("profile_max") is not None
             else "-"),
            "—",
        ])

    for r in report.get("rows", []):
        rows.append([
            r["label"],
            _num(r["cd_top"]), _num(r["cd_mid"]), _num(r["cd_bot"]),
            _num(r["cd_mean"]),
            _num(r["delta"], signed=True),
            r["judge"],
            r.get("profile", "SQUARE"),
            _num(r.get("score"), signed=False),
            _num(r.get("similarity"), signed=False),
            (f"{g(r.get('profile_min'), 2)}~{g(r.get('profile_max'), 2)}"
             if r.get("profile_min") is not None and r.get("profile_max") is not None
             else "-"),
            r.get("geo", "-"),
        ])

    # ---- W(y) 单形貌 + Profile/Target 方向性得分的小计 ----
    pc = s.get("profile_counts", {}) or {}
    if pc:
        head.append("")
        head.append("形貌分布：" + "，".join(
            f"{PROFILE_CODE_NAME.get(k, k)}×{v}"
            for k, v in sorted(pc.items(), key=lambda kv: -kv[1])
        ))

    # 判定依据附在表后，便于追溯每根柱为什么被判成该形貌
    detail = ["", "—— 判定依据 ——"]
    for r in report.get("rows", []):
        if r.get("reason"):
            detail.append(f"{r['label']}: {r['reason']}  →  {r['shape']}")

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
            cells.append(text + pad if i == 0 else pad + text)
        lines.append("  ".join(cells).rstrip())

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + [""] + lines + detail) + "\n")


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
        "warning": str(aux.get("detection_warning", "") or ""),
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
        ["识别置信度", confidence + f"；重试阈值 {float(AUTO_RETRY_MIN_CONFIDENCE):g}"],
        ["候选尝试次数", attempts],
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
    """
    直接从原始像素轮廓转 float nm 后统计。
    不经过 boundaries_nm.txt，因此没有文本舍入误差。
    """
    col_labels = []
    per_col = []
    shapes = []
    shape_reasons = []
    shape_metrics = []
    flags = []
    rows_nm_all = []

    for col in columns:
        rows_nm = [
            (
                float(y) * nm_per_px,
                float(xs) * nm_per_px,
                float(xe) * nm_per_px,
            )
            for y, xs, xe in col["ranges"]
        ]
        if not rows_nm:
            continue

        col_labels.append(f"PR{col['label']}")
        per_col.append(analyze_pr_column(rows_nm))
        rows_nm_all.append(rows_nm)

        metrics = analyze_shape_metrics(rows_nm)
        shape_metrics.append(metrics)

        if SHAPE_ENABLE:
            label, why = classify_shape(metrics)
        else:
            label, why = "未启用", ""
        shapes.append(label)
        shape_reasons.append(why)

        notes = []
        if col.get("footing_trimmed"):
            notes.append("已按 footing 台阶修剪底边")
        if col.get("bridge_warning"):
            notes.append("与邻柱疑似粘连")
        flags.append("；".join(notes))

    rows = []
    means = []

    for key, _desc in STAT_ROWS:
        vals = [d[key] for d in per_col]
        rows.append(vals)
        means.append(_mean_finite(vals))

    # ---- W(y) 形貌归类：每根柱一个形貌 + Profile/Target 方向性得分 ----
    profile_labels = []   # 每根柱的唯一形貌字符串
    profile_scores = []   # 每根柱 mean(Profile CD)/Target CD × 100 (%)
    profile_feat = []
    profile_asym = []
    asym_flags = []

    for rows_nm in rows_nm_all:
        w = resample_w_profile(rows_nm)
        feat = w_profile_features(w) if w is not None else None
        ratio, move = column_asymmetry(rows_nm)
        profile_feat.append(feat)
        profile_asym.append((ratio, move))
        asym_flags.append(ratio >= PROFILE_ASYM_RATIO and move >= PROFILE_ASYM_MIN_NM)
        label, score = classify_and_judge(
            feat, ratio, move, TARGET_CD_NM, target_tolerance(TARGET_CD_NM),
            use_target=bool(TARGET_ENABLE)
        )
        profile_labels.append(label)
        profile_scores.append(score)

    avg_w = average_w_profile(rows_nm_all)
    avg_feat = w_profile_features(avg_w) if avg_w is not None else None
    avg_ratio = (sum(1 for f in asym_flags if f) / len(asym_flags)) if asym_flags else 0.0
    avg_move = float(np.mean([
        column_asymmetry(r)[1] for r in rows_nm_all
    ])) if rows_nm_all else 0.0

    avg_label, avg_score = classify_and_judge(
        avg_feat,
        avg_ratio,
        avg_move,
        TARGET_CD_NM,
        target_tolerance(TARGET_CD_NM),
        use_target=bool(TARGET_ENABLE)
    )

    return {
        "desc": [desc for _, desc in STAT_ROWS],
        "cols": col_labels,
        "rows": rows,
        "means": means,
        "per_col": per_col,
        "shapes": shapes,
        "shape_reasons": shape_reasons,
        "shape_metrics": shape_metrics,
        "flags": flags,
        "profile_labels": profile_labels,
        "profile_scores": profile_scores,
        "profile_feat": profile_feat,
        "profile_asym": profile_asym,
        "avg_w": avg_w,
        "avg_profile": {
            "label": avg_label,
            "score": avg_score,
            "feat": avg_feat,
            "asym": (avg_ratio, avg_move),
        },
    }


def write_stats_file(out_txt_path, data, detection_meta=None):
    def fmt(v):
        if v is None:
            return "-"
        try:
            if not math.isfinite(float(v)):
                return "-"
        except Exception:
            return "-"
        return f"{float(v):.3f}"

    # 列顺序：统计项 + 全部柱平均 + PR1..PRn （平均放在 PR1 前面）
    header = ["统计项", "全部柱平均"] + list(data["cols"])
    lines = [header]

    for desc, vals, mean in zip(data["desc"], data["rows"], data["means"]):
        lines.append(
            [desc] + [fmt(mean)] + [fmt(v) for v in vals]
        )

    # ---- 最后两行：W(y) 形貌归类 + 方向性的 Profile/Target 得分 ----
    labels = data.get("profile_labels") or []
    scores = data.get("profile_scores") or []
    # 若调用方已经按一个自定义 Target 重算过报告，优先采用报告里的
    # Profile/Judge，避免 analyze_columns 当时使用的旧全局 Target 设置
    # 把导出文件写成过时的 “-” 或旧分数。
    target_report = data.get("target")
    if isinstance(target_report, dict) and target_report.get("rows"):
        report_rows = target_report.get("rows") or []
        labels = [r.get("profile", "SQUARE") for r in report_rows]
        scores = [r.get("score") for r in report_rows]

    if labels and data["cols"]:
        avg_info = data.get("avg_profile") or {}
        if isinstance(target_report, dict) and target_report.get("avg"):
            avg_info = target_report.get("avg") or avg_info
        # Target 报告的 avg.label 是行标题“平均”，真正的形貌类别在
        # avg.profile；若没有 Target 报告则兼容 analyze_columns 的旧字段。
        avg_label = avg_info.get("profile", avg_info.get("label", "SQUARE"))
        avg_score = avg_info.get("score")

        def fmt_score(s):
            if s is None:
                return "-"
            try:
                return f"{float(s):.2f}"
            except Exception:
                return "-"

        row_profile = ["Profile", avg_label] + list(labels)
        row_judge = ["Judge", fmt_score(avg_score)] + [
            fmt_score(s) for s in scores
        ]

        lines.append(row_profile)
        lines.append(row_judge)

    # 在表格末尾写清楚 Judge 的单位和解释，避免只看到一个数字时
    # 误以为它是限制在 0~100 的“相似度”。
    target_report_obj = data.get("target")
    report_effective_tol = (
        target_report_obj.get("tol")
        if isinstance(target_report_obj, dict)
        and isinstance(target_report_obj.get("tol"), (int, float))
        else None
    )
    target = target_report_obj or {
        "enable": bool(TARGET_ENABLE),
        "cd": float(TARGET_CD_NM),
        "tol_nm": float(TARGET_TOL_NM),
        "tol_pct": float(TARGET_TOL_PCT),
        "height": float(TARGET_HEIGHT_NM),
    }
    # build_target_report() 返回的是完整 report，Target 配置位于
    # report["target"]；也兼容调用方直接传入配置 dict 的旧方式。
    if isinstance(target, dict) and isinstance(target.get("target"), dict):
        target = target.get("target") or target
    target_cd = target.get("cd", TARGET_CD_NM)
    target_enabled = bool(target.get("enable"))
    try:
        target_cd_text = f"{float(target_cd):.3f} nm"
    except Exception:
        target_cd_text = "-"
    try:
        tol_nm = float(target.get("tol_nm", TARGET_TOL_NM))
    except Exception:
        tol_nm = float(TARGET_TOL_NM)
    try:
        tol_pct = float(target.get("tol_pct", TARGET_TOL_PCT))
    except Exception:
        tol_pct = float(TARGET_TOL_PCT)
    effective_tol = (
        float(report_effective_tol)
        if report_effective_tol is not None
        else max(tol_nm, abs(float(target_cd)) * tol_pct / 100.0)
    )
    try:
        target_height = float(target.get("height", TARGET_HEIGHT_NM) or 0.0)
    except Exception:
        target_height = float(TARGET_HEIGHT_NM)
    lines.extend([
        ["Judge 定义", "Profile/Target = mean(W(y)) / Target CD × 100 (%)"],
        ["Judge 解读", "100=等宽；>100=Profile 比 Target 宽；<100=Profile 比 Target 窄"],
        ["Target 状态", "启用" if target_enabled else "未启用（Judge 显示 -）"],
        ["Target CD", target_cd_text],
        ["Target 容差", f"±{effective_tol:.3f} nm（绝对 ±{tol_nm:g} nm；相对 ±{tol_pct:g}% 取大者）"],
        ["Target 高度", "不判断" if target_height <= 0 else f"{target_height:.3f} nm"],
    ])

    # 识别元数据与统计同文件落盘，方便复核“本次结果是如何得到的”。
    # 兼容旧调用：没有显式参数时也尝试读取 data['detection']。
    meta = detection_meta if isinstance(detection_meta, dict) else data.get("detection")
    meta_rows = _detection_meta_rows(meta)
    if meta_rows:
        lines.append(["", ""])
        lines.append(["—— 识别参数与候选 ——", ""])
        lines.extend(meta_rows)

    write_aligned_table(out_txt_path, lines, left_first_text=True)


# =====================================================================
# 输出
# =====================================================================

def _output_dir_for(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    base = INPUT_DIR
    out_dir = os.path.join(base, stem) if OUTPUT_SUBDIR else base
    os.makedirs(out_dir, exist_ok=True)
    return stem, out_dir


def export_detection_outputs(
    image_path,
    img,
    columns,
    structure_profile,
    tag,
    detection_meta=None,
):
    """
    tag="_tool" / "_user"
    返回 (px_main, stats_data, stats_path)
    """
    stem, out_dir = _output_dir_for(image_path)
    nm_per_px = SCALE_NM / SCALE_PIXELS

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
    # 标准 Target（理想方波）对比 + 形貌判定
    stats_data["target"] = build_target_report(stats_data)
    stats_data["target"]["detection"] = meta
    # 先挂上完整 Target 配置再写 stats，输出文件中的 Judge 定义、
    # 目标 CD 与启用状态始终和 target_check.txt 保持一致。
    write_stats_file(stats_path, stats_data, detection_meta=meta)
    target_path = os.path.join(out_dir, f"{prefix}_target_check.txt")
    try:
        write_target_file(target_path, stats_data["target"])
    except Exception:
        target_path = None
    stats_data["target_path"] = target_path

    return px_main, stats_data, stats_path


def process_image(image_path, on_image=None):
    img, binary, aux = preprocess_image(image_path)
    columns, structure_profile = _detect_columns(img, binary, aux, roi_mode=False)
    detection_meta = _detection_meta_from_aux(aux, columns)

    stem, out_dir = _output_dir_for(image_path)

    if DEBUG_OUTPUT:
        imwrite_unicode(os.path.join(out_dir, f"{stem}_debug_binary.png"), binary)

        gx8 = np.clip(aux["grad_x"] * 255.0, 0, 255).astype(np.uint8)
        gy8 = np.clip(aux["grad_y"] * 255.0, 0, 255).astype(np.uint8)
        imwrite_unicode(os.path.join(out_dir, f"{stem}_debug_grad_x.png"), gx8)
        imwrite_unicode(os.path.join(out_dir, f"{stem}_debug_grad_y.png"), gy8)
        imwrite_unicode(os.path.join(out_dir, f"{stem}_debug_enhanced.png"), aux["enhanced"])

    px_main, stats_data, stats_path = export_detection_outputs(
        image_path,
        img,
        columns,
        structure_profile,
        "_tool",
        detection_meta=detection_meta,
    )

    payload = {
        "path": image_path,
        "img": img,
        "columns": columns,
        "structure_profile": structure_profile,
        "tag": "_tool",
        "stats": stats_data,
        "stats_path": stats_path,
        "quality": aux["quality"],
        "threshold_mode": aux["threshold_mode"],
        "detection_mode": aux.get("detection_mode", "unknown"),
        "detection_variant": aux.get("detection_variant", "unknown"),
        "detection_confidence": aux.get("detection_confidence", -1.0),
        "detection_attempts": aux.get("detection_attempts", 0),
        "detection_warning": aux.get("detection_warning", ""),
        "cd_hint_px": aux.get("cd_hint_px"),
        "cd_range_px": aux.get("cd_range_px"),
        "cd_observed_px": detection_meta.get("cd_observed_px"),
        "detection": detection_meta,
        "binary_fraction": aux["binary_fraction"],
    }

    if on_image is not None:
        on_image(img, px_main, os.path.basename(image_path), payload)

    return payload


def process_image_roi(image_path, roi, on_image=None):
    img, binary, aux = preprocess_image(image_path)
    h, w = img.shape[:2]

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

    sub_img = img[y0:y1 + 1, x0:x1 + 1].copy()
    sub_bin = binary[y0:y1 + 1, x0:x1 + 1].copy()

    sub_aux = {
        "gray": cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY),
        "enhanced": aux["enhanced"][y0:y1 + 1, x0:x1 + 1].copy(),
        "blur": aux["blur"][y0:y1 + 1, x0:x1 + 1].copy(),
        "grad_x": aux["grad_x"][y0:y1 + 1, x0:x1 + 1].copy(),
        "grad_y": aux["grad_y"][y0:y1 + 1, x0:x1 + 1].copy(),
    }

    columns, local_profile = _detect_columns(
        sub_img,
        sub_bin,
        sub_aux,
        roi_mode=True
    )
    columns = _offset_columns(columns, x0, y0)
    # 轮廓已偏移回整图坐标；元数据中的边缘/完整柱统计必须按整图宽度
    # 判断，不能拿局部 ROI 宽度把 ROI 左右两侧误标成整图残柱。
    detection_meta = _detection_meta_from_aux(
        sub_aux, columns, image_width=w
    )

    # ROI 结果绘制在整图，横向结构投影不再拼到右下综合图
    px_main, stats_data, stats_path = export_detection_outputs(
        image_path,
        img,
        columns,
        None,
        "_user",
        detection_meta=detection_meta,
    )

    payload = {
        "path": image_path,
        "img": img,
        "columns": columns,
        "structure_profile": None,
        "local_structure_profile": local_profile,
        "tag": "_user",
        "stats": stats_data,
        "stats_path": stats_path,
        "quality": aux["quality"],
        "threshold_mode": aux["threshold_mode"],
        "detection_mode": sub_aux.get("detection_mode", aux.get("detection_mode", "unknown")),
        "detection_variant": sub_aux.get("detection_variant", "unknown"),
        "detection_confidence": sub_aux.get("detection_confidence", -1.0),
        "detection_attempts": sub_aux.get("detection_attempts", 0),
        "detection_warning": sub_aux.get("detection_warning", ""),
        "cd_hint_px": sub_aux.get("cd_hint_px"),
        "cd_range_px": sub_aux.get("cd_range_px"),
        "cd_observed_px": detection_meta.get("cd_observed_px"),
        "detection": detection_meta,
        "binary_fraction": aux["binary_fraction"],
        "roi": (x0, y0, x1, y1),
    }

    if on_image is not None:
        on_image(img, px_main, os.path.basename(image_path), payload)

    return payload


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

# GUI 主参数：只保留“需要用户调”的核心项。
# 内部智能参数（自适应阈值窗口/C、强梯度分位、平滑核、窗口外扩等）
# 仍生效，但不再占用界面 —— 由逐像素横/列智能斜率分析自动接管。
LEGACY_MAIN_PARAMS = [
    # —— 二值化（核心：先看横向灰度斜率，再兼顾亮度） ——
    ("THRESHOLD",          "二值化阈值",       "int",   0,    255,  1),
    ("THRESHOLD_TOL",      "灰度容差",         "int",   0,    80,   1),
    ("GRADIENT_THRESHOLD_ENABLE", "斜率智能选阈值", "int", 0, 1,   1),
    ("GRADIENT_THRESHOLD_GAIN",   "斜率选阈值增益", "float", 1.0,  2.0,  0.05),
    # —— 分柱 ——
    ("VALLEY_RATIO",       "谷值比例",         "float", 0.02, 0.9,  0.02),
    # —— 错拆合并（防把一根真实柱切成两根） ——
    ("MERGE_GAP_PX",       "近柱合并(px)",     "int",   0,    30,   1),
    ("MERGE_VALLEY_RATIO", "浅谷合并比",       "float", 0.0,  0.5,  0.01),
    ("MERGE_DEEP_CRACK_RATIO", "深裂合并比",   "float", 0.0,  0.5,  0.01),
    ("MERGE_HEIGHT_RATIO_MAX", "合并高度比",   "float", 1.0,  3.0,  0.05),
    ("MERGE_BOTTOM_DIFF_MAX", "合并底边差(px)", "int",  0,    50,   1),
    # —— 几何过滤 ——
    ("MIN_COLUMN_HEIGHT",  "最小柱高(px)",     "int",   1,    800,  1),
    ("MIN_AVG_WIDTH",      "最小平均宽(px)",   "int",   1,    500,  1),
    ("MIN_ASPECT_RATIO",   "最小高宽比",       "float", 0.2,  10.0, 0.1),
    # —— 后处理 ——
    ("BOTTOM_OFFSET",      "底部延伸(px)",     "int",   0,    50,   1),
    ("EDGE_MARGIN",        "边缘忽略(px)",     "int",   0,    100,  1),
    ("FOOTING_TRIM_ENABLE","底部Footing修剪",  "int",   0,    1,    1),
    ("FOOTING_TRIM_RATIO", "Footing判定倍数",  "float", 1.0,  3.0,  0.05),
    ("BRIDGE_GAP_PX",      "粘连提示间隙(px)", "int",   0,    10,   1),
]

PARAM_DESC = {
    "THRESHOLD": "固定阈值候选。高质量图优先使用；低/中质量图会自动使用 Otsu 或融合策略。",
    "THRESHOLD_TOL": "允许略低于固定阈值的像素参与固定阈值候选。",
    "GRADIENT_THRESHOLD_ENABLE": "智能二值化：按“逐像素横向灰度斜率”搜索使柱壁边界与灰度突变最吻合的阈值"
                                 "（1=开启，0=关闭）。不同图亮暗差异大时更稳。",
    "GRADIENT_THRESHOLD_GAIN": "智能二值化：斜率候选需优于规则选择该倍数才替换（越大越保守）。",
    "VALLEY_RATIO": "结构投影低于最大值×该比例时视为柱间谷。现在会真实参与算法。",
    "MERGE_GAP_PX": "智能后处理：相邻两柱水平间隙 < 该值时直接合并成一根（防错拆）。0=关闭。",
    "MERGE_VALLEY_RATIO": "智能后处理：相邻柱间亮度投影谷值 ≥ 该比例×两侧峰值平均时合并。"
                          "处理“shallow valley”把一根柱错拆的情况；0=关闭。",
    "MERGE_DEEP_CRACK_RATIO": "智能后处理：相邻柱间亮度投影谷值 ≤ 该比例×两侧峰值平均，"
                              "且高度/底边对齐时合并。处理“同一根柱被中间深缝错拆成两根”。",
    "MERGE_HEIGHT_RATIO_MAX": "智能后处理：浅谷/深裂合并要求两柱高度比 ≤ 该值，避免错合上下错位的两根柱。",
    "MERGE_BOTTOM_DIFF_MAX": "智能后处理：浅谷/深裂合并要求两柱底边 y 差 ≤ 该像素，避免错合高度不同的两根柱。",
    "MIN_COLUMN_HEIGHT": "排除过矮亮块。",
    "MIN_AVG_WIDTH": "排除过窄亮条。",
    "MIN_ASPECT_RATIO": "排除扁宽噪声/底座亮斑。",
    "BOTTOM_OFFSET": "识别后柱底额外向下延伸，但自动限制在图像范围内。",
    "EDGE_MARGIN": "忽略图像四边，避免框线/标尺；程序还会自动检测并裁掉连续白边框。",
    "FOOTING_TRIM_ENABLE": "智能后处理：底部因 footing/基座突然变宽时，自动把柱体底边上移到展宽起点"
                           "（1=开启，0=关闭）。可避免基座被算进柱高、污染底部 CD。",
    "FOOTING_TRIM_RATIO": "智能后处理：底部行宽 ≥ 柱身中位宽 × 该倍数时判定为 footing 台阶。",
    "BRIDGE_GAP_PX": "智能后处理：相邻两柱在共同高度上的水平间隙 ≤ 该值时，判定表提示“疑似粘连”。",
}

# 普通用户只需提供 Line CD 的大致像素值和允许误差；其余低层参数、
# 候选版本和重试次数全部由程序内部自动管理，避免左侧面板充满难以
# 解释的阈值旋钮。自动重试始终使用 AUTO_RETRY_ENABLE/MAX 控制，
# 但不要求用户参与调节。
MAIN_PARAMS = [
    ("AUTO_CD_ESTIMATE_PX", "估计 Line CD (px，0=自动)", "float", 0, 100000, 1),
    ("AUTO_CD_TOLERANCE_PX", "允许误差 ±(px)", "float", 1, 10000, 1),
]
PARAM_DESC.update({
    "AUTO_CD_ESTIMATE_PX": "完整柱 Line CD 的大致边界距离（px）。填 0 时由候选柱的中位宽自动估计。",
    "AUTO_CD_TOLERANCE_PX": "用户先验的半宽范围。输出会同时列出 ±1 倍用户范围与 ±2 倍候选保护范围；边缘残柱/软候选还会自动放宽。",
})

FORMULA_HELP = """统计说明

1. 高度：
   height = 最底 y - 最顶 y + 行间距

2. 单行 CD：
   CD = x_end - x_start

3. 上/中/下三段：
   按轮廓行数均分为三段，分别计算平均 CD。

4. 侧壁角：
   对边缘 x~y 做线性拟合，斜率 b=dx/dy。
   angle = atan2(1,b)
   竖直壁=90°；向右倾<90°；向左倾>90°。

5. LER 3σ：
   对整根柱边缘做 x=a+b*y 拟合，
   LER = 3 × std(实际 x - 拟合 x)

6. 新版精度：
   stats_nm 直接由原始像素轮廓 × nm/px 的 float64 数值计算，
   不经过 boundaries_nm 文本舍入。

7. Judge（Profile/Target %）：
   Judge = mean(W(y)) / Target CD × 100。
   100 = 平均 Profile CD 等于 Target；>100 = Profile 偏宽；
   <100 = Profile 偏窄。该分数是方向性比例，不限制上限；
   未启用 Target 或缺少有效曲线时显示 “-”。

8. 像素宽度口径：
   Line CD 先验、观测范围和统计表 CD 统一使用 x_end-x_start（左右边界距离），
   与手动量测的 px/nm 一致；内部几何过滤可能使用含端点的像素格数量。
"""

SHAPE_HELP = """形貌（Profile）判定说明
================================================

〇、W(y) 形貌归类表（本版采用；所有柱与平均值都与 Target 对比后归类）
------------------------------------------------
    Profile             W(y) 特征
    Ideal / Square      基本恒定（理想方波）
    Footing             靠近底部突然变宽
    Undercut            靠近底部突然变窄
    Positive taper      从顶到底逐渐变宽
    Negative taper      从顶到底逐渐变窄
    Bowing              中间比上下更宽
    Necking             中间局部变窄
    T-top / Top flare   顶部局部变宽
    Asymmetric          左右边缘变化不对称

判定标记：✅ 正常（偏离 ≤ 0.5×容差）
          ⚠️ 临界（0.5~1×容差）
          ❌ 超差（> 1×容差）
（Judge 使用 “mean(W(y)) / Target CD × 100%” 的方向性评分，
不限制上限：100 表示等宽，>100 表示 Profile 比 Target 宽，<100 表示偏窄；
一根 PR 柱子只会给出一个最严重的形貌。）

输出位置：
  · 统计表 xx_*_stats_nm.txt 的最后两行：
        Profile  → 每根柱 + 平均值的归类（多项用 “+” 连接）
        Judge    → Profile/Target (%) 方向性分数（100=等宽，>100=偏宽，<100=偏窄）
  · xx_*_target_check.txt 里的“对象 × 类别”判定矩阵；
  · 右侧“标准 Target”面板的 Profile / Judge 两列（最后一行是平均值）。

一、纵向逐像素扫描除了“分柱”，还能做什么？
------------------------------------------------
横向压缩（列投影）只能大致看出柱子在哪些 x 位置、底部大致范围；
而“纵向逐像素扫描”给出每根柱每一行的左右边界，于是可以得到
CD(y) 曲线（线宽随高度的变化）。这条曲线是形貌判定的全部依据：

  · CD(y) 全程基本水平      → 侧壁竖直，正常；
  · CD(y) 顶部变大          → T-top（蘑菇头 / 顶部外扩）；
  · CD(y) 中段变小          → Undercut（腰部被钻进去、内凹）；
  · CD(y) 底部变大          → Footing（底部站裙 / 脚宽）；
  · CD(y) 单调收窄或放宽    → Taper（梯形）；
  · CD(y) 抖动大            → Irregular（形状异常 / 噪声大）。

二、本工具实际计算的形貌量
------------------------------------------------
1) 顶部/中段/底部三段 CD
   取轮廓行数的上 15% / 中间 / 下 15% 的平均 CD。

2) widen_top = CD_top / CD_mid
   顶部相对中段的展宽倍数。

3) widen_bot = CD_bot / CD_mid
   底部相对中段的展宽倍数（Footing 的主判据）。

4) waist_ratio = CD_mid / ((CD_top + CD_bot) / 2)
   腰部相对“肩部”的收缩程度（Undercut 的主判据，越小越内凹）。

5) taper_ratio = CD_top / CD_bot
   上下的整体收放趋势（Taper 判据）。

6) cd_uniformity = 3σ(CD) / mean(CD)
   整根柱 CD 的均匀性。越小越接近理想方波；过大说明边缘抖动/形状异常。

7) footing_step_nm / footing_height_nm
   自下往上找到第一个“回到柱身宽度”的行，得到底部展宽台阶的
   宽度增量与高度。可用于量化 footing 有多宽、多高。

8) bow_nm
   CD(y) 对 y 做线性拟合后的最大残差，即侧壁相对直线的最大弯曲量。

三、判定规则（可在右侧“标准 Target”面板调灵敏度）
------------------------------------------------
  · widen_top ≥ 顶部展宽阈值 且 绝对增量 ≥ SHAPE_MIN_DELTA_NM → T-top
  · waist_ratio ≤ 腰部收缩阈值 且 绝对收缩 ≥ SHAPE_MIN_DELTA_NM → Undercut
  · widen_bot ≥ 底部展宽阈值 且 绝对增量 ≥ SHAPE_MIN_DELTA_NM → Footing
  · 以上都没有，但 cd_uniformity > SHAPE_IRREGULAR_3SIG → Irregular
  · 以上都没有，且 taper_ratio 明显 ≠ 1 → Taper（上宽下窄 / 上窄下宽）
  · 其余 → Vertical(正常)

多个形貌会同时成立，界面用 “+” 连接，例如 “Undercut+Footing”。

四、智能后处理（纵向扫描带来的额外收益）
------------------------------------------------
1) footing 台阶修剪
   底部因 footing/基座突然变宽时，自动把“柱体底边”上移到展宽起点，
   避免把基座算进柱高、污染底部 CD 与侧壁角。
   开关：FOOTING_TRIM_ENABLE；灵敏度：FOOTING_TRIM_RATIO。

2) 粘连检测
   相邻两根柱在共同高度上的水平间隙 ≤ BRIDGE_GAP_PX 时，
   在判定表“提示”里标记“与邻柱疑似粘连”，提醒可能需要调谷值比例重新分柱。
"""

TARGET_HELP = """标准 Target（理想方波）对比说明
================================================

一、什么叫“理想方波”
------------------------------------------------
理想 PR 胶图形的侧壁是竖直的：从顶到底线宽恒定为 Target CD。
把每根柱的 CD(y) 画出来就是一条“方波”。因此：

    理想：  CD_top = CD_mid = CD_bot = Target
    实际：  三段各自偏离 Target，偏离的方向就对应缺陷类型。

二、怎么判（启用 Target 后）
------------------------------------------------
设容差 tol = max(绝对容差 nm, Target × 相对容差 %)：

  · 中段 CD 比 Target 细（Target − CD_mid > tol）      → Undercut
  · 底部 CD 比 Target 宽（CD_bot − Target > tol）      → Footing
  · 顶部 CD 比 Target 宽（CD_top − Target > tol）      → T-top
  · 三段都在 ±tol 内                                    → Normal
  · 多个条件同时成立 → 用 “+” 连接（如 Undercut+Footing）

判定列：
  · PASS = 上/中/下三段都在容差内（若填了目标高度，高度也要合格）；
  · FAIL = 任一段超差。

三、输出的其它量
------------------------------------------------
  · Δvs Target   = 该柱平均 CD − Target（带正负号）；
  · Judge（Profile/Target %） = mean(W(y)) / Target CD × 100；
                    100=等宽，>100=Profile 偏宽，<100=Profile 偏窄，分数无上限；
  · 方波相似度    = 1 − mean(|CD_i − Target|) / Target，越接近 1 越像方波；
  · 最大 |Δ|     = 所有柱里偏差最大的绝对值；
  · 片内均匀性3σ = 所有柱平均 CD 的 3σ / 均值，衡量片内一致性；
  · 形貌分布      = 各类形貌各有多少根。

四、怎么用
------------------------------------------------
1) 右侧“标准 Target（理想方波）对比 / 形貌判定”面板；
2) 勾选“启用 Target 对比”，填入 Target CD、容差（nm 与 % 取大者）、
   目标高度（填 0 表示不判断高度）；
3) 选“对比对象”（_tool 自动整图 / _user 人工 ROI）；
4) 参数改完会自动重算（只重算统计与判定，不重新做图像检测，很快）；
5) 程序会自动保存
   xx_tool_target_check.txt / xx_user_target_check.txt；
   界面提供“保存统计表另存为…”用于复制 stats 文件，判定表可在输出目录直接取得。

五、与 W(y) 归类的关系
------------------------------------------------
启用 Target 后，“Profile(归类)”这一列按 W(y) 与 Target 方波的差异给出：
    Ideal/Square、Footing、Undercut、Positive taper、Negative taper、
    Bowing、Necking、T-top/Top flare、Asymmetric
并在 Judge 列给出 Profile/Target (%) 分数；100 表示平均线宽等于 Target，
大于 100 表示偏宽，小于 100 表示偏窄。PASS/FAIL 仍按 CD 与容差判断。
未启用 Target 时，仍按柱体自身几何归类，只是判定不以 Target 为基准。

六、输出
------------------------------------------------
  · 统计表最后两行：Profile（归类）与 Judge（Profile/Target (%)）；
  · xx_*_target_check.txt：完整判定矩阵（逐柱 + 平均值）；
  · 右侧面板：逐柱 + 平均值一行的 Profile / Judge。

提示：不勾选启用时，仍会按柱体自身几何给出形貌，Judge 显示 “-”；
勾选后分数可大于 100，用于明确表示 Profile 比 Target 更宽。
"""

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

因为曲线横轴就是像素 index，所以：
  · 横向行剖面上量一段 = 横向距离（可直接当 CD / 间距用，单位 nm）；
  · 纵向列剖面上量一段 = 纵向距离（可直接当高度 / footing 高度用）；
  · 直线剖面上量一段   = 沿该直线的实际长度。

三、小技巧
------------------------------------------------
  · 想量多根柱的间距：切“横向行剖面”，取柱底附近那一行，
    在曲线上 Shift+点击相邻两个亮区的中点；
  · 想看 footing：切“纵向列剖面”，取柱子中心列，
    底部灰度变宽的位置就是 footing 起点；
   · 量测结果会随比例尺自动换算 nm，改比例尺后立即生效。
   · 也可以直接在原图拖动蓝色 A/B 端点，直线剖面会同步刷新；
   · 行/列剖面的橙色量测端点独立于直线采样 A/B，不会意外改变原曲线。
 """

FILE_TAG_HELP = """输出文件说明
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
  xx_tool_stats_nm.txt          统计表（nm；末尾附识别候选、置信度、CD范围和 Target/Judge 定义）
  xx_tool_target_check.txt      Target 方波对比 + 形貌判定表（含识别参数摘要）

人工 ROI 结果（_user）
  xx_user_result.png / xx_user_result_nm.png
  xx_user_boundaries.txt / xx_user_boundaries_nm.txt
  xx_user_stats_nm.txt          统计表（nm；末尾附识别候选、置信度、CD范围和 Target/Judge 定义）
  xx_user_target_check.txt      Target 方波对比 + 形貌判定表（含识别参数摘要）

调试图（勾选“保存调试图”时）
  xx_debug_binary.png           二值图
  xx_debug_grad_x.png           Sobel-X 梯度
  xx_debug_grad_y.png           Sobel-Y 梯度
  xx_debug_enhanced.png         CLAHE 增强后的灰度图

输出位置：
  · 勾选“输出到同名子文件夹” → 每张图一个同名子文件夹；
  · 不勾选 → 直接放在图像文件夹里。

每次识别的文本输出都会附带：识别模式、候选版本（含亮/暗极性）、
置信度（候选排序分数，非百分比）、候选尝试次数、估计 Line CD/允许误差、
用户先验范围、候选保护范围、完整主体柱/全部柱/边缘残柱观测范围，以及
Target CD、有效容差和目标高度。Line CD 与统计 CD 均按左右边界距离输出。
"""

ABOUT_HELP = """PR 柱子智能识别与测量工具
作者：E924744 Kang An

版本要点：
  · 自动整图识别（_tool）+ 人工框选 ROI 识别（_user）
  · 灰度斜率（Sobel-X）+ 亮度 + 结构投影联合定位柱区
  · 自适应阈值按图像质量自动选择（固定 / Otsu / 局部自适应）
  · 纵向逐像素扫描 → CD(y) 曲线 → 形貌判定
      （Vertical / T-top / Undercut / Footing / Taper / Irregular）
  · footing 台阶自动修剪、相邻柱粘连检测
  · 标准 Target（理想方波）对比：PASS/FAIL + 形貌分类 +
      Profile/Target (%) 方向性 Judge（可大于 100，表示 Profile 偏宽）
  · 灰度剖面三种模式（直线 / 横向行 / 纵向列）
      曲线上单击定位到原图；Shift+单击两点量取线段（px 与 nm）
  · 直线量测、三点角度量测（水平/竖直自动吸附）
  · 参数变化防抖自动重识别；比例尺变化只重算 nm 输出与统计
  · 帮助菜单：识别流程 / 参数说明 / 统计公式 / 形貌判定 /
              Target 对比 / 灰度曲线与量测 / 输出文件 / 关于

本文件为单文件版：分析核心已内置，不依赖其它 .py。
同目录附带《使用说明.md》，包含参数、曲线量测、Target/Judge 和输出字段的完整说明。
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("PR 柱子智能识别与测量工具")
        self.geometry("1660x960")
        self.minsize(1280, 760)

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
        self._last_stat_src = {"tool": None, "user": None}
        self._analyzed = []

        self._live_job = None
        self._live_dirty = False
        self._scale_job = None
        self._scale_dirty = False

        self._build_ui()
        self._refresh_files()

        # 识别参数变化 -> 防抖重识别
        for var in self.spins.values():
            var.trace_add("write", self._schedule_live)

        # 比例尺变化 -> 只重导出/重统计，不做检测
        for var in self.scale_vars.values():
            var.trace_add("write", self._on_scale_var_change)

        self.after(80, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------

    def _build_ui(self):
        try:
            ttk.Style(self).theme_use("vista")
        except Exception:
            pass

        try:
            st = ttk.Style(self)
            st.configure(".", font=("Microsoft YaHei UI", 9))
            st.configure(
                "TLabelframe.Label",
                font=("Microsoft YaHei UI", 9, "bold"),
                foreground="#175a7a"
            )
        except Exception:
            pass

        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(
            label="识别流程（含纵向扫描）",
            command=self._show_pipeline
        )
        help_menu.add_command(label="参数说明", command=self._show_params_help)
        help_menu.add_command(
            label="统计公式",
            command=lambda: self._show_doc("统计公式", FORMULA_HELP)
        )
        help_menu.add_separator()
        help_menu.add_command(
            label="形貌判定说明",
            command=lambda: self._show_doc("形貌判定说明", SHAPE_HELP)
        )
        help_menu.add_command(
            label="Target 方波对比说明",
            command=lambda: self._show_doc("Target 方波对比说明", TARGET_HELP)
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
        ttk.Entry(top, textvariable=self.var_dir, width=32).pack(side="left", padx=4)

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

        self.btn_run1 = ttk.Button(
            top,
            text="▶ 分析选中图像",
            command=self._start_single
        )
        self.btn_run1.pack(side="left", padx=4)

        self.btn_run_all = ttk.Button(
            top,
            text="批量分析全部",
            command=self._start_batch
        )
        self.btn_run_all.pack(side="left", padx=2)

        self.btn_open = ttk.Button(
            top,
            text="打开输出文件夹",
            command=self._open_output
        )
        self.btn_open.pack(side="left", padx=4)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(2, 4))

        # 左
        left = ttk.Frame(body, width=440)
        left.pack(side="left", fill="y", padx=(0, 6))
        left.pack_propagate(False)

        pf = ttk.LabelFrame(left, text="识别参数")
        pf.pack(fill="x", padx=4, pady=(2, 6))

        self.spins = {}

        for i, spec in enumerate(MAIN_PARAMS):
            attr, name, kind, lo, hi, inc = spec
            row = i // 2
            col = i % 2

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
                "只需估计 Line CD 与允许误差；程序会自动建立候选范围，"
                "统一评估最多 5 个灰度候选并选择综合得分最高者，"
                "同时保留左右边缘残柱。"
            ),
            foreground="#666666",
            wraplength=410,
            justify="left"
        ).grid(
            row=(len(MAIN_PARAMS) + 1) // 2,
            column=0,
            columnspan=2,
            sticky="w",
            padx=8,
            pady=(4, 6)
        )

        sf = ttk.LabelFrame(left, text="比例尺与输出")
        sf.pack(fill="x", padx=4)

        self.scale_vars = {}

        for attr, name, lo, hi, inc in [
            ("SCALE_PIXELS", "比例尺长度(px)", 1, 100000, 1),
            ("SCALE_NM", "实际长度(nm)", 0.01, 100000, 0.5),
        ]:
            cell = ttk.Frame(sf)
            cell.pack(fill="x", padx=8, pady=3)

            ttk.Label(cell, text=name).pack(side="left")

            var = tk.StringVar(value=str(getattr(core, attr)))
            tk.Spinbox(
                cell,
                from_=lo,
                to=hi,
                increment=inc,
                width=10,
                textvariable=var,
                font=("Segoe UI", 10)
            ).pack(side="left", padx=4)

            self.scale_vars[attr] = var

        self.lbl_scale_info = ttk.Label(sf, text="")
        self.lbl_scale_info.pack(anchor="w", padx=8, pady=(0, 3))

        self.var_subdir = tk.BooleanVar(value=bool(core.OUTPUT_SUBDIR))
        ttk.Checkbutton(
            sf,
            text="输出到同名子文件夹",
            variable=self.var_subdir
        ).pack(anchor="w", padx=8)

        self.var_debug = tk.BooleanVar(value=bool(core.DEBUG_OUTPUT))
        ttk.Checkbutton(
            sf,
            text="保存调试图（二值 / x梯度 / y梯度 / CLAHE）",
            variable=self.var_debug
        ).pack(anchor="w", padx=8, pady=(0, 4))

        ttk.Label(
            sf,
            text="比例尺修改时：若当前识别结果已缓存，只重算 nm 输出与统计，不重新识别。",
            foreground="#777777",
            wraplength=410,
            justify="left"
        ).pack(anchor="w", padx=8, pady=(2, 6))

        self._update_scale_info()

        hf = ttk.LabelFrame(left, text="已分析文件")
        hf.pack(fill="both", expand=True, padx=4, pady=(8, 4))

        self.lst_files = tk.Listbox(
            hf,
            height=7,
            exportselection=False,
            font=("Microsoft YaHei UI", 9)
        )
        self.lst_files.pack(fill="both", expand=True, padx=6, pady=6)
        self.lst_files.bind("<<ListboxSelect>>", self._on_pick_file)

        # 中
        mid = ttk.Frame(body)
        mid.pack(side="left", fill="both", expand=True, padx=6)
        self.frm_mid = mid

        tools = ttk.Frame(mid)
        tools.pack(fill="x", pady=(0, 2))

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
            tools,
            text="▶ 套索分析所选区域",
            command=self._start_roi
        )
        self.btn_roi_run.pack(side="right")

        self.btn_roi_clear = ttk.Button(
            tools,
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

        orig_row = ttk.Frame(mid)
        orig_row.pack(fill="both", expand=True)

        self.frm_orig = ttk.LabelFrame(orig_row, text="原始图像")
        self.frm_orig.pack(side="left", fill="both", expand=True, padx=(0, 2))

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
                text="灰度剖面（直线 / 横向行 / 纵向列）"
            )
            self.frm_line_profile.pack(side="left", fill="both", expand=True, padx=(2, 0))

            ctl = ttk.Frame(self.frm_line_profile)
            ctl.pack(fill="x", padx=4, pady=(4, 0))

            self.var_profile_mode = tk.StringVar(value="直线剖面")
            ttk.Combobox(
                ctl,
                textvariable=self.var_profile_mode,
                state="readonly",
                width=11,
                values=("直线剖面", "横向行剖面", "纵向列剖面")
            ).pack(side="left")
            self.var_profile_mode.trace_add("write", self._on_profile_mode_change)

            ttk.Label(ctl, text="行 y / 列 x：").pack(side="left", padx=(8, 0))
            self.var_profile_pos = tk.StringVar(value="0")
            self.spn_profile_pos = tk.Spinbox(
                ctl,
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

        result_row = ttk.Frame(mid)
        result_row.pack(fill="both", expand=True, pady=(12, 0))
        # 增加预览区域上边距，使预览位置更高

        self.frm_tool = ttk.LabelFrame(result_row, text="工具自动识别预览（_tool）")
        self.frm_tool.pack(side="left", fill="none", expand=False, padx=(0, 2))
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

        self.frm_user = ttk.LabelFrame(result_row, text="人工 ROI 识别预览（_user）")
        self.frm_user.pack(side="left", fill="none", expand=False, padx=(2, 0))
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
        right = ttk.Frame(body, width=510)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.pack_propagate(False)

        self._stat_blocks = {}
        for kind, title in [
            ("tool", "工具自动识别统计（_tool）"),
            ("user", "人工 ROI 统计（_user）"),
        ]:
            blk = self._make_stats_block(right, kind, title)
            blk["frame"].pack(fill="both", expand=True, pady=(0, 4))
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

        var = tk.StringVar(value="（尚无结果）")
        ttk.Label(
            frame,
            textvariable=var,
            foreground="#555555",
            wraplength=440
        ).pack(anchor="w", padx=6, pady=(2, 0))

        # ---- Target 设定行（每个统计块各自独立）----
        trow = ttk.Frame(frame)
        trow.pack(fill="x", padx=6, pady=(4, 0))

        tvars = {
            "enable": tk.BooleanVar(value=bool(core.TARGET_ENABLE)),
            "cd": tk.StringVar(value=f"{core.TARGET_CD_NM:g}"),
            "tol_nm": tk.StringVar(value=f"{core.TARGET_TOL_NM:g}"),
            "tol_pct": tk.StringVar(value=f"{core.TARGET_TOL_PCT:g}"),
            "height": tk.StringVar(value=f"{core.TARGET_HEIGHT_NM:g}"),
        }

        ttk.Checkbutton(
            trow, text="Target", variable=tvars["enable"],
            command=lambda k=kind: self._on_target_change(k)
        ).pack(side="left")

        for label, var in (
            ("Line CD", tvars["cd"]),
            ("±nm", tvars["tol_nm"]),
            ("±%", tvars["tol_pct"]),
            ("高度nm", tvars["height"]),
        ):
            ttk.Label(trow, text=label).pack(side="left", padx=(6, 0))
            tk.Entry(trow, textvariable=var, width=5).pack(side="left", padx=2)

        for v in (tvars["cd"], tvars["tol_nm"], tvars["tol_pct"], tvars["height"]):
            v.trace_add("write", lambda *a, k=kind: self._on_target_change(k))

        ttk.Label(
            frame,
            text="容差取 ±nm 与 ±% 换算值中的较大者；高度填 0 表示不判断。",
            foreground="#687789",
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w", padx=8, pady=(2, 0))

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
            "var": var,
            "save": save,
            "target": tvars,
        }

    # -----------------------------------------------------------------
    # 每个统计块各自的 Target 设定与判定（结果直接追加到统计表最后两行）
    # -----------------------------------------------------------------

    def _on_target_change(self, kind):
        """某个统计块的 Target 参数变化：只重算该块的判定，不重新识别。"""
        key = f"_target_job_{kind}"
        if getattr(self, key, None) is not None:
            try:
                self.after_cancel(getattr(self, key))
            except Exception:
                pass
        setattr(
            self,
            key,
            self.after(260, lambda k=kind: self._target_job_run(k))
        )

    def _target_job_run(self, kind):
        setattr(self, f"_target_job_{kind}", None)
        self._refresh_target_rows(kind)

    def _target_dict(self, kind):
        """读取某个统计块的 Target 设定为 dict。"""
        tv = self._stat_blocks[kind]["target"]

        def f(var, default):
            try:
                return float(str(var.get()).strip())
            except Exception:
                return default

        cd = f(tv["cd"], core.TARGET_CD_NM)
        tol_nm = f(tv["tol_nm"], core.TARGET_TOL_NM)
        return {
            "enable": bool(tv["enable"].get()),
            "cd": max(1e-9, cd),
            "tol_nm": max(0.0, tol_nm),
            "tol_pct": max(0.0, f(tv["tol_pct"], core.TARGET_TOL_PCT)),
            "height": max(0.0, f(tv["height"], core.TARGET_HEIGHT_NM)),
        }

    def _refresh_target_rows(self, kind):
        """用该块 Target 重新计算判定，并重建统计表（含最后两行）。"""
        payload = self._analysis_cache.get(kind)
        if payload is None or not self._last_img_name:
            return
        if os.path.basename(payload["path"]) != self._last_img_name:
            return
        self._populate_stats_tree(kind, payload.get("stats"), persist_target=True)

    # -----------------------------------------------------------------
    # 文件/参数
    # -----------------------------------------------------------------

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

        self._scan_existing_results()

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

            if found and stem not in self._analyzed:
                self._analyzed.append(stem)

        self._refresh_analyzed_list()

    def _refresh_analyzed_list(self):
        self.lst_files.delete(0, "end")
        for stem in self._analyzed:
            self.lst_files.insert("end", stem)

    def _browse_dir(self):
        d = filedialog.askdirectory(
            initialdir=self._folder(),
            title="选择包含 tif 图像的文件夹"
        )
        if d:
            self.var_dir.set(os.path.normpath(d))
            self._refresh_files()

    def _browse_file(self):
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

    def _on_pick_file(self, _evt=None):
        if self.running:
            return

        sel = self.lst_files.curselection()
        if not sel:
            return

        stem = self.lst_files.get(sel[0])
        for name in self.combo["values"]:
            if os.path.splitext(name)[0] == stem:
                self.var_sel.set(name)
                self._start_single()
                return

    def _collect_params(self):
        folder = self._folder()
        if not os.path.isdir(folder):
            raise ValueError("图像文件夹不存在")

        core.INPUT_DIR = folder
        core.OUTPUT_SUBDIR = bool(self.var_subdir.get())
        core.DEBUG_OUTPUT = bool(self.var_debug.get())

        for attr, name, kind, *_ in MAIN_PARAMS:
            raw = self.spins[attr].get().strip().replace("，", ".").replace(",", ".")
            value = int(float(raw)) if kind == "int" else float(raw)

            if not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限数值")
            if attr == "AUTO_CD_ESTIMATE_PX" and value < 0:
                raise ValueError("估计 Line CD 只能填 0 或正数")
            if attr == "AUTO_CD_TOLERANCE_PX" and value <= 0:
                raise ValueError("允许误差必须大于 0 px")

            if attr in ("MORPH_CLOSE_K", "SMOOTH_KERNEL", "ADAPTIVE_BLOCK_SIZE"):
                value = ensure_odd(value)

            setattr(core, attr, value)

        self._collect_scale_only()

    def _collect_scale_only(self):
        px = float(
            self.scale_vars["SCALE_PIXELS"].get()
            .strip().replace("，", ".").replace(",", ".")
        )
        nm = float(
            self.scale_vars["SCALE_NM"].get()
            .strip().replace("，", ".").replace(",", ".")
        )

        if px <= 0:
            raise ValueError("比例尺像素数必须大于 0")
        if nm <= 0:
            raise ValueError("比例尺实际长度必须大于 0")

        core.SCALE_PIXELS = px
        core.SCALE_NM = nm
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

        p = os.path.join(self._folder(), self._last_img_name)
        if not os.path.isfile(p):
            return

        self._start_jobs(
            [{"path": p, "mode": "full", "roi": None}],
            f"参数变化 → 自动重识别：{self._last_img_name}"
        )

    def _on_scale_var_change(self, *_args):
        self._update_scale_info()

        if not self._last_img_name or not self._params_are_valid():
            return

        if self.running:
            self._scale_dirty = True
            return

        if self._scale_job is not None:
            self.after_cancel(self._scale_job)

        self._scale_job = self.after(320, self._run_scale_refresh)

    def _run_scale_refresh(self):
        self._scale_job = None

        if self.running:
            self._scale_dirty = True
            return

        caches = [
            self._analysis_cache.get("tool"),
            self._analysis_cache.get("user"),
        ]
        caches = [
            c for c in caches
            if c is not None and os.path.basename(c["path"]) == self._last_img_name
        ]

        if not caches:
            return

        try:
            self._collect_scale_only()
        except Exception:
            return

        self._set_running(True)
        self.var_status.set("比例尺变化 → 仅重算 nm 输出与统计…")

        def worker():
            try:
                for payload in caches:
                    px_main, stats, stats_path = core.export_detection_outputs(
                        payload["path"],
                        payload["img"],
                        payload["columns"],
                        payload.get("structure_profile"),
                        payload["tag"],
                        detection_meta=payload.get("detection"),
                    )

                    new_payload = dict(payload)
                    new_payload["stats"] = stats
                    new_payload["stats_path"] = stats_path

                    self.q.put((
                        "img",
                        payload["tag"],
                        payload["img"],
                        px_main,
                        os.path.basename(payload["path"]),
                        new_payload
                    ))

                self.q.put(("scale_done", None))
            except Exception:
                self.q.put(("scale_done", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

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

        p = os.path.join(self._folder(), self._last_img_name)
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
                            on_image=self._make_cb("_user")
                        )
                    else:
                        core.process_image(
                            job["path"],
                            on_image=self._make_cb("_tool")
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
        ):
            widget.configure(state=state)

        self.combo.configure(
            state="disabled" if on else "readonly"
        )
        self.lst_files.configure(
            state="disabled" if on else "normal"
        )

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

    def _set_preview(self, tag, orig, paint, name, payload):
        changed = self._last_img_name != name

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
            self._user_bgr = None
            self._analysis_cache["user"] = None
            self._line_profile_data = None
            self._profile_meta = {"mode": "line"}
            self._profile_pick = None
            self._profile_span = []
            self._profile_drag = None

        self._last_img_name = name
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

        stem = os.path.splitext(name)[0]
        if stem not in self._analyzed:
            self._analyzed.append(stem)
            self._refresh_analyzed_list()

        self._redraw_preview()

        kind = "tool" if tag == "_tool" else "user"
        # 首次显示也把当前块的 Target 设定同步到文件，避免用户此前
        # 修改过 Target 后切换图像时出现“表格新、文件旧”。
        self._populate_stats_tree(kind, payload["stats"], persist_target=True)

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

        s = min(1.0, cw / ow, ch / oh)
        s = max(s, 0.02)
        self._scale = s

        draw = self._orig_bgr
        if s < 1.0:
            draw = cv2.resize(
                draw,
                (max(1, int(ow * s)), max(1, int(oh * s))),
                interpolation=cv2.INTER_AREA
            )

        ph = self._bgr_to_photo(draw)
        if ph is not None:
            self._photos.append(ph)
            self.cv_orig.create_image(0, 0, anchor="nw", image=ph)

        self._img_x = 0
        self._img_y = 0

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
        common = max(0.02, min(1.0, fw_t / tool_w, fh_t / tool_h))

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
            s = min(s, max(0.02, min(1.0, fw / w, fh / h)))

            draw = bgr
            if s < 1.0:
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
        square_size = max(250, min(square_size, 600))  # 限制在 250-600 之间（增大）

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
            ax.set_xlabel("位置 index (px)", fontsize=9)
            ax.set_ylabel("Gray", fontsize=9)
            ax.grid(True, alpha=0.25, color="#9aaabd", linewidth=0.6)
            self._line_profile_fig.tight_layout()
            self._line_profile_canvas.draw()
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
        ax.set_xlabel("位置 index (px)", fontsize=9)
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

            dpx = i1 - i0
            nm = self._px_to_nm(dpx)
            dgray = float(data[i1]) - float(data[i0])

            extra = f"  Δ={dpx}px"
            if nm is not None:
                extra += f" ≈ {nm:.2f}nm"
            extra += f"  Δgray={dgray:+.0f}"
            ax.set_title(title + extra, fontsize=10)
        else:
            ax.set_title(title, fontsize=10)

        self._line_profile_fig.tight_layout()
        self._line_profile_canvas.draw()
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

        parts = [self._profile_title(), f"长度 {len(data)}px"]

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
            dpx = i1 - i0
            nm = self._px_to_nm(dpx)
            p0 = self._profile_index_to_xy(i0)
            p1 = self._profile_index_to_xy(i1)

            txt = f"量取 Δ={dpx}px"
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
            p = float(self.scale_vars["SCALE_PIXELS"].get())
            nm = float(self.scale_vars["SCALE_NM"].get())
            if p <= 0:
                return None
            return px * nm / p
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

    def _persist_gui_target_report(self, kind, data, report):
        """把右侧统计块当前 Target 同步写回对应输出文件。

        GUI 允许工具结果与 ROI 结果各自设置 Target。若只刷新 Treeview，
        屏幕上的 Judge 会和磁盘文件不一致；这里在防抖回调中轻量重写
        stats/target_check，两份结果始终可追溯。
        """
        if not isinstance(data, dict) or not isinstance(report, dict):
            return

        detection = data.get("detection")
        if isinstance(detection, dict) and "detection" not in report:
            report["detection"] = detection
        data["target"] = report

        payload = self._analysis_cache.get(kind) or {}
        stats_path = payload.get("stats_path") or data.get("stats_path")
        if not stats_path:
            return
        stats_path = os.path.abspath(str(stats_path))
        target_path = data.get("target_path") or payload.get("target_path")
        if not target_path:
            stem, _ = os.path.splitext(stats_path)
            suffix = "_stats_nm"
            if stem.endswith(suffix):
                target_path = stem[:-len(suffix)] + "_target_check.txt"
            else:
                target_path = stem + "_target_check.txt"
        try:
            core.write_stats_file(stats_path, data)
            core.write_target_file(str(target_path), report)
            data["target_path"] = str(target_path)
        except Exception:
            # 输出失败不应阻塞 GUI；状态栏会继续显示内存中的最新结果。
            return

    def _populate_stats_tree(self, kind, data, persist_target=False):
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
            f"{len(cols)} 根柱；统计直接使用原始浮点轮廓 × nm/px"
        )

        # ---- 最后两行：Profile（单形貌）+ 方向性 Judge（Profile/Target %）----
        target = self._target_dict(kind)
        report = core.build_target_report(data, target)
        if persist_target:
            self._persist_gui_target_report(kind, data, report)

        def _fmt_score(s):
            if s is None:
                return "-"
            try:
                return f"{float(s):.2f}"
            except Exception:
                return "-"

        avg = report.get("avg") or {}
        row_profile = [avg.get("profile", "SQUARE")]
        row_judge = [_fmt_score(avg.get("score"))]
        # 标签行文本（不带括号说明）

        for r in report.get("rows", []):
            row_profile.append(r.get("profile", "SQUARE"))
            row_judge.append(_fmt_score(r.get("score")))

        tree.tag_configure("profile", foreground="#0055cc")
        tree.tag_configure("judge", foreground="#0055cc")

        tree.insert(
            "", "end", text="Profile", values=row_profile, tags=("profile",)
        )
        tree.insert(
            "", "end", text="Judge", values=row_judge, tags=("judge",)
        )

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
        text = (
            "识别流程（新版：亮度 + 灰度斜率 + 结构 + 纵向扫描）\n\n"
            "1. TIFF → 8bit 灰度归一化\n"
            "2. CLAHE + 中值 + 高斯降噪\n"
            "3. Sobel-X / Sobel-Y 计算灰度变化斜率\n"
            "4. 固定阈值 / Otsu / 自适应阈值按图像质量自动选择\n"
            "5. 小核开运算 + 程序自适应闭运算\n"
            "6. y 投影找柱体高度平台\n"
            "7. 横向结构投影 = 亮度投影 + Sobel-X 梯度投影\n"
            "   —— 横向压缩：快速看出柱子在哪些 x 位置、底部大致范围\n"
            "8. 结构投影谷值分柱\n"
            "9. 每根柱纵向逐像素扫描（逐行追踪左右边界）\n"
            "   —— 得到 CD(y)：柱子的排列、间距、宽度随高度的变化\n"
            "10. 几何过滤\n"
            "11. 智能后处理（纵向扫描结果驱动）\n"
            "    · footing 台阶修剪：底边回到柱体主体，不被基座拉长\n"
            "    · 粘连检测：相邻柱间隙过小时给出提示\n"
            "12. 原始浮点轮廓直接统计 nm 尺寸 / 角度 / LER\n"
            "13. 形貌判定：CD(y) → Vertical / T-top / Undercut / Footing /\n"
            "    Taper / Irregular（详见“形貌判定说明”）\n"
            "14. 标准 Target 方波对比：PASS/FAIL + Profile/Target (%) Judge\n"
            "    100=等宽；>100=Profile 偏宽；<100=Profile 偏窄\n\n"
            "三种“看灰度”的方式各司其职：\n"
            "  · 横向压缩（列投影）      → 柱子在哪、底部大致范围\n"
            "  · 横向逐像素灰度曲线      → 某一行上柱子的排列与间距\n"
            "  · 纵向逐像素扫描 CD(y)    → 分柱 + 形貌 + footing + 侧壁质量\n\n"
            "重点：不同图之间整体灰度变化时，不再只依赖一个固定二值阈值。"
        )
        self._show_doc("识别流程", text)

    def _show_params_help(self):
        lines = ["■ 左侧“识别参数”（改动后会自动防抖重识别）\n"]
        for attr, name, *_ in MAIN_PARAMS:
            lines.append(
                f"■ {name} [{attr}]\n"
                f"  默认：{getattr(core, attr)}\n"
                f"  {PARAM_DESC.get(attr, '')}\n"
            )

        lines.append(
            "■ 自动重试（内部策略）\n"
            f"  候选库包括原图、反相、增强、平滑、均衡化和局部对比度版本；"
            f"程序会在最多 {int(core.AUTO_RETRY_MAX)} 个候选内统一评分并选最优结果；"
            f"  置信度是候选排序分数（不是百分比）；{float(core.AUTO_RETRY_MIN_CONFIDENCE):g}"
            " 是低置信度提示阈值。状态栏会显示候选版本、分数和实际尝试次数。\n"
            "  已知大致线宽时可填例如 82±5（左右边界距离）；不确定时把估计 Line CD 填 0，"
            "程序会从候选柱的中位宽自动估计。输出文件会区分用户先验范围、"
            "候选保护范围和边缘残柱范围。\n"
        )

        lines.append(
            "■ 右侧“标准 Target”面板里的参数（只重算统计与判定，不重新识别）\n"
            f"  Target CD (nm)：目标线宽，默认 {core.TARGET_CD_NM:g}\n"
            f"  容差 ±(nm)：绝对容差，默认 {core.TARGET_TOL_NM:g}\n"
            f"  容差 ±(%)：相对容差，默认 {core.TARGET_TOL_PCT:g}（与绝对值取大者）\n"
            f"  目标高度 (nm)：默认 {core.TARGET_HEIGHT_NM:g}，填 0 表示不判断高度\n"
            f"  顶部展宽≥：T-top 判据，默认 {core.SHAPE_TOP_RATIO:g}\n"
            f"  底部展宽≥：Footing 判据，默认 {core.SHAPE_FOOTING_RATIO:g}\n"
            f"  腰部收缩≤：Undercut 判据，默认 {core.SHAPE_WAIST_RATIO:g}\n"
            "  Judge：mean(W(y)) / Target CD × 100%；100=等宽，>100=Profile 偏宽，<100=Profile 偏窄\n"
            "  Judge 只在启用 Target 且曲线有效时计算；未启用时显示 “-”。\n"
        )

        self._show_doc("参数说明", "\n".join(lines))

    def _show_about(self):
        self._show_doc("关于", ABOUT_HELP)

    def _show_doc(self, title, text):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("760x580")

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        box = tk.Text(
            frame,
            wrap="word",
            bg="#fbfbfb",
            font=("Microsoft YaHei UI", 10)
        )
        sb = ttk.Scrollbar(frame, orient="vertical", command=box.yview)
        box.configure(yscrollcommand=sb.set)

        box.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        box.insert("1.0", text)
        box.configure(state="disabled")

    # -----------------------------------------------------------------
    # Queue
    # -----------------------------------------------------------------

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

                    # 运行中改过识别参数 -> 补跑最后一版
                    if self._live_dirty:
                        self._live_dirty = False
                        self.after(80, self._run_live)

                    # 若只是比例尺在运行中变化，也补一次轻量重导出
                    elif self._scale_dirty:
                        self._scale_dirty = False
                        self.after(80, self._run_scale_refresh)

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

                    # 比例尺重算期间也可能收到识别参数/比例尺修改。
                    # 若只更新状态而不消费 dirty 标记，用户会看到旧轮廓，
                    # 直到下一次手动点击才恢复一致；按普通分析完成路径
                    # 补跑最后一版，且优先处理识别参数变更。
                    if self._live_dirty:
                        self._live_dirty = False
                        self.after(80, self._run_live)
                    elif self._scale_dirty:
                        self._scale_dirty = False
                        self.after(80, self._run_scale_refresh)

        except queue.Empty:
            pass

        self.after(80, self._drain_queue)

    # -----------------------------------------------------------------
    # Misc
    # -----------------------------------------------------------------

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

    def _on_close(self):
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

    try:
        if not os.path.isfile(probe):
            lines.append("probe image not found; algorithm selftest skipped")
        else:
            payload = process_image(probe)
            lines.append(
                f"process_image ok: n_col={len(payload['columns'])}, "
                f"detection={payload.get('detection_mode', 'unknown')}, "
                f"variant={payload.get('detection_variant', 'unknown')}, "
                f"confidence={float(payload.get('detection_confidence', -1.0)):.3f}, "
                f"attempts={int(payload.get('detection_attempts', 0))}, "
                f"threshold={payload['threshold_mode']}, "
                f"cd_range={payload.get('cd_range_px')}"
            )
    except Exception as e:
        lines.append(f"FAILED: {e}")
        lines.append(traceback.format_exc())

    out = os.path.join(out_dir, "selftest_ok.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def gui_main():
    if "--selftest" in sys.argv[1:]:
        run_selftest()
        return

    App().mainloop()


if __name__ == "__main__":
    gui_main()
