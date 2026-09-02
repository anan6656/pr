# -*- coding: utf-8 -*-
"""
PR 柱子自动识别与测量工具 —— 单文件版（GUI）

本文件已内置全部分析核心，不再依赖其他 .py：
  · 直接运行：python pr2_gui.py
  · 打包单 exe：PyInstaller --onefile --windowed pr2_gui.py

布局：左=参数 | 中=原始图(框选/直线量测)+识别涂色图 | 右=统计分析表 | 底部=进度行
生成文件命名：工具自动整图=_tool；用户框选套索=_user。
"""
import os
import sys
import glob
import math
import queue
import threading
import traceback
import contextlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2

# ====================================================================
# 分析核心（原 pr2_analysis.py，已并入本文件）
# ====================================================================
# ========== 可调参数 ==========

# --- 输入/输出 ---
INPUT_DIR = os.path.dirname(os.path.abspath(__file__))  # 输入/输出根目录：自动扫描该目录下的 tif/tiff 图像
INPUT_GLOBS = ('*.tif', '*.tiff')                        # 识别的图像扩展名（TIF 系列）
OUTPUT_SUBDIR = True                                     # True: 每张图建立「xx.tif 同名」子文件夹存放结果

# --- 二值化与边缘 ---
THRESHOLD = 150    # 二值化阈值：灰度 > THRESHOLD 视为亮（柱子区域）。若固定阈值结果异常（全黑/全白），自动改用 Otsu
EDGE_MARGIN = 20   # 忽略图像四边边缘像素数：将四边置黑，避免图像边框/标尺线干扰识别
MORPH_CLOSE_K = 3  # 闭运算核大小（应为奇数）：填充柱体内部小孔/毛刺；柱间窄谷宽度 > 核大小，不受影响
DEBUG_OUTPUT = False  # True 时额外保存二值化中间结果图，便于批量调试阈值

# --- 投影与区域划分 ---
SMOOTH_KERNEL = 21   # 列/行投影平滑核大小（应为奇数），越大曲线越平滑、谷值越少
VALLEY_RATIO = 0.25  # 列投影中低于「最大值 * 该比例」的区域视为谷值/凹槽（用于分隔柱子）
COL_PROFILE_Y0 = 0.1  # 列投影取图的上起始位置比例（相对图像高度，避开顶部噪声）
COL_PROFILE_Y1 = 0.8  # 列投影取图的下结束位置比例（相对图像高度，避开底部基座）

# --- 柱子有效性过滤 ---
MIN_COLUMN_HEIGHT = 20  # 柱子最小高度（像素），低于则视为噪声丢弃
MIN_AVG_WIDTH = 15      # 柱子平均宽度下限（像素），过滤文字/窄条等杂散区域
MIN_ASPECT_RATIO = 1.2  # 最小高宽比：高度 < 宽度*该值 视为扁宽噪声块（基底亮斑等）丢弃
BOTTOM_OFFSET = 4       # 底部识别后再向下延伸的像素数（覆盖基座边缘毛刺）
WINDOW_MARGIN = 3       # 逐行扫描窗口外扩容差（像素）：封闭图形左右边缘应平缓变化
GLOBAL_Y_TOLERANCE = 5  # 全局柱体区 bottom 对单柱 bottom 的约束容差（像素）

# --- 比例尺换算（物理尺寸） ---
SCALE_PIXELS = 100.0  # 比例尺对应的像素总长度
SCALE_NM = 50.0       # 比例尺对应的实际长度（nm）
# 换算系数 = SCALE_NM / SCALE_PIXELS（每像素代表多少 nm），自动用于 nm 版图像与表格

# --- 绘图样式 ---
ALPHA = 0.15          # 柱子半透明填充透明度（0~1，越小越透明）
LINE_THICKNESS = 1    # 轮廓线宽（像素）
AXIS_MARGIN_L = 42    # 主图左侧白边（像素）：y 轴刻度数字空间
AXIS_MARGIN_B = 24    # 主图底部白边（像素）：x 轴刻度数字空间
PANEL_W = 90          # 右侧每根柱子 y 投影窗格的宽度（像素），越大越清晰
# 每根柱子的颜色（BGR 三元组），按检测顺序循环使用；超过颜色数时自动循环
PR_COLORS = [
    (255, 0, 0),    # 红
    (0, 200, 0),    # 绿
    (0, 0, 255),    # 蓝
    (255, 0, 255),  # 品红
    (255, 255, 0),  # 青
    (0, 165, 255),  # 橙
]


def preprocess_image(image_path):
    """
    读取图像并预处理，返回 (img_rgb, binary)。
    - 兼容 8/16-bit、灰度/彩色 tif：统一转为 8bit 灰度再转 BGR 供绘制
    - 高斯模糊 → 固定阈值（若亮像素占比异常则自动改用 Otsu）→ 闭运算 → 四边边缘置黑
    """
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
        raise ValueError(f"无法读取图片: {image_path}")
    # 彩色 → 灰度
    if img_raw.ndim == 3:
        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_raw
    # 16-bit 等 → 按 1%~99% 分位拉伸到 8bit（避免直接右移导致图像过暗）
    if gray.dtype != np.uint8:
        lo, hi = np.percentile(gray, (1, 99))
        gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 统一 8bit BGR，供绘图使用

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)
    bright_frac = binary.mean() / 255.0
    # 固定阈值失效保护：亮像素占比过低/过高说明阈值不适配当前图像，自动改用 Otsu
    if bright_frac < 0.01 or bright_frac > 0.9:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"[warn] 固定阈值(THRESHOLD={THRESHOLD})结果异常(亮像素占比={bright_frac:.2%})，已自动改用 Otsu")
    # 闭运算：填充柱体内部小孔/边缘毛刺
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), np.uint8))
    # 忽略四边边缘信息：避免图像边框/标尺线干扰
    binary[:EDGE_MARGIN, :] = 0
    binary[-EDGE_MARGIN:, :] = 0
    binary[:, :EDGE_MARGIN] = 0
    binary[:, -EDGE_MARGIN:] = 0
    return img_rgb, binary


def column_profile(binary, h, y_lo=None, y_hi=None):
    """
    整图 x=n 列投影。
    给定 y_lo/y_hi（全局柱体区）时只统计该区间，彻底避开底部基底亮区；
    否则按 COL_PROFILE_Y0/Y1 比例确定。返回平滑后的投影曲线。
    """
    if y_lo is None or y_hi is None:
        y_lo, y_hi = int(h * COL_PROFILE_Y0), int(h * COL_PROFILE_Y1)
    prof = np.sum(binary[y_lo:y_hi + 1, :], axis=0).astype(np.float32)
    return cv2.GaussianBlur(prof.reshape(1, -1), (1, SMOOTH_KERNEL), 0).flatten()


def find_valleys(profile, valley_ratio=VALLEY_RATIO, min_gap=20):
    """
    找列投影中的显著低谷区间（凹槽）。
    返回低谷区间列表 [(start_x, end_x), ...]。
    """
    maxv = np.max(profile)
    lim = maxv * valley_ratio
    low = profile < lim
    valleys = []
    in_v = False
    start = 0
    for x in range(len(profile)):
        if low[x] and not in_v:
            in_v = True
            start = x
        elif not low[x] and in_v:
            in_v = False
            if x - start >= 3:
                valleys.append((start, x - 1))
    if in_v and len(profile) - start >= 3:
        valleys.append((start, len(profile) - 1))
    # 合并过近的谷（避免窄凹槽）
    merged = []
    for v in valleys:
        if merged and v[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], v[1])
        else:
            merged.append(v)
    return merged


def valley_bottom(profile, valley):
    """谷区间的谷底位置（区间内列投影最小点）。"""
    s, e = valley
    return s + int(np.argmin(profile[s:e + 1]))


def split_columns_by_valleys(profile, w):
    """
    根据谷值把图像分成若干柱子区域，区域为「两个谷底之间的值」：
        [左边界, 谷底1], [谷底1, 谷底2], ..., [谷底N, 右边界]
    对每个区域，取区域内「最长连续亮列段」（柱体主体）裁剪左右边界并取中心列，
    避免区域边缘的窄亮条/文字干扰。
    返回 [(left, right, center_x), ...]。
    """
    valleys = find_valleys(profile)
    bottoms = [valley_bottom(profile, v) for v in valleys]
    regions = []
    if bottoms:
        regions.append((0, bottoms[0]))
        for i in range(len(bottoms) - 1):
            regions.append((bottoms[i], bottoms[i + 1]))
        regions.append((bottoms[-1], w - 1))
    else:
        regions.append((0, w - 1))
    # 不做最小宽度限制；跳过无亮像素的空区域
    cols = []
    for left, right in regions:
        seg = profile[left:right + 1]
        if seg.max() <= 0:
            continue
        # 区域内最长连续亮列段（柱体主体）
        bright = seg > 0
        best_s, best_len = 0, 0
        s = None
        for i, b in enumerate(bright):
            if b and s is None:
                s = i
            elif not b and s is not None:
                if i - s > best_len:
                    best_s, best_len = s, i - s
                s = None
        if s is not None and len(bright) - s > best_len:
            best_s, best_len = s, len(bright) - s
        if best_len < 1:
            continue
        # 裁剪区域到柱体主体，center_x 取主体中点
        left += best_s
        right = left + best_len - 1
        cx = left + best_len // 2
        cols.append((left, right, cx))
    return cols


def bright_segments(row, lo, hi):
    """返回二进制行 row 在 [lo, hi] 范围内的连续亮段列表 [(s, e), ...]。"""
    segs = []
    in_seg = False
    s = lo
    for x in range(lo, hi + 1):
        if row[x] == 255 and not in_seg:
            in_seg = True
            s = x
        elif row[x] != 255 and in_seg:
            in_seg = False
            segs.append((s, x - 1))
    if in_seg:
        segs.append((s, hi))
    return segs


def seg_overlap(a, b):
    """两个 [s, e] 段的水平重叠长度（用于选择与上一行连通的段）。"""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def trace_column_ranges(binary, left, right, y_top, y_bottom):
    """
    PR 是封闭图形，轮廓内全部属于该柱子。
    从下向上逐行扫描：以上一行 [s, e] 边界为窗口（外扩 WINDOW_MARGIN 像素）
    在当前行找亮段，并选择与上一行重叠最大的段，保证轮廓连续；
    与柱体不连通的基底亮暗噪声因不在窗口内而被自然隔离。
    返回 ranges（按 y 升序）。
    """
    ranges = []
    prev_s, prev_e = left, right  # 起始窗口：整列区域（底部最宽处）
    for y in range(y_bottom, y_top - 1, -1):
        row = binary[y, :]
        wl = max(left, prev_s - WINDOW_MARGIN)
        wr = min(right, prev_e + WINDOW_MARGIN)
        segs = bright_segments(row, wl, wr)
        if not segs:
            continue  # 窗口内无亮（柱体内部有孔/断行）：保持窗口继续向上
        # 选择与上一行重叠最大的段（保持连通）
        s, e = max(segs, key=lambda seg: seg_overlap(seg, (prev_s, prev_e)))
        prev_s, prev_e = s, e
        if e - s + 1 >= 5:
            ranges.append((y, s, e))
    ranges.reverse()  # 转为从上到下
    return ranges


def filter_ranges_by_width(ranges, sigma=2.0):
    """
    根据行宽度剔除离群行（通常是底部基座/顶部噪声）。
    使用 MAD 作为稳健离散度估计，只保留最长连续主体段。
    """
    if len(ranges) < 3:
        return ranges
    widths = np.array([e - s + 1 for _, s, e in ranges])
    med = np.median(widths)
    mad = np.median(np.abs(widths - med)) or 1.0
    threshold = med + sigma * 1.4826 * mad
    filtered = [r for r, w in zip(ranges, widths) if w <= threshold]
    # 保留最长连续段（防止中间空洞）
    groups = []
    cur = []
    for r in filtered:
        if not cur or r[0] == cur[-1][0] + 1:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    if cur:
        groups.append(cur)
    return max(groups, key=len)


def find_column_y_extent(binary, left, right):
    """
    用该柱子 x 区域内的行投影（y=n）确定独立的 top/bottom。
    原理：柱子主体的行投影相对稳定，底部基座比柱体更宽，投影会明显升高。
    - top：投影超过 0.15*max 的最长连续段起点
    - bottom：从段底自下而上找「投影 ≥ 中位数×1.35 的连续基座段」，
      柱体底部取基座顶端上一行（无基座时保持段底，不会误缩）。
    返回 (y_top, y_bottom)，并返回平滑后的 y 投影供绘制。
    """
    region = binary[:, left:right + 1]
    row_prof = np.sum(region, axis=1).astype(np.float32)
    row_prof_smooth = cv2.GaussianBlur(row_prof.reshape(-1, 1), (SMOOTH_KERNEL, 1), 0).flatten()
    maxv = row_prof_smooth.max()
    if maxv <= 0:
        return 0, 0, row_prof_smooth
    body = row_prof_smooth > maxv * 0.15
    segs = []
    cur = []
    for y, b in enumerate(body):
        if b:
            cur.append(y)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    if not segs:
        return 0, 0, row_prof_smooth
    seg = max(segs, key=len)
    y_top = seg[0]
    y_bot_raw = seg[-1]
    # 底部：柱体主体行投影近似恒定（中位数 med），底部基座明显更宽 → 投影显著升高。
    # 从下往上找"连续显著高于 med"的基座段顶端；柱体底部取该段顶端上一行。
    # 若无基座（等宽柱体），整段都不超阈值，bottom 保持为段底，不会误缩。
    med = np.median(row_prof_smooth[y_top:y_bot_raw + 1])
    thr = med * 1.35  # 基座比柱体宽 ≥35% 视为抬升
    y_bottom = y_bot_raw
    for y in range(y_bot_raw, y_top - 1, -1):
        if row_prof_smooth[y] >= thr:
            y_bottom = y  # 记录基座段最顶端行
        else:
            break
    if y_bottom < y_bot_raw:  # 确实检测到基座：柱体底部 = 基座顶端上一行
        y_bottom -= 1
    if y_bottom < y_top:
        y_bottom = y_top
    return y_top, y_bottom, row_prof_smooth


def find_global_y_extent(binary):
    """
    全局观察整图 y=n 行投影曲线：
    顶部背景暗（低值）、底部基底亮（高值）、PR 柱子处于「中间亮度平台」。
    取该中间平台的最长连续段作为柱体所在全局 y 区间 (g_top, g_bottom)，
    用于在单柱细化前先排除底部基底亮区。
    返回 (g_top, g_bottom)；未识别到平台时返回 (0, 0)。
    """
    prof = np.sum(binary, axis=1).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (SMOOTH_KERNEL, 1), 0).flatten()
    maxv = prof.max()
    if maxv <= 0:
        return 0, 0
    # 1) 亮区（> 0.15*max）最长连续段：柱体 + 基座的整体范围
    body = prof > maxv * 0.15
    segs = []
    cur = []
    for y, b in enumerate(body):
        if b:
            cur.append(y)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    if not segs:
        return 0, 0
    main = max(segs, key=len)
    # 2) 柱体平台的典型亮度：主体段行投影中位数
    med = np.median(prof[main[0]:main[-1] + 1])
    # 3) 中间亮度平台：投影在 [0.5*med, 1.5*med] 的行（柱体区；基座/背景被排除）
    plat = (prof >= 0.5 * med) & (prof <= 1.5 * med)
    best_s = best_e = -1
    s = None
    for y, b in enumerate(plat):
        if b and s is None:
            s = y
        elif not b and s is not None:
            if best_s < 0 or (y - 1 - s) > (best_e - best_s):
                best_s, best_e = s, y - 1
            s = None
    if s is not None and (best_s < 0 or (len(plat) - 1 - s) > (best_e - best_s)):
        best_s, best_e = s, len(plat) - 1
    if best_s < 0:
        return 0, 0
    return best_s, best_e


def build_contour(ranges):
    """把每行 (y, s, e) 构建成闭合轮廓。"""
    ys = np.array([r[0] for r in ranges])
    xs = np.array([r[1] for r in ranges])
    xe = np.array([r[2] for r in ranges])
    left = np.column_stack((xs, ys))
    right = np.column_stack((xe, ys))[::-1]
    return np.vstack([left, right]).astype(np.int32).reshape((-1, 1, 2))


def detect_one_column(binary, left, right, cx, h, g_top=0, g_bottom=0):
    """
    检测单根柱子：
    行投影确定独立 top/bottom → 全局柱体区约束（忽略底部基底亮区）→
    逐行 y=n 扫描 → 宽度离群过滤 → 底部延伸 → 尺寸过滤。
    返回 columns 字典；不合格返回 None。
    """
    y_top, y_bottom, y_proj = find_column_y_extent(binary, left, right)
    # 柱顶必须在图像上半部（排除底部噪声/文字被误判）
    if y_top >= int(h * 0.5):
        return None
    # 全局中间亮度平台（柱体区）约束 bottom：忽略底部基底亮区。
    # g_top 不约束（每根柱子保留独立 top）；bottom 允许少量容差，防止平台段
    # 被单柱孔洞/文字截短而误伤正常底部。
    if g_bottom > 0 and y_bottom > g_bottom + GLOBAL_Y_TOLERANCE:
        y_bottom = g_bottom + GLOBAL_Y_TOLERANCE
    if y_bottom < y_top:
        y_bottom = y_top
    ranges = trace_column_ranges(binary, left, right, y_top, y_bottom)
    ranges = filter_ranges_by_width(ranges)
    # 底部向下延伸几个像素（覆盖基座边缘毛刺）
    if ranges:
        last_y, s, e = ranges[-1]
        for dy in range(1, BOTTOM_OFFSET + 1):
            ranges.append((last_y + dy, s, e))
    if len(ranges) < MIN_COLUMN_HEIGHT:
        return None
    widths = [e - s + 1 for _, s, e in ranges]
    if np.mean(widths) < MIN_AVG_WIDTH:
        return None
    # 高宽比过滤：PR 是竖长柱体，扁宽块（基底亮斑/横条）不可能是柱
    height = max(r[0] for r in ranges) - min(r[0] for r in ranges) + 1
    if height < np.mean(widths) * MIN_ASPECT_RATIO:
        return None
    return {
        'left': left,
        'right': right,
        'center_x': cx,
        'y_top': y_top,
        'y_bottom': max(r[0] for r in ranges),  # 过滤后的实际底部（与轮廓一致）
        'y_proj': y_proj,
        'ranges': ranges,
        'contour': build_contour(ranges),
    }


def draw_axis_with_ticks(img, tick_step=50, nm_per_px=1.0):
    """
    在图像外侧绘制黑色坐标轴（刻度朝外）：
    左侧白边放 y 轴刻度，底部白边放 x 轴刻度，轴线紧贴图像边缘。
    nm_per_px=1.0 时显示像素坐标；否则显示物理尺寸（nm，保留 1 位小数）。
    返回的画布尺寸为 (h + AXIS_MARGIN_B, w + AXIS_MARGIN_L)。
    """
    h, w = img.shape[:2]
    out = np.full((h + AXIS_MARGIN_B, w + AXIS_MARGIN_L, 3), 255, dtype=np.uint8)
    out[0:h, AXIS_MARGIN_L:AXIS_MARGIN_L + w] = img  # 原图放画布右上，左/下留白边
    unit = 'nm' if nm_per_px != 1.0 else 'px'

    def fmt(val):
        return f"{val * nm_per_px:.1f}" if nm_per_px != 1.0 else str(val)

    # x 轴：轴线沿图像底边，刻度短线向下（向外），数字在白边内
    y_axis = h - 1  # 原图底边（out 中的行号）
    cv2.line(out, (AXIS_MARGIN_L, y_axis), (AXIS_MARGIN_L + w - 1, y_axis), (0, 0, 0), 1)
    for x in range(0, w, tick_step):
        px = AXIS_MARGIN_L + x
        cv2.line(out, (px, y_axis), (px, y_axis + 6), (0, 0, 0), 1)
        cv2.putText(out, fmt(x), (px + 2, y_axis + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    # y 轴：轴线沿图像左边，刻度短线向左（向外），数字在白边内
    x_axis = AXIS_MARGIN_L  # 原图左边（out 中的列号）
    cv2.line(out, (x_axis, 0), (x_axis, h - 1), (0, 0, 0), 1)
    for y in range(0, h, tick_step):
        cv2.line(out, (x_axis, y), (x_axis - 6, y), (0, 0, 0), 1)
        cv2.putText(out, fmt(y), (x_axis - 34, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    # 单位标注（轴端，黑色）
    cv2.putText(out, unit, (AXIS_MARGIN_L + w - 40, y_axis + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(out, unit, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return out


def draw_1d_profile_vertical(data, width, height, color=(255, 0, 0), fill_color=(200, 200, 255), title=""):
    """
    绘制一维曲线图（水平方向为数据索引，垂直方向为数值）。
    返回 BGR 图像 (height, width, 3)。
    """
    data = np.array(data, dtype=np.float32)
    if data.max() == data.min():
        data_norm = np.zeros_like(data)
    else:
        data_norm = (data - data.min()) / (data.max() - data.min())
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    n = len(data_norm)
    xs = np.linspace(0, width - 1, n).astype(np.int32)
    # 曲线整体下移：最高点距标题条留 10px 空隙（标题条高 26），最低点距底边 21px
    ys = (height - 1 - data_norm * (height - 57) - 20).astype(np.int32)
    # 填充
    pts = np.column_stack((xs, ys))
    pts_bottom = np.array([[xs[0], height - 1], [xs[-1], height - 1]])
    cv2.fillPoly(canvas, [np.vstack([pts, pts_bottom[::-1]]).astype(np.int32)], fill_color)
    # 曲线
    for i in range(n - 1):
        cv2.line(canvas, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), color, 1, cv2.LINE_AA)
    # 标题（灰色背景条，加粗文字，增强可读性；标题位置不动）
    if title:
        cv2.rectangle(canvas, (0, 0), (width - 1, 26), (200, 200, 200), -1)
        cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    # 边框（蓝色）
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (255, 0, 0), 1)
    return canvas


def draw_1d_profile_horizontal(data, width, height, color=(255, 0, 0), fill_color=(200, 255, 200), title=""):
    """
    绘制一维曲线图（垂直方向为数据索引，水平方向为数值）。
    返回 BGR 图像 (height, width, 3)。
    """
    data = np.array(data, dtype=np.float32)
    if data.max() == data.min():
        data_norm = np.zeros_like(data)
    else:
        data_norm = (data - data.min()) / (data.max() - data.min())
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    n = len(data_norm)
    ys = np.linspace(0, height - 1, n).astype(np.int32)
    xs = (5 + data_norm * (width - 30)).astype(np.int32)
    pts = np.column_stack((xs, ys))
    pts_left = np.array([[0, ys[0]], [0, ys[-1]]])
    cv2.fillPoly(canvas, [np.vstack([pts, pts_left[::-1]]).astype(np.int32)], fill_color)
    for i in range(n - 1):
        cv2.line(canvas, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), color, 1, cv2.LINE_AA)
    # 标题（灰色背景条，加粗文字，增强可读性）
    if title:
        cv2.rectangle(canvas, (0, 0), (width - 1, 26), (200, 200, 200), -1)
        cv2.putText(canvas, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    # 边框（蓝色）
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (255, 0, 0), 1)
    return canvas


def _cjk_width(s):
    """按等宽字体统计显示宽度（中日韩字符按 2 个半角字符计）。"""
    return sum(2 if ord(ch) > 0x7F else 1 for ch in s)


def write_aligned_table(txt_path, rows, left_first_text=False, footer=None):
    """
    把 rows（二维字符串表）写入 txt，保证「列对齐」：
    每一列宽度统一为该列最长内容的显示宽度，数字列靠右对齐（个位/末位数字
    严格上下对齐），第一列文字可选左对齐。列之间用 2 个空格分隔。
    footer：末尾追加的说明行（以 # 开头），不参与对齐。
    """
    if not rows:
        open(txt_path, 'w', encoding='utf-8').close()
        return
    ncols = max(len(r) for r in rows)
    widths = [0] * ncols
    for r in rows:
        for c, t in enumerate(r):
            if c < ncols:
                widths[c] = max(widths[c], _cjk_width(t))
    lines = []
    for r in rows:
        cells = []
        for c in range(ncols):
            t = r[c] if c < len(r) else ''
            pad = ' ' * (widths[c] - _cjk_width(t))
            cells.append((t + pad) if (left_first_text and c == 0) else (pad + t))
        lines.append('  '.join(cells).rstrip())
    if footer:
        lines.append(footer)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_txt(txt_path, columns, nm_per_px=1.0):
    """
    导出 txt 表格：y | PR1(xs,xe) | PR2 | ...
    nm_per_px=1.0 时输出像素坐标；否则输出物理尺寸（nm，保留 1 位小数）。
    列宽自动按内容对齐，数字列末位严格上下对齐。
    """
    unit = 'nm' if nm_per_px != 1.0 else 'px'

    def fmt(val):
        return f"{val * nm_per_px:.1f}" if nm_per_px != 1.0 else str(val)

    all_y = sorted({r[0] for col in columns for r in col['ranges']})
    col_index = {c['label']: {r[0]: (r[1], r[2]) for r in c['ranges']} for c in columns}
    header = [f'y({unit})'] + [f"PR{c['label']}(xs,xe)" for c in columns]
    rows = [header]
    for y in all_y:
        row = [fmt(y)]
        for col in columns:
            seg = col_index[col['label']].get(y)
            row.append(f"({fmt(seg[0])},{fmt(seg[1])})" if seg else "-")
        rows.append(row)
    write_aligned_table(txt_path, rows,
                        footer=f"# 列格式=(x_start,x_end); 单位={unit}; '-'=该y处该柱无轮廓")


# ============================================================================
# 边界表统计分析（读取 *_boundaries_nm.txt，输出 *_stats_nm.txt）
#
# 口径说明：
#   · 边界表每行对应柱体轮廓的一条水平切片 y：(x_start, x_end)。
#   · 把每根柱子按高度平均分成 上/中/下 三段（按行数三等分）。
#   · CD = 切片宽度 (x_end - x_start)，某段平均 CD = 该段所有切片宽度的平均。
#   · 侧壁角：每根柱左/右边缘“沿柱壁自顶向底（y 增大方向）”与图像 +x 轴
#     （向右）的夹角（0~180°）：竖直侧壁 = 90°，向右侧倾 < 90°，
#     向左侧倾 > 90°（可大于 90°）。行数足够时按段内边缘回归斜率计算，
#     行数少时退回段首/段尾端点法。左、右两侧分别给出。
#   · 边缘粗糙度 LER：对整根柱的左右边缘分别做“水平坐标 ~ y”的直线拟合，
#     取残差的 3σ（3 倍标准差），单位 nm。
# ============================================================================

def parse_boundaries_table(txt_path):
    """
    解析 *_boundaries.txt / *_boundaries_nm.txt。
    返回 dict：{'columns': [(标签, [(y, x_start, x_end), ...]), ...]}
    每列行按 y 升序；文件名含 "nm" 时数值视为 nm。
    """
    cols = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    if not lines:
        return {'columns': []}
    header = lines[0].split()   # 兼容旧版 Tab 分隔与新版空格对齐格式
    labels = []
    for cell in header[1:]:
        name = cell.split('(')[0].strip()
        labels.append(name or f'col{len(labels) + 1}')
    data = {lb: [] for lb in labels}
    for line in lines[1:]:
        if line.lstrip().startswith('#'):   # 跳过文件末尾的说明行
            continue
        cells = line.split()
        if not cells:
            continue
        try:
            y = float(cells[0])
        except ValueError:
            continue
        for i, lb in enumerate(labels):
            if i + 1 >= len(cells):
                continue
            seg = cells[i + 1].strip()
            if seg.startswith('(') and seg.endswith(')'):
                parts = seg[1:-1].split(',')
                try:
                    data[lb].append((y, float(parts[0]), float(parts[1])))
                except (ValueError, IndexError):
                    pass
    for lb in labels:
        data[lb].sort(key=lambda r: r[0])
        cols.append((lb, data[lb]))
    return {'columns': cols}


def _split_three(rows):
    """按行数把 rows（y 升序）平均分为上/中/下三段。返回三个行列表。"""
    n = len(rows)
    segs = [[], [], []]
    for i, r in enumerate(rows):
        segs[min(2, (i * 3) // n)].append(r)
    return segs


def _row_step(rows):
    """行间距（nm）：相邻两行 y 的最小正差；只有一行时返回 0。"""
    if len(rows) < 2:
        return 0.0
    diffs = [rows[i + 1][0] - rows[i][0] for i in range(len(rows) - 1)]
    return min(d for d in diffs if d > 0)


def _reg_slope(rows, right_side=False):
    """一侧边缘 (x ~ y) 的最小二乘斜率 dx/dy（nm/nm），用于角度与去趋势。"""
    ys = np.array([r[0] for r in rows], dtype=np.float64)
    xs = np.array([r[2] if right_side else r[1] for r in rows], dtype=np.float64)
    denom = float(np.sum((ys - ys.mean()) ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum((ys - ys.mean()) * (xs - xs.mean()))) / denom


def _side_angle_from_slope(b):
    """
    由边缘斜率 b=dx/dy（y 为向下增长的行坐标）求侧壁角：
    壁沿“y 增大（自顶向底）”的方向与图像 +x 轴(向右)的夹角，
    竖直壁 = 90°，向右倾 < 90°，向左倾 > 90°（0~180° 均可）。
    """
    if b is None or not math.isfinite(b):
        return None
    return math.degrees(math.atan2(1.0, b))


def _side_angle(x_top, x_bot, h):
    """
    端点法（行数少时用）：竖直分量 h，水平分量 (x_bot - x_top)，
    angle = atan2(h, x_bot - x_top)，同样允许 > 90°（>90 表示向左倾）。
    """
    if h <= 0:
        return None
    return math.degrees(math.atan2(h, x_bot - x_top))


def _edge_ler(rows, right_side=False):
    """一侧边缘的线粗糙度 LER（3σ，nm）：x ~ y 线性拟合残差的 3 倍标准差。"""
    n = len(rows)
    if n < 3:
        return None
    ys = np.array([r[0] for r in rows], dtype=np.float64)
    xs = np.array([r[2] if right_side else r[1] for r in rows], dtype=np.float64)
    b, a = np.polyfit(ys, xs, 1)  # x = a + b*y
    res = xs - (a + b * ys)
    return 3.0 * float(np.std(res))


def analyze_pr_column(rows):
    """
    对单根柱子（边界行列表）计算各项指标。
    返回 dict，键：
      height, cd_top/cd_mid/cd_bot, ratio_tm/ratio_td/ratio_md,
      angle_l_top/mid/bot, angle_r_top/mid/bot, ler_left, ler_right
    数值单位为 nm（角度为 °），无法计算时取 None。
    """
    out = {k: None for k in (
        'height', 'cd_top', 'cd_mid', 'cd_bot',
        'ratio_tm', 'ratio_td', 'ratio_md',
        'angle_l_top', 'angle_l_mid', 'angle_l_bot',
        'angle_r_top', 'angle_r_mid', 'angle_r_bot',
        'ler_left', 'ler_right')}
    if not rows:
        return out
    step = _row_step(rows)
    out['height'] = (rows[-1][0] - rows[0][0]) + step
    segs = _split_three(rows)

    def mean_cd(seg):
        if not seg:
            return None
        return sum(xe - xs for _, xs, xe in seg) / len(seg)

    out['cd_top'], out['cd_mid'], out['cd_bot'] = [mean_cd(s) for s in segs]

    def ratio(a, b):
        if a is None or b is None or b == 0:
            return None
        return a / b

    out['ratio_tm'] = ratio(out['cd_top'], out['cd_mid'])
    out['ratio_td'] = ratio(out['cd_top'], out['cd_bot'])
    out['ratio_md'] = ratio(out['cd_mid'], out['cd_bot'])

    def seg_angle(seg):
        """段侧壁角：行数足够时用段内整列回归斜率（抗噪声），否则退回端点法。"""
        if len(seg) < 2:
            return None, None
        if len(seg) >= 4:
            return (_side_angle_from_slope(_reg_slope(seg, False)),
                    _side_angle_from_slope(_reg_slope(seg, True)))
        h = (seg[-1][0] - seg[0][0]) + step
        return (_side_angle(seg[0][1], seg[-1][1], h),
                _side_angle(seg[0][2], seg[-1][2], h))

    for key, seg in zip(('top', 'mid', 'bot'), segs):
        al, ar = seg_angle(seg)
        out[f'angle_l_{key}'] = al
        out[f'angle_r_{key}'] = ar

    out['ler_left'] = _edge_ler(rows, right_side=False)
    out['ler_right'] = _edge_ler(rows, right_side=True)
    return out


# 统计项描述（顺序即表格行顺序）；'unit' 仅用于 GUI 提示
STAT_ROWS = [
    ('height',        '高度 (nm)'),
    ('cd_top',        '上段平均 CD (nm)'),
    ('cd_mid',        '中段平均 CD (nm)'),
    ('cd_bot',        '下段平均 CD (nm)'),
    ('ratio_tm',      'CD 比值 上/中'),
    ('ratio_td',      'CD 比值 上/下'),
    ('ratio_md',      'CD 比值 中/下'),
    ('angle_l_top',   '上段侧壁角 左 (°)'),
    ('angle_r_top',   '上段侧壁角 右 (°)'),
    ('angle_l_mid',   '中段侧壁角 左 (°)'),
    ('angle_r_mid',   '中段侧壁角 右 (°)'),
    ('angle_l_bot',   '下段侧壁角 左 (°)'),
    ('angle_r_bot',   '下段侧壁角 右 (°)'),
    ('ler_left',      '左缘粗糙度 LER 3σ (nm)'),
    ('ler_right',     '右缘粗糙度 LER 3σ (nm)'),
]


def _mean_finite(vals):
    finite = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def analyze_boundaries_file(txt_path, out_txt_path=None):
    """
    分析一张图的边界表（读取 *_boundaries_nm.txt），生成新的统计 txt：
      第一列=计算项描述，随后每列=一根 PR 柱，最后一列=所有柱平均。
    out_txt_path 为 None 时不写文件。
    返回 { 'desc', 'cols', 'rows', 'means' }，供 GUI 展示：
      desc:  各统计项描述（顺序）
      cols:  柱标签（如 ['PR1','PR2']）
      rows:  每统计项对各柱的值（含 None）
      means: 每统计项所有柱平均
    """
    parsed = parse_boundaries_table(txt_path)
    col_labels, col_rows = [], []
    for lb, rs in parsed['columns']:
        if rs:
            col_labels.append(lb)
            col_rows.append(rs)

    per_col = [analyze_pr_column(rs) for rs in col_rows]
    rows = []
    means = []
    for key, _desc in STAT_ROWS:
        vals = [d[key] for d in per_col]
        rows.append(vals)
        means.append(_mean_finite(vals))

    if out_txt_path is not None:
        def fmt(v):
            if v is None or not math.isfinite(v):
                return '-'
            return f'{v:.3f}'
        head = ['统计项'] + col_labels + ['全部柱平均']
        lines = [head]
        for (_, desc), vals, m in zip(STAT_ROWS, rows, means):
            lines.append([desc] + [fmt(v) for v in vals] + [fmt(m)])
        write_aligned_table(out_txt_path, lines, left_first_text=True)

    return {'desc': [d for _, d in STAT_ROWS],
            'cols': col_labels,
            'rows': rows,
            'means': means}


def render_main_image(img, columns, nm_per_px=1.0):
    """
    绘制「识别涂色主图」：原始图像 + 每根柱子半透明彩色填充 + 同色轮廓 +
    顶部 PR 标签 + 底部虚线，并加黑色坐标轴（刻度随 nm_per_px 换算）。
    返回 BGR 画布 (h + AXIS_MARGIN_B, w + AXIS_MARGIN_L, 3)。
    该函数同时被 render_combined 与 GUI 预览共用，保证两处画法一致。
    """
    h, w = img.shape[:2]
    main_img = img.copy()
    for col in columns:
        color = col['color']
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [col['contour']], -1, 255, -1)
        color_layer = np.full_like(img, np.array(color, dtype=np.uint8).reshape(1, 1, 3))
        fill = cv2.addWeighted(img, 1.0, color_layer, ALPHA, 0)
        main_img[m == 255] = fill[m == 255]
        cv2.drawContours(main_img, [col['contour']], -1, color, LINE_THICKNESS)
    main_img = draw_axis_with_ticks(main_img, tick_step=50, nm_per_px=nm_per_px)

    # 顶部标签 + 底部虚线
    # 注意：main_img 已被 draw_axis_with_ticks 扩出左侧/底部白边，
    # 原图内容整体右移 AXIS_MARGIN_L，故所有 x 坐标需加该偏移。
    xoff = AXIS_MARGIN_L
    for col in columns:
        color = col['color']
        top = min(r[0] for r in col['ranges'])
        bot = col['y_bottom']
        cx = col['center_x'] + xoff
        # 顶部标签（白底 + 彩色字，避免被背景干扰）
        label = f"PR{col['label']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        lx, ly = cx - tw // 2, max(top - 16, 12)
        cv2.rectangle(main_img, (lx - 4, ly - th - 2), (lx + tw + 4, ly + 3), (255, 255, 255), -1)
        cv2.putText(main_img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # 底部虚线（该柱子颜色，标注识别出的底部线）
        last = col['ranges'][-1]
        for xx in range(last[1] + xoff, last[2] + xoff + 1, 6):
            cv2.line(main_img, (xx, bot), (min(xx + 3, last[2] + xoff), bot), color, 1)
    return main_img


def render_combined(img, columns, col_profile_smooth, nm_per_px=1.0, main_img=None):
    """
    绘制综合结果图：主涂色图 + 下方整图 x=n 列投影 +
    右侧每根柱子独立 y=n 投影窗格（含 top/bottom 线标注）。
    nm_per_px=1.0 时坐标/标注显示像素；否则显示物理尺寸（nm，保留 1 位小数）。
    main_img：可选，外部已算好的主涂色图（避免重复绘制，供 GUI 预览复用）。
    """
    h, w = img.shape[:2]
    unit = 'nm' if nm_per_px != 1.0 else 'px'

    def fmt(val):
        return f"{val * nm_per_px:.1f}" if nm_per_px != 1.0 else str(val)

    # ---- 主涂色图（默认内部绘制，也可由外部传入复用） ----
    main_img = main_img if main_img is not None else render_main_image(img, columns, nm_per_px=nm_per_px)

    # ---- 下方 x=n 方向灰度投影图 ----
    profile_h = 120
    x_profile_img = draw_1d_profile_vertical(col_profile_smooth, w, profile_h,
                                             color=(255, 0, 0), fill_color=(220, 220, 255),
                                             title=f"x=n projection ({unit})")

    # ---- 右侧 y=n 方向灰度投影图：每根柱子独立窗格 ----
    panels = []
    for col in columns:
        p = draw_1d_profile_horizontal(col['y_proj'], PANEL_W, h,
                                       color=col['color'], fill_color=(245, 245, 245),
                                       title=f"y=n PR{col['label']}")
        yt = col['y_top']
        yb = col['y_bottom']
        # 顶部线（绿色虚线）与 top 数值
        for xx in range(0, PANEL_W, 8):
            cv2.line(p, (xx, yt), (min(xx + 4, PANEL_W - 1), yt), (0, 200, 0), 1)
        cv2.putText(p, f"top={fmt(yt)}{unit}", (3, max(yt + 14, 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        # 底部线（红色虚线）与 bottom 数值
        for xx in range(0, PANEL_W, 8):
            cv2.line(p, (xx, yb), (min(xx + 4, PANEL_W - 1), yb), (0, 0, 255), 1)
        cv2.putText(p, f"bottom={fmt(yb)}{unit}", (3, min(yb + 14, h - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        panels.append(p)
    y_profile_img = np.hstack(panels) if panels else np.full((h, PANEL_W, 3), 255, dtype=np.uint8)
    profile_w = y_profile_img.shape[1]

    # ---- 组合 ----
    mh, mw = main_img.shape[:2]  # 主图含左/下白边
    combined_w = mw + profile_w
    combined_h = mh + profile_h
    combined = np.full((combined_h, combined_w, 3), 255, dtype=np.uint8)
    combined[:mh, :mw] = main_img
    # 下方 x=n 投影：与主图内容列对齐（从左侧白边开始，宽为原图宽）
    combined[mh:mh + profile_h, AXIS_MARGIN_L:AXIS_MARGIN_L + w] = x_profile_img
    # 右侧 y=n 投影：与主图内容行对齐（高为原图高）
    combined[:h, mw:mw + profile_w] = y_profile_img
    # 分隔线
    cv2.line(combined, (mw, 0), (mw, combined_h - 1), (170, 170, 170), 1)
    cv2.line(combined, (0, mh), (combined_w - 1, mh), (170, 170, 170), 1)
    # 右下角空白区简短单位标注（避免长图例超出画布被截断）
    cv2.putText(combined, f"unit: {unit}",
                (mw + profile_w - 60, mh + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    return combined


def _detect_columns(img, binary):
    """
    公共检测步骤（整图与 ROI 框内共用）：在给定二值图上完成
     1. 全局 y 柱体区 + x=n 列投影谷值分割
     2. 每根柱子 y=n 横向扫描（独立 top/bottom）
    返回 (columns, col_profile_smooth)，坐标为「该图局部坐标」。
    """
    h, w = binary.shape[:2]
    g_top, g_bottom = find_global_y_extent(binary)
    print(f"[info] 检测区 y=({g_top},{g_bottom}) (全局柱体区=中间亮度平台)")
    # 列投影优先用全局柱体区（彻底避开基底亮区），识别失败时回退比例范围
    col_profile_smooth = column_profile(binary, h, g_top, g_bottom) if g_bottom > g_top else column_profile(binary, h)
    col_defs = split_columns_by_valleys(col_profile_smooth, w)
    print(f"[info] 列投影检测到 {len(col_defs)} 个潜在 PR 区域: {col_defs}")
    columns = []
    for left, right, cx in col_defs:
        col = detect_one_column(binary, left, right, cx, h, g_top, g_bottom)
        if col is None:
            continue
        col['label'] = len(columns) + 1
        col['color'] = PR_COLORS[(col['label'] - 1) % len(PR_COLORS)]
        columns.append(col)
    print(f"[info] 有效 PR 柱子数量: {len(columns)}")
    return columns, col_profile_smooth


def _offset_columns(columns, dx, dy):
    """把局部坐标检测结果平移 (dx, dy) 回整图坐标（轮廓同步重建）。"""
    for col in columns:
        col['left'] += dx
        col['right'] += dx
        col['center_x'] += dx
        col['y_top'] += dy
        col['y_bottom'] += dy
        col['ranges'] = [(y + dy, s + dx, e + dx) for (y, s, e) in col['ranges']]
        col['contour'] = build_contour(col['ranges'])
    return columns


def process_image(image_path, on_image=None):
    """
    处理单张 tif 图像：识别 PR 柱子并输出结果。
    结果按 INPUT_GLOBS 指定的扩展名图像建「同名子文件夹」存放。
    on_image：可选回调 on_image(img_rgb, main_img, file_name)，在识别完成后
    把「原始灰度转 BGR 图」与「识别涂色主图」交给外部（如 GUI 实时预览）。
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    # 输出目录：INPUT_DIR/<stem>/（每张图独立子文件夹），可由 OUTPUT_SUBDIR 关闭
    out_dir = os.path.join(INPUT_DIR, stem) if OUTPUT_SUBDIR else INPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'=' * 60}\n[info] 处理图像: {image_path}")

    # ---------- 0. 预处理：灰度/位深兼容 + 二值化 + 闭运算 + 四边边缘置黑 ----------
    img, binary = preprocess_image(image_path)
    h, w = img.shape[:2]
    if DEBUG_OUTPUT:
        cv2.imwrite(os.path.join(out_dir, f'{stem}_debug_binary.png'), binary)

    # ---------- 1~2. 检测（公共函数：整图分析与 ROI 框内分析共用） ----------
    columns, col_profile_smooth = _detect_columns(img, binary)

    # ---------- 3. 绘制综合结果图（像素版 + 物理尺寸 nm 版） ----------
    nm_per_px = SCALE_NM / SCALE_PIXELS
    # 像素版主涂色图：优先算一次，既用于输出也通过 on_image 交给 GUI 预览
    px_main = render_main_image(img, columns, nm_per_px=1.0)
    combined_px = render_combined(img, columns, col_profile_smooth, nm_per_px=1.0, main_img=px_main)
    out_path = os.path.join(out_dir, f'{stem}_tool_result.png')
    cv2.imwrite(out_path, combined_px)
    print(f"[info] 综合结果图(像素)已保存: {out_path}")
    # 物理尺寸版（坐标轴/标注按比例尺换算为 nm，主图自动按 nm 刻度重绘）
    combined_nm = render_combined(img, columns, col_profile_smooth, nm_per_px=nm_per_px)
    out_path_nm = os.path.join(out_dir, f'{stem}_tool_result_nm.png')
    cv2.imwrite(out_path_nm, combined_nm)
    print(f"[info] 综合结果图({nm_per_px:.4f} nm/px)已保存: {out_path_nm}")

    # ---------- 4. 导出 txt 表格（像素版 + 物理尺寸 nm 版） ----------
    txt_path = os.path.join(out_dir, f'{stem}_tool_boundaries.txt')
    export_txt(txt_path, columns, nm_per_px=1.0)
    print(f"[info] 边界表格(像素)已保存: {txt_path}")
    txt_path_nm = os.path.join(out_dir, f'{stem}_tool_boundaries_nm.txt')
    export_txt(txt_path_nm, columns, nm_per_px=nm_per_px)
    print(f"[info] 边界表格({nm_per_px:.4f} nm/px)已保存: {txt_path_nm}")

    # ---------- 4.5 统计分析：高度 / 上中下三段 CD / 比值 / 侧壁角 / 粗糙度 ----------
    stats_path = os.path.join(out_dir, f'{stem}_tool_stats_nm.txt')
    analyze_boundaries_file(txt_path_nm, stats_path)
    print(f"[info] 统计分析表格已保存: {stats_path}")

    # ---------- 5. 统计信息（像素 + 换算后物理尺寸） ----------
    print("\n[info] 各 PR 柱子统计:")
    for col in columns:
        ys = [r[0] for r in col['ranges']]
        xs = [r[1] for r in col['ranges']]
        xe = [r[2] for r in col['ranges']]
        widths = [e - s + 1 for s, e in zip(xs, xe)]
        w_nm = np.mean(widths) * nm_per_px
        h_nm = (max(ys) - min(ys) + 1) * nm_per_px
        print(f"  PR{col['label']}: x=({min(xs)},{max(xe)})px, y=({min(ys)},{max(ys)})px, "
              f"平均宽度={np.mean(widths):.1f}px({w_nm:.1f}nm), "
              f"高度={max(ys) - min(ys) + 1}px({h_nm:.1f}nm), "
              f"中心x={col['center_x']}px({col['center_x'] * nm_per_px:.1f}nm), 颜色BGR={col['color']}")

    # 本图全部文件输出完成后，再交给 GUI 预览（此时统计表已可读）
    if on_image is not None:
        on_image(img, px_main, os.path.basename(image_path))


def process_image_roi(image_path, roi, on_image=None):
    """
    在用户框选的矩形区域 (x0, y0, x1, y1)（整图像素坐标）内做灰度自动识别：
    裁剪该区域 → 用与整图完全相同的阈值/谷值分割逐根套索 PR 柱 →
    坐标平移回整图 → 按整图坐标出图/导出 txt（文件名带 _user，区分于
    工具自动生成的 _tool 结果，互不覆盖）。
    on_image：回调 on_image(img_rgb, main_img, file_name)，预览整图与整图涂色。
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(INPUT_DIR, stem) if OUTPUT_SUBDIR else INPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    x0, y0, x1, y1 = [int(v) for v in roi]
    print(f"\n{'=' * 60}\n[info] ROI 区域分析: {image_path}  roi=({x0},{y0},{x1},{y1})")

    img, binary = preprocess_image(image_path)
    h, w = img.shape[:2]
    x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
    y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError(f"ROI 区域过小 ({x1 - x0 + 1}x{y1 - y0 + 1})，请重新框选更大的区域。")

    # 区域裁剪后做相同的灰度自动套索（阈值/投影/连通轮廓），坐标为局部
    sub_bin = binary[y0:y1 + 1, x0:x1 + 1]
    sub_img = img[y0:y1 + 1, x0:x1 + 1].copy()
    columns, _prof_local = _detect_columns(sub_img, sub_bin)
    columns = _offset_columns(columns, x0, y0)

    nm_per_px = SCALE_NM / SCALE_PIXELS
    tag = f"{stem}_user"
    # 整图坐标涂色主图（px / nm 两版）
    px_main = render_main_image(img, columns, nm_per_px=1.0)
    nm_main = render_main_image(img, columns, nm_per_px=nm_per_px)
    cv2.imwrite(os.path.join(out_dir, f'{tag}_result.png'), px_main)
    cv2.imwrite(os.path.join(out_dir, f'{tag}_result_nm.png'), nm_main)
    # 边界 txt（px/nm）+ 统计
    txt_path = os.path.join(out_dir, f'{tag}_boundaries.txt')
    export_txt(txt_path, columns, nm_per_px=1.0)
    txt_path_nm = os.path.join(out_dir, f'{tag}_boundaries_nm.txt')
    export_txt(txt_path_nm, columns, nm_per_px=nm_per_px)
    stats_path = os.path.join(out_dir, f'{tag}_stats_nm.txt')
    analyze_boundaries_file(txt_path_nm, stats_path)
    print(f"[info] ROI 结果已保存到 {out_dir}（{tag}_result*.png / boundaries*.txt / stats_nm.txt），"
          f"识别到 {len(columns)} 根柱")

    if on_image is not None:
        on_image(img, px_main, os.path.basename(image_path))


def main(on_image=None):
    """批量处理 INPUT_DIR 下的所有 tif/tiff 图像。
    on_image：可选回调，逐张透传给 process_image，供 GUI 实时预览。"""
    tif_files = sorted({p for ext in INPUT_GLOBS for p in glob.glob(os.path.join(INPUT_DIR, ext))})
    if not tif_files:
        print(f"[info] 目录 {INPUT_DIR} 下未找到 {INPUT_GLOBS} 图像，退出。")
        return
    print(f"[info] 共找到 {len(tif_files)} 张 tif 图像:")
    for p in tif_files:
        print(f"  - {os.path.basename(p)}")
    ok, fail = 0, 0
    for p in tif_files:
        try:
            process_image(p, on_image=on_image)
            ok += 1
        except Exception as e:
            print(f"[error] 处理 {p} 失败: {e}")
            fail += 1
    print(f"\n{'=' * 60}\n[info] 批量处理完成: 成功 {ok} 张, 失败 {fail} 张")


# ====================================================================
# GUI 界面代码（core.xxx 即本文件内置核心）
# ====================================================================
core = sys.modules[__name__]

# ---------------------------------------------------------------------------
# 界面参数表：每项 (模块常量名, 中文名, 类型, 允许最小值, 最大值, 步长)
# ---------------------------------------------------------------------------
MAIN_PARAMS = [
    ("THRESHOLD",       "二值化阈值",     "int",   0,   255,   1),
    ("MORPH_CLOSE_K",   "闭运算核(奇)",   "int",   1,   101,   2),
    ("SMOOTH_KERNEL",   "投影平滑核(奇)", "int",   1,   101,   2),
    ("VALLEY_RATIO",    "谷值比例",       "float", 0.0, 1.0,   0.05),
    ("MIN_COLUMN_HEIGHT", "最小柱高(px)", "int",   1,   480,   1),
    ("MIN_AVG_WIDTH",   "最小平均宽(px)", "int",   1,   300,   1),
    ("MIN_ASPECT_RATIO", "最小高宽比",    "float", 0.5, 10.0,  0.1),
    ("BOTTOM_OFFSET",   "底部延伸(px)",   "int",   0,   30,    1),
    ("EDGE_MARGIN",     "边缘忽略(px)",   "int",   0,   60,    1),
    ("WINDOW_MARGIN",   "扫描窗口外扩(px)", "int",  0,   20,    1),
]

# 每个参数的详细说明（帮助 → 参数说明）
PARAM_DESC = {
    "THRESHOLD": "二值化阈值：灰度 > 该值的像素视为“亮”（柱子区域）。调低更容易把亮度偏暗的柱子纳入，"
                 "调高可滤掉偏暗噪声。若整图亮像素占比异常（<1% 或 >90%）会自动改用 Otsu 自适应阈值。",
    "MORPH_CLOSE_K": "闭运算核大小（自动取奇）：先膨胀再腐蚀，填补柱体内部小孔/毛刺。核需小于柱间谷宽，"
                     "否则会把相邻柱子粘连。柱体越宽可适度调大。",
    "SMOOTH_KERNEL": "投影平滑核大小（自动取奇）：对 x=n 列投影 / y=n 行投影做高斯平滑，越大曲线越光滑、"
                     "谷值越少（可避免把一根柱子中间的小凹槽误切，但过大可能把小柱与邻居并在一起）。",
    "VALLEY_RATIO": "谷值比例：列投影中低于“最高峰 × 该比例”的区域视为分隔柱子的谷槽。值越大，切分越激进"
                    "（谷判定越宽松），值越小越保守。",
    "MIN_COLUMN_HEIGHT": "最小柱高(px)：柱子轮廓高度低于该值判为噪声丢弃。",
    "MIN_AVG_WIDTH": "最小平均宽(px)：柱子各行宽度平均值低于该值判为文字/窄条噪声丢弃。",
    "MIN_ASPECT_RATIO": "最小高宽比：高度 < 平均宽度 × 该值 的扁宽亮块（基底亮斑等）判为噪声丢弃。",
    "BOTTOM_OFFSET": "底部延伸(px)：识别出的底部再向下延伸几像素，覆盖基座边缘的毛刺，让底边更整齐。",
    "EDGE_MARGIN": "边缘忽略(px)：把图像四边各裁掉该宽度并置黑，避免图像边框/标尺线参与识别。",
    "WINDOW_MARGIN": "扫描窗口外扩(px)：逐行连通扫描时，允许当前行亮段相对上一行左右扩展的余量，"
                     "用于跟随柱壁轻微抖动。",
    "SCALE_PIXELS": "比例尺像素数：图像上已知实际长度所对应的像素总长度（如标尺 50 nm 对应 100 px 则填 100）。",
    "SCALE_NM": "实际长度(nm)：上述比例尺对应的真实物理长度。换算系数 = SCALE_NM / SCALE_PIXELS。",
    "OUTPUT_SUBDIR": "输出到“同名子文件夹”：为每张 tif 建立同名文件夹存放结果；关闭时全部输出到图像文件夹根目录。",
    "DEBUG_OUTPUT": "调试模式：额外保存一张二值化中间图（*_debug_binary.png），便于观察阈值是否合适。",
}

# 生成文件的来源标记说明（帮助 → 输出文件说明 里的简短行）
FILE_TAG_HELP = (
    "文件名中的 _tool 表示“工具自动整图分析”生成，_user 表示“你在图上框选后由套索分析”生成；\n"
    "两类结果都放在同一张图对应的输出目录里，互不覆盖。\n"
    "· 自动(tool)：xx_tool_result.png / xx_tool_result_nm.png / xx_tool_boundaries.txt /\n"
    "              xx_tool_boundaries_nm.txt / xx_tool_stats_nm.txt\n"
    "· 套索(user)：xx_user_result.png / xx_user_result_nm.png / xx_user_boundaries.txt /\n"
    "              xx_user_boundaries_nm.txt / xx_user_stats_nm.txt"
)

FORMULA_HELP = """统计口径与计算公式说明（单位统一为 nm；像素 px 与 nm 的换算系数 k = SCALE_NM / SCALE_PIXELS）

■ 原始数据（*_boundaries_nm.txt）
   文件中每一行 = 柱体轮廓的一条水平切片 y，每列给出该柱体在该 y 处的
   左右边缘 (x_start, x_end)。轮廓来自对柱体“逐行连通扫描”得到的完整封闭图形，
   y 从柱顶一直取到柱底（约每 0.5 nm 一行）。

■ 三段划分（上 / 中 / 下）
   把一根柱从顶到底的切片按行数平均分成三段：上段(top)、中段(mid)、下段(bot)，
   每段约占总行数的 1/3。

■ 高度 height
   height = (最底一行 y − 最顶一行 y) + 行间距 step
   其中 step 为相邻两行 y 的最小正差值（约等于 0.5 nm）。相当于“底边 − 顶边 + 1 行”。

■ 三段平均 CD（线宽）
   单行 CD = x_end − x_start（该 y 处的线宽）
   某段平均 CD = 该段内所有切片行 CD 的平均值。

■ CD 比值（无量纲，用于判断上窄下宽 / 梯形程度）
   上/中 = CD_top / CD_mid     上/下 = CD_top / CD_bot     中/下 = CD_mid / CD_bot

■ 侧壁角（°，与 +x 轴正向夹角，范围可大于 90°）
   把每根柱左/右边缘看成一条沿柱壁“自顶向底（y 增大方向）”的直线，
   求它与图像水平 +x 轴(向右)的夹角 angle = atan2(竖直分量, 水平分量)：
   · 竖直侧壁 = 90°；向右倾（水平分量>0）< 90°；向左倾（水平分量<0）> 90°。
   行数足够时，对某段的左/右边缘分别做“x ~ y”的最小二乘拟合得斜率 b=dx/dy，
   侧壁角 = atan2(1, b)；段内行数 < 4 时退回端点法：
   角度 = atan2(段高, x_bottom − x_top)。
   因此梯形/倾斜柱两壁会分别落在 90° 两侧，不再被绝对值折叠到 [0,90]。

■ 边缘粗糙度 LER 3σ（nm）
   把整根柱的左（或右）边缘所有 (y, x) 点做线性拟合 x = a + b·y，取拟合残差
   （实际边缘 − 拟合直线）的标准差，LER = 3 × 标准差。它描述边缘局部起伏的程度，
   值越大边缘越粗糙。文件里 _boundaries 边缘逐点原始值即 LER 的输入。

■ “全部柱平均”列
   对每个统计项，把所有 PR 柱该行取值求算术平均（无法计算的行自动跳过）。
"""


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PR 柱子自动识别与测量工具")
        self.geometry("1640x960")
        self.minsize(1240, 720)

        self.q = queue.Queue()
        self.running = False
        self.last_dir = core.INPUT_DIR       # 最后处理的目录（“打开输出文件夹”用）
        self._prev = None                    # (原始BGR, 涂色BGR, 文件名)，供图像列显示
        self._photos = []                    # 保存 tk.PhotoImage 引用，防止被回收
        self._resize_job = None
        self._scale = 1.0                    # 当前原图显示缩放（canvas 像素 → 原图像素）
        self._roi = None                     # 框选矩形：整图像素坐标 (x0,y0,x1,y1)
        self._drag = None                    # 正在拖拽画框
        self._drag_id = None                 # 拖拽过程中的临时矩形
        self._m1 = None                      # 直线量测：第一点（原图像素）
        self._m2 = None                      # 直线量测：第二点（原图像素）
        self._mv_id = None                   # 量测过程中的临时连线
        self._last_stem = None               # 当前预览图 stem
        self._last_out_tag = "_tool"         # 生成文件来源标记：_tool 自动 / _user 用户套索

        self._build_ui()
        self._refresh_files()
        self.last_dir = self._folder()   # 打开输出文件夹：默认跟随界面所选文件夹
        self.after(80, self._drain_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ===================================================================
    # 界面构建（三列 + 底部进度行）
    # ===================================================================
    def _build_ui(self):
        try:
            ttk.Style(self).theme_use("vista")
        except tk.TclError:
            pass

        # ---------------- 顶部菜单栏：帮助 ----------------
        menubar = tk.Menu(self)
        m_help = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=m_help)
        m_help.add_command(label="调节参数含义", command=self._show_params_help)
        m_help.add_command(label="统计计算公式说明", command=lambda: self._show_doc("统计计算公式说明", FORMULA_HELP))
        m_help.add_command(label="输出文件命名说明", command=lambda: self._show_doc("输出文件命名说明", FILE_TAG_HELP))
        m_help.add_separator()
        m_help.add_command(label="关于", command=self._show_about)
        self.config(menu=menubar)

        # ---------------- 顶部工具栏 ----------------
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Label(top, text="图像文件夹：").pack(side="left")
        default_dir = core.INPUT_DIR
        if getattr(sys, "frozen", False):  # exe：默认打开 exe 所在目录
            d = os.path.dirname(os.path.abspath(sys.executable))
            if os.path.isdir(d):
                default_dir = d
        self.var_dir = tk.StringVar(value=default_dir)
        ttk.Entry(top, textvariable=self.var_dir, width=24).pack(side="left", padx=4)
        ttk.Button(top, text="浏览文件夹", command=self._browse_dir).pack(side="left")
        ttk.Button(top, text="选择 tif", command=self._browse_file).pack(side="left", padx=(4, 0))

        self.var_sel = tk.StringVar()
        self.combo = ttk.Combobox(top, textvariable=self.var_sel, state="readonly", width=14)
        self.combo.pack(side="left", padx=(10, 2))

        self.btn_run1 = ttk.Button(top, text="▶ 分析选中图像", command=self._start_single)
        self.btn_run1.pack(side="left", padx=(8, 2))
        self.btn_run_all = ttk.Button(top, text="批量分析全部", command=self._start_batch)
        self.btn_run_all.pack(side="left", padx=2)
        self.btn_open = ttk.Button(top, text="打开输出文件夹", command=self._open_output)
        self.btn_open.pack(side="left", padx=(4, 8))

        # ---------------- 主体三列 ----------------
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10, pady=(2, 4))

        # ---- 第一列：设定的参数 ----
        left = ttk.Frame(body, width=430)
        left.pack(side="left", fill="y", padx=(0, 6))
        left.pack_propagate(False)

        pf = ttk.LabelFrame(left, text="识别参数")
        pf.pack(fill="x", padx=4, pady=(2, 6))
        self.spins = {}
        for i in range(0, len(MAIN_PARAMS), 2):
            row = i // 2
            for col, spec in enumerate(MAIN_PARAMS[i:i + 2]):
                attr, name, kind, lo, hi, inc = spec
                cell = ttk.Frame(pf)
                cell.grid(row=row, column=col, sticky="w", padx=6, pady=3)
                ttk.Label(cell, text=name).pack(side="left")
                var = tk.StringVar(value=str(getattr(core, attr)))
                tk.Spinbox(cell, from_=lo, to=hi, increment=inc, width=6,
                           textvariable=var, font=("Segoe UI", 10)).pack(side="left", padx=3)
                self.spins[attr] = var
            pf.grid_columnconfigure(1, weight=1)
            pf.grid_columnconfigure(3, weight=1)
        ttk.Label(pf, text="提示：核大小需为奇数（自动取奇）；默认值已按样例调好。",
                  foreground="#888888", wraplength=390, justify="left").grid(
            row=(len(MAIN_PARAMS) + 1) // 2, column=0, columnspan=4,
            sticky="w", padx=8, pady=(2, 6))

        rf = ttk.LabelFrame(left, text="比例尺换算与输出")
        rf.pack(fill="x", padx=4)
        self.scale_vars = {}
        for attr, name, lo, hi, inc, unit in [
            ("SCALE_PIXELS", "比例尺长度(px)", 1, 100000, 1, "图上比例尺像素数"),
            ("SCALE_NM",     "实际长度(nm)",   0.1, 100000, 0.5, "对应真实长度"),
        ]:
            cell = ttk.Frame(rf)
            cell.pack(fill="x", padx=8, pady=3)
            ttk.Label(cell, text=name).pack(side="left")
            var = tk.StringVar(value=str(getattr(core, attr)))
            tk.Spinbox(cell, from_=lo, to=hi, increment=inc, width=9,
                       textvariable=var, font=("Segoe UI", 10)).pack(side="left", padx=4)
            ttk.Label(cell, text=unit, foreground="#888888").pack(side="left")
            self.scale_vars[attr] = var
        for attr, var in self.scale_vars.items():
            var.trace_add("write", lambda *_: self._update_scale_info())
        self.lbl_scale_info = ttk.Label(rf, text="")
        self.lbl_scale_info.pack(anchor="w", padx=8, pady=(0, 3))

        self.var_subdir = tk.BooleanVar(value=bool(core.OUTPUT_SUBDIR))
        ttk.Checkbutton(rf, text="输出到“同名子文件夹”",
                        variable=self.var_subdir).pack(anchor="w", padx=8)
        self.var_debug = tk.BooleanVar(value=bool(core.DEBUG_OUTPUT))
        ttk.Checkbutton(rf, text="调试：另存二值化中间图",
                        variable=self.var_debug).pack(anchor="w", padx=8)
        ttk.Label(rf, text="每张图生成两类结果（互不覆盖）：\n自动整图：xx_tool_result.png / _boundaries.txt / _stats_nm.txt…\n框选套索：xx_user_result.png / _boundaries.txt / _stats_nm.txt…",
                  foreground="#888888", justify="left", wraplength=400).pack(anchor="w", padx=8, pady=(8, 6))
        self._update_scale_info()

        # ---- 第二列：ROI 工具栏 + 原始图(canvas) + 识别涂色图 ----
        self.frm_mid = ttk.Frame(body)
        self.frm_mid.pack(side="left", fill="both", expand=True, padx=6)
        self.frm_mid.grid_rowconfigure(1, weight=1)
        self.frm_mid.grid_rowconfigure(2, weight=1)
        self.frm_mid.grid_columnconfigure(0, weight=1)

        roibar = ttk.Frame(self.frm_mid)
        roibar.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.var_canvas_tool = tk.StringVar(value="roi")
        ttk.Radiobutton(roibar, text="框选区域", value="roi", variable=self.var_canvas_tool,
                        command=self._on_tool_change).pack(side="left")
        ttk.Radiobutton(roibar, text="直线量测", value="measure", variable=self.var_canvas_tool,
                        command=self._on_tool_change).pack(side="left", padx=(10, 0))
        self.lbl_roi_info = ttk.Label(roibar, text="提示：在原始图上按住左键拖出矩形分析区域",
                                      foreground="#555555")
        self.lbl_roi_info.pack(side="left", padx=8)
        self.btn_roi_clear = ttk.Button(roibar, text="清除", command=self._clear_canvas_marks,
                                        width=6)
        self.btn_roi_clear.pack(side="left")
        self.btn_roi_run = ttk.Button(roibar, text="▶ 套索分析所选区域",
                                      command=self._start_roi)
        self.btn_roi_run.pack(side="right")

        self.frm_orig = ttk.LabelFrame(self.frm_mid, text="原始图像（框选区域 / 直线量测，工具在上方切换）")
        self.frm_orig.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.cv_orig = tk.Canvas(self.frm_orig, bg="white", highlightthickness=0,
                                 cursor="crosshair")
        self.cv_orig.pack(fill="both", expand=True)
        self._bind_canvas()

        self.frm_paint = ttk.LabelFrame(self.frm_mid, text="识别结果（彩色涂色）")
        self.frm_paint.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)
        self.lbl_paint = tk.Label(self.frm_paint, text="（识别预览）", bg="white",
                                  fg="#999999")
        self.lbl_paint.pack(expand=True)

        self.frm_mid.bind("<Configure>", self._on_mid_resize)

        # ---- 第三列：统计分析表（无日志区） ----
        right = ttk.Frame(body, width=460)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.pack_propagate(False)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        stf = ttk.LabelFrame(right, text="统计分析  (高度·三段CD·比值·侧壁角·粗糙度)")
        stf.grid(row=0, column=0, sticky="nsew")
        self.var_stat_title = tk.StringVar(value="运行后在此显示当前图像的统计结果")
        ttk.Label(stf, textvariable=self.var_stat_title,
                  foreground="#555555", wraplength=430).pack(anchor="w", padx=8, pady=(3, 0))
        tcol = ttk.Frame(stf)
        tcol.pack(fill="both", expand=True, padx=4, pady=2)
        self.tree = ttk.Treeview(tcol, columns=("avg",), show="tree headings",
                                 selectmode="none")
        self.tree.heading("#0", text="计算项")
        self.tree.heading("avg", text="")
        self.tree.column("#0", width=205, minwidth=150, anchor="w", stretch=True)
        self.tree.column("avg", width=80, minwidth=70, anchor="e", stretch=False)
        self.tree.tag_configure("cat", font=("Microsoft YaHei UI", 9, "bold"),
                                background="#e8f0e8")
        self.tree.tag_configure("odd", background="#f4f7f4")
        ys = ttk.Scrollbar(tcol, orient="vertical", command=self.tree.yview)
        xs = ttk.Scrollbar(tcol, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tcol.grid_rowconfigure(0, weight=1)
        tcol.grid_columnconfigure(0, weight=1)
        ttk.Label(stf, text="侧壁角：边缘沿柱壁向下与 +x 轴的夹角，90°=竖直，可 >90°（向左倾）；LER=去趋势边缘残差 3σ。",
                  foreground="#888888", wraplength=430, justify="left").pack(
            anchor="w", padx=8, pady=(0, 4))

        # ---------------- 最底部：正在识别的图片 + 进度 ----------------
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom", padx=10, pady=(0, 6))
        self.var_status = tk.StringVar(value="就绪：选择一个 tif 或“批量分析全部”开始")
        ttk.Label(bar, textvariable=self.var_status).pack(side="left")
        self.pbar = ttk.Progressbar(bar, mode="determinate", length=260, maximum=100)
        self.pbar.pack(side="right", padx=(10, 4))
        self.lbl_pct = ttk.Label(bar, text="0%", width=5)
        self.lbl_pct.pack(side="right")

    # ===================================================================
    # 图像列表 / 目录
    # ===================================================================
    def _list_tifs(self):
        folder = self.var_dir.get().strip().strip('"') or core.INPUT_DIR
        if not os.path.isdir(folder):
            return []
        return sorted({p for ext in core.INPUT_GLOBS
                       for p in glob.glob(os.path.join(folder, ext))})

    def _refresh_files(self):
        names = [os.path.basename(p) for p in self._list_tifs()]
        self.combo["values"] = names
        if names and self.var_sel.get() not in names:
            self.var_sel.set(names[0])

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.var_dir.get() or core.INPUT_DIR,
                                    title="选择包含 tif 图像的文件夹")
        if d:
            self.var_dir.set(os.path.normpath(d))
            self._refresh_files()

    def _browse_file(self):
        f = filedialog.askopenfilename(
            initialdir=self.var_dir.get() or core.INPUT_DIR,
            title="选择一张 tif 图像",
            filetypes=[("TIF 图像", "*.tif *.tiff"), ("所有文件", "*.*")])
        if f:
            self.var_dir.set(os.path.dirname(f))
            self._refresh_files()
            self.var_sel.set(os.path.basename(f))

    # ===================================================================
    # 显示辅助（原图 canvas + 涂色图 label）
    # ===================================================================
    def _update_scale_info(self):
        try:
            px = float(self.scale_vars["SCALE_PIXELS"].get())
            nm = float(self.scale_vars["SCALE_NM"].get())
            if px > 0:
                self.lbl_scale_info.configure(text=f"换算：1 px = {nm / px:.4f} nm")
                return
        except ValueError:
            pass
        self.lbl_scale_info.configure(text="换算：请输入有效数值")

    def _bgr_to_photo(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", rgb)  # rgb 数组 → PNG，通道顺序与显示一致
        if not ok:
            return None
        return tk.PhotoImage(data=buf.tobytes())

    def _set_preview(self, orig_bgr, paint_bgr, name):
        self._prev = (orig_bgr, paint_bgr, name)
        self._roi = None        # 新图像 → 之前的框选区域/量测点作废
        self._m1 = self._m2 = None
        self._redraw_preview()

    def _redraw_preview(self):
        self._resize_job = None
        self.cv_orig.delete("all")
        if self._prev is None:
            self._placeholder()
            return
        self.update_idletasks()
        cw = max(self.frm_mid.winfo_width() - 24, 60)
        ch = max(self.frm_mid.winfo_height() - 40, 60)
        per_h = max((ch - 14) // 2, 30)
        orig, paint, name = self._prev
        # 统一缩放因子，让两张图比例一致地放进各自半区
        s = min(1.0,
                cw / paint.shape[1], cw / orig.shape[1],
                per_h / paint.shape[0], per_h / orig.shape[0])
        s = max(s, 0.02)
        self._scale = s
        self._ow, self._oh = orig.shape[1], orig.shape[0]
        photos = []
        for img in (orig, paint):
            if s < 1.0:
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                                 interpolation=cv2.INTER_AREA)
            ph = self._bgr_to_photo(img)
            photos.append(ph if ph is not None else tk.PhotoImage(width=1, height=1))
        self._photos = photos
        self.frm_orig.configure(text=f"原始图像（左键拖框=选取分析区域）  {name}"
                                     f"  ({orig.shape[1]}×{orig.shape[0]})")
        self.frm_paint.configure(text=f"识别结果（彩色涂色）  {name}")
        self.cv_orig.create_image(0, 0, anchor="nw", image=photos[0])
        self.lbl_paint.configure(image=photos[1], text="")
        self._redraw_marks()

    def _placeholder(self):
        w = max(self.cv_orig.winfo_width(), 200)
        h = max(self.cv_orig.winfo_height(), 100)
        self.cv_orig.create_text(w // 2, h // 2,
                                 text="（选择图像并点击“分析选中图像”或先拖框选区域再“套索分析”）",
                                 fill="#999999", font=("Microsoft YaHei UI", 10))

    def _on_mid_resize(self, _evt=None):
        if self._prev is not None and self._resize_job is None:
            self._resize_job = self.after(120, self._redraw_preview)

    # ===================================================================
    # 第二列画布交互：框选区域(roi) / 直线量测(measure)
    # ===================================================================
    def _bind_canvas(self):
        self.cv_orig.bind("<ButtonPress-1>", self._cv_press)
        self.cv_orig.bind("<B1-Motion>", self._cv_move)
        self.cv_orig.bind("<ButtonRelease-1>", self._cv_release)

    def _cv_to_orig(self, ex, ey):
        """canvas 显示坐标 → 原图像素坐标（越界自动裁剪到图像内）。"""
        ox = min(max(int(ex / self._scale), 0), self._ow - 1)
        oy = min(max(int(ey / self._scale), 0), self._oh - 1)
        return ox, oy

    def _tool(self):
        return self.var_canvas_tool.get()

    def _on_tool_change(self):
        """切换工具：清空另一工具的标记，避免两类图元叠在一起造成误读。"""
        self._drag = None
        if self._tool() == "measure":
            self._roi = None
        else:
            self._m1 = self._m2 = None
        self._redraw_marks()

    def _clear_canvas_marks(self):
        self._roi = None
        self._m1 = self._m2 = None
        self._drag = None
        self._redraw_marks()

    # ---- 鼠标事件 ----
    def _cv_press(self, e):
        if self.running or self._prev is None:
            return
        if self._tool() == "roi":
            self._drag = (e.x, e.y)
            return
        # 直线量测：单击打点
        p = self._cv_to_orig(e.x, e.y)
        if self._m1 is None:
            self._m1 = p
            self._m2 = None
        elif self._m2 is None:
            self._m2 = p
        else:                      # 已测完一组 → 点击开始量测下一组
            self._m1, self._m2 = p, None
        self._redraw_marks()

    def _cv_move(self, e):
        if self.running or self._prev is None:
            return
        if self._tool() == "roi":
            if self._drag is None:
                return
            self._redraw_marks()   # 清除旧预览框
            x0, y0 = self._drag
            self.cv_orig.create_rectangle(x0, y0, e.x, e.y,
                                          outline="#ff3333", width=2, dash=(5, 3),
                                          tags="ovl")
        else:
            # 量测中：第一点已定、第二点未定 → 实时预览连线
            if self._m1 is not None and self._m2 is None:
                self._redraw_marks()
                ax, ay = self._m1
                self.cv_orig.create_line(ax * self._scale, ay * self._scale,
                                         e.x, e.y,
                                         fill="#0099ff", width=1, dash=(2, 2),
                                         tags="ovl")

    def _cv_release(self, e):
        if self._drag is None or self._tool() != "roi":
            return
        x0, y0 = self._drag
        self._drag = None
        self._redraw_marks()
        ax, ay = self._cv_to_orig(x0, y0)
        bx, by = self._cv_to_orig(e.x, e.y)
        if abs(bx - ax) < 3 and abs(by - ay) < 3:
            return                      # 拖动过小视为误操作，仅清除预览
        self._roi = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        self._redraw_marks()

    # ---- 覆盖图元绘制 ----
    def _clear_overlays(self):
        self.cv_orig.delete("ovl")

    def _redraw_marks(self):
        """把 ROI 矩形与量测线段按当前缩放重画到原图 canvas 上。"""
        if self._prev is None:
            return
        self._clear_overlays()
        s = self._scale
        if self._roi is not None:
            x0, y0, x1, y1 = self._roi
            self.cv_orig.create_rectangle(x0 * s, y0 * s, x1 * s, y1 * s,
                                          outline="#ff0000", width=2, dash=(5, 3),
                                          tags="ovl")
        if self._m1 is not None:
            ax, ay = self._m1
            r = max(int(3 * s), 2)
            self.cv_orig.create_oval(ax * s - r, ay * s - r, ax * s + r, ay * s + r,
                                     outline="#0066ff", fill="#33aaff", tags="ovl")
            if self._m2 is not None:
                bx, by = self._m2
                self.cv_orig.create_line(ax * s, ay * s, bx * s, by * s,
                                         fill="#0066ff", width=2, tags="ovl")
                self.cv_orig.create_oval(bx * s - r, by * s - r, bx * s + r, by * s + r,
                                         outline="#0066ff", fill="#33aaff", tags="ovl")
                px = math.hypot(bx - ax, by - ay)
                self.cv_orig.create_text((ax + bx) * s / 2,
                                         max(min((ay + by) * s / 2 - 8, self._oh * s - 10), 10),
                                         text=f"L={px:.2f} px", fill="#0044cc",
                                         font=("Microsoft YaHei UI", 9, "bold"),
                                         tags="ovl")
        self._update_canvas_info()

    def _update_canvas_info(self):
        """根据当前工具与已绘制内容，更新工具条左侧的提示/结果文字。"""
        if self._tool() == "measure":
            if self._m1 is not None and self._m2 is not None:
                ax, ay = self._m1
                bx, by = self._m2
                px = math.hypot(bx - ax, by - ay)
                nm = self._px_to_nm(px)
                txt = f"量测 A({ax},{ay}) → B({bx},{by})：{px:.2f} px"
                if nm is not None:
                    txt += f"（≈{nm:.2f} nm）"
                self.lbl_roi_info.configure(text=txt, foreground="#0055bb")
            elif self._m1 is not None:
                self.lbl_roi_info.configure(text="量测中：请单击第二个点 B …", foreground="#aa5500")
            else:
                self.lbl_roi_info.configure(
                    text="提示：单击 A 点后再单击 B 点，量测两点间直线长度",
                    foreground="#555555")
        else:
            if self._roi is None:
                self.lbl_roi_info.configure(
                    text="提示：按住左键拖出矩形框选分析区域（套索分析）",
                    foreground="#555555")
            else:
                x0, y0, x1, y1 = self._roi
                w, h = x1 - x0 + 1, y1 - y0 + 1
                self.lbl_roi_info.configure(
                    text=f"ROI：({x0},{y0})→({x1},{y1})  {w}×{h}px",
                    foreground="#cc0000")

    def _px_to_nm(self, px):
        try:
            nm = float(self.scale_vars["SCALE_NM"].get())
            p = float(self.scale_vars["SCALE_PIXELS"].get())
            return None if p <= 0 else px * nm / p
        except ValueError:
            return None

    # ===================================================================
    # 运行控制
    # ===================================================================
    def _collect_params(self):
        """读取界面参数写回 pr2_analysis 模块常量，供分析核心使用。"""
        folder = self.var_dir.get().strip().strip('"')
        if not folder or not os.path.isdir(folder):
            raise ValueError("图像文件夹不存在，请先选择有效的文件夹。")
        core.INPUT_DIR = os.path.normpath(folder)
        core.OUTPUT_SUBDIR = bool(self.var_subdir.get())
        core.DEBUG_OUTPUT = bool(self.var_debug.get())
        for attr, name, kind, *_ in MAIN_PARAMS:
            raw = self.spins[attr].get().strip().replace("，", ".")
            val = int(raw) if kind == "int" else float(raw)
            if attr in ("MORPH_CLOSE_K", "SMOOTH_KERNEL") and val % 2 == 0:
                val += 1
                self.spins[attr].set(str(val))
            setattr(core, attr, val)
        for attr in ("SCALE_PIXELS", "SCALE_NM"):
            setattr(core, attr, float(self.scale_vars[attr].get().strip()))
        if core.SCALE_PIXELS <= 0:
            raise ValueError("比例尺像素数必须大于 0。")

    def _make_cb(self, out_tag):
        def cb(orig, paint, name):
            self._last_stem = os.path.splitext(os.path.basename(name))[0]
            self._last_out_tag = out_tag
            self.q.put(("img", orig, paint, name))
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
        threading.Thread(target=self._worker, args=(jobs,), daemon=True).start()

    def _folder(self):
        """界面当前选择的图像文件夹（归一化）。"""
        f = self.var_dir.get().strip().strip('"')
        return os.path.normpath(f or core.INPUT_DIR)

    def _start_single(self):
        if not self.var_sel.get():
            messagebox.showwarning("提示", "请先在列表中选择一张图像。")
            return
        p = os.path.join(self._folder(), self.var_sel.get())
        jobs = [{"path": p, "mode": "full", "roi": None}]
        self._start_jobs(jobs, f"准备分析：{self.var_sel.get()} …")

    def _start_batch(self):
        paths = self._list_tifs()
        if not paths:
            messagebox.showwarning("提示", "文件夹中没有 *.tif / *.tiff 图像。")
            return
        jobs = [{"path": p, "mode": "full", "roi": None} for p in paths]
        self._start_jobs(jobs, f"准备批量分析（{len(paths)} 张）…")

    def _start_roi(self):
        if self._roi is None or self._prev is None:
            messagebox.showwarning("提示", "请先在原始图像上按住左键拖出矩形分析区域。")
            return
        name = self._prev[2]
        p = os.path.join(self._folder(), name)
        if not os.path.isfile(p):
            messagebox.showwarning("提示",
                                   f"预览图不在当前图像文件夹中：\n{p}\n"
                                   f"请先选择其所在文件夹并分析一次该图。")
            return
        x0, y0, x1, y1 = self._roi
        jobs = [{"path": p, "mode": "roi", "roi": (x0, y0, x1, y1)}]
        self._start_jobs(jobs, f"ROI 框内套索分析：{name}  ({x0},{y0})-({x1},{y1}) …")

    def _worker(self, jobs):
        total = len(jobs)
        ok = 0
        fails = []
        with open(os.devnull, "w", encoding="utf-8") as sink, \
                contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            for i, j in enumerate(jobs, 1):
                bname = os.path.basename(j["path"])
                self.q.put(("prog", i, total, bname))
                try:
                    if j["mode"] == "roi":
                        core.process_image_roi(j["path"], j["roi"],
                                               on_image=self._make_cb("_user"))
                    else:
                        core.process_image(j["path"], on_image=self._make_cb("_tool"))
                    ok += 1
                except Exception:
                    fails.append(bname)
        self.q.put(("done", ok, total, fails))

    def _set_running(self, on):
        self.running = on
        state = "disabled" if on else "normal"
        for b in (self.btn_run1, self.btn_run_all, self.btn_open,
                  self.btn_roi_run, self.btn_roi_clear):
            b.configure(state=state)
        self.combo.configure(state="disabled" if on else "readonly")

    # ===================================================================
    # 第三列统计分析视图
    # ===================================================================
    @staticmethod
    def _fmt_stat(desc, v):
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "-"
        if "角" in desc:
            return f"{v:.1f}"
        if "比值" in desc:
            return f"{v:.3f}"
        return f"{v:.3f}"

    def _refresh_stats_view(self):
        stem = self._last_stem
        if not stem:
            self.var_stat_title.set("尚未分析图像")
            return
        out_dir = core.INPUT_DIR if not core.OUTPUT_SUBDIR \
            else os.path.join(core.INPUT_DIR, stem)
        tag = self._last_out_tag or "_tool"
        nm_txt = os.path.join(out_dir, f"{stem}{tag}_boundaries_nm.txt")
        if not os.path.isfile(nm_txt):
            self.var_stat_title.set(f"{stem}: 未找到 {os.path.basename(nm_txt)}")
            return
        try:
            data = core.analyze_boundaries_file(nm_txt)
            self._populate_stats_tree(data)
            src = "框选套索" if tag == "_user" else "工具自动"
            self.var_stat_title.set(f"当前：{stem}（{src}）  ← {os.path.basename(nm_txt)}")
        except Exception as e:
            self.var_stat_title.set(f"统计表读取失败：{e}")

    def _populate_stats_tree(self, data):
        self.tree.delete(*self.tree.get_children())
        cols, descs, rows, means = data["cols"], data["desc"], data["rows"], data["means"]
        if not cols:
            self.var_stat_title.set("没有识别到 PR 柱，无统计")
            return
        col_ids = [f"p{i}" for i in range(len(cols))] + ["avg"]
        self.tree.configure(columns=col_ids)
        for cid, label in zip(col_ids[:-1], cols):
            self.tree.heading(cid, text=label)
            self.tree.column(cid, width=66, minwidth=58, anchor="e", stretch=False)
        self.tree.heading("avg", text="全部柱平均")
        self.tree.column("avg", width=88, minwidth=80, anchor="e", stretch=False)
        self.tree.heading("#0", text="计算项")

        groups = [
            ("尺寸与 CD（nm）", (0, 4)),
            ("CD 比值（无量纲）", (4, 7)),
            ("三段侧壁角（°，与+x轴，可>90°）", (7, 13)),
            ("边缘粗糙度 LER 3σ（nm）", (13, 15)),
        ]
        toggle = 0
        for gname, (s, e) in groups:
            pid = self.tree.insert("", "end", text=gname, values=[""] * len(col_ids),
                                   tags=("cat",))
            for idx in range(s, e):
                vals = [self._fmt_stat(descs[idx], v) for v in rows[idx]]
                vals.append(self._fmt_stat(descs[idx], means[idx]))
                tag = ("odd",) if toggle % 2 else ()
                toggle += 1
                self.tree.insert(pid, "end", text=descs[idx], values=vals, tags=tag)
            toggle += 1

    # ===================================================================
    # 帮助窗口
    # ===================================================================
    def _show_params_help(self):
        lines = ["可调节参数与默认值：\n"]
        for attr, name, kind, lo, hi, inc in MAIN_PARAMS:
            lines.append(f"■ {name}  [{attr}]\n"
                         f"   类型={kind}，默认 {getattr(core, attr)}，范围 {lo}~{hi}，步长 {inc}\n"
                         f"   {PARAM_DESC.get(attr, '')}\n")
        lines.append(f"■ 比例尺换算（SCALE_PIXELS / SCALE_NM）\n"
                     f"   默认 {core.SCALE_PIXELS} px 对应 {core.SCALE_NM} nm → "
                     f"1 px = {core.SCALE_NM / core.SCALE_PIXELS:.4f} nm。\n"
                     f"   只影响 nm 版结果图坐标轴刻度、boundaries_nm / stats_nm 的数值。\n\n"
                     f"■ 输出方式\n"
                     f"   · 输出到“同名子文件夹”：每张 tif 的结果放进其同名子文件夹。\n"
                     f"   · 调试中间图：把二值化结果另存一张 png，便于确认阈值。\n")
        self._show_doc("调节参数含义", "".join(lines))

    def _show_about(self):
        self._show_doc("关于", "PR 柱子自动识别与测量工具\n\n"
                        "第二列两种工具（原始图左上单选）：\n"
                        "  框选区域  — 左键拖出矩形 → 点“▶ 套索分析所选区域”在框内自动定位 PR 柱；\n"
                        "  直线量测  — 依次单击 A、B 两点，量测直线像素长度（并按比例尺换算 nm）。\n\n"
                        "输出文件按来源区分：工具自动整图分析 = _tool，用户框选套索分析 = _user。\n\n"
                        "本文件是单文件版：分析核心已内置，不依赖其他 .py，\n"
                        "直接 python pr2_gui.py 运行，或用 PyInstaller 只打包本文件即可。")

    def _show_doc(self, title, text):
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("780x600")
        win.transient(self)
        win.minsize(560, 360)
        frm = ttk.Frame(win)
        frm.pack(fill="both", expand=True, padx=8, pady=8)
        box = tk.Text(frm, wrap="word", bg="#fbfbfb", fg="#222222",
                      font=("Microsoft YaHei UI", 10))
        sb = ttk.Scrollbar(frm, orient="vertical", command=box.yview)
        box.configure(yscrollcommand=sb.set)
        box.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        frm.grid_rowconfigure(0, weight=1)
        frm.grid_columnconfigure(0, weight=1)
        box.insert("1.0", text)
        box.configure(state="disabled")
        ttk.Button(win, text="关闭", command=win.destroy).pack(pady=(0, 8))

    # ===================================================================
    # 队列事件
    # ===================================================================
    def _drain_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                if not isinstance(item, tuple):
                    continue
                kind = item[0]
                if kind == "prog":
                    _, i, total, bname = item
                    self.var_status.set(f"正在识别：{bname}    第 {i} / {total} 张")
                    self.pbar.configure(maximum=total, value=i)
                    self.lbl_pct.configure(text=f"{int(i * 100 / total)}%")
                elif kind == "img":
                    _, orig, paint, name = item
                    self._set_preview(orig, paint, name)
                    self._refresh_stats_view()
                elif kind == "done":
                    _, ok, total, fails = item
                    self._set_running(False)
                    self.pbar.configure(maximum=total, value=ok)
                    self.lbl_pct.configure(text=f"{int(ok * 100 / total)}%")
                    if fails:
                        names = "、".join(fails[:4])
                        self.var_status.set(f"完成：成功 {ok}/{total} 张，失败：{names}"
                                            + (" 等" if len(fails) > 4 else ""))
                    else:
                        self.var_status.set(f"处理完成：{total} 张全部成功，"
                                            f"可点“打开输出文件夹”查看结果")
        except queue.Empty:
            pass
        self.after(80, self._drain_queue)

    # ===================================================================
    # 收尾
    # ===================================================================
    def _open_output(self):
        d = self.last_dir or core.INPUT_DIR
        if not os.path.isdir(d):
            messagebox.showwarning("提示", "输出目录还不存在。")
            return
        try:
            os.startfile(d)  # Windows
        except AttributeError:
            try:
                import subprocess
                subprocess.Popen(["xdg-open", d])
            except Exception as e:
                messagebox.showwarning("提示", f"无法自动打开目录:\n{e}\n目录为: {d}")

    def _on_close(self):
        if self.running and not messagebox.askyesno("确认", "正在分析，确定要退出吗？"):
            return
        self.destroy()


def main():
    """--selftest：免窗口自检（不弹 GUI），结果写入 exe 同目录 selftest_ok.txt。"""
    if "--selftest" in sys.argv[1:]:
        probe = os.environ.get("PR_PROBE_IMG", "pr2.tif")
        lines = [f"numpy={np.__version__}  cv2={cv2.__version__}", f"cwd={os.getcwd()}"]
        try:
            if os.path.isfile(probe):
                img, binary = core.preprocess_image(probe)
                lines.append(f"preprocess ok: shape={img.shape}")
            else:
                lines.append(f"probe image not found: {probe}")
            cols, _ = core._detect_columns(img, binary)
            for i, c in enumerate(cols, 1):
                c['label'] = i
                c['color'] = core.PR_COLORS[(i - 1) % len(core.PR_COLORS)]
            main_img = core.render_main_image(img, cols, nm_per_px=1.0)
            lines.append(f"render_main_image ok: main={main_img.shape} n_col={len(cols)}")
            got = {}
            core.process_image(probe, on_image=lambda o, m, n: got.update(img=o.shape, main=m.shape, name=n))
            lines.append(f"process_image on_image ok: {got}")
            # ROI 框内套索链路自检（用图像中部区域）
            core.process_image_roi(probe, (60, 110, 600, 350),
                                   on_image=lambda o, m, n: got.update(roi=(o.shape, m.shape)))
            lines.append(f"process_image_roi ok: {got}")
            if os.path.isfile("pr2/pr2_tool_boundaries_nm.txt"):
                st = core.analyze_boundaries_file("pr2/pr2_tool_boundaries_nm.txt")
                lines.append(f"analyze ok: cols={st['cols']} n_items={len(st['rows'])} "
                             f"means[0]={st['means'][0]}")
        except Exception as e:
            lines.append(f"FAILED: {e}")
        if getattr(sys, "frozen", False):
            out_dir = os.path.dirname(os.path.abspath(sys.executable))
        else:
            out_dir = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(out_dir, "selftest_ok.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    App().mainloop()


if __name__ == "__main__":
    main()
