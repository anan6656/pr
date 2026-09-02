# -*- coding: utf-8 -*-
"""
PR 柱子自动识别与测量工具 —— 单文件版（GUI）

本文件已内置全部分析核心，不再依赖其他 .py：
  · 直接运行：python pr2_gui.py
  · 打包单 exe：PyInstaller --onefile --windowed pr2_gui.py

布局：左=参数 | 中=原始图(框选/直线量测/角度量测)+识别涂色图 | 右=统计分析表 | 底部=进度行
生成文件命名：工具自动整图=_tool；用户框选套索=_user。
"""
import os
import sys
import glob
import math
import queue
import shutil
import threading
import traceback
import contextlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from typing import Any

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
THRESHOLD_TOL = 0  # 灰度容差：允许灰度略低于 THRESHOLD 的像素也被选入柱子区域，满足“灰度值大于或接近阈值”
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
    lower = max(0, THRESHOLD - THRESHOLD_TOL)
    _, binary = cv2.threshold(blur, lower, 255, cv2.THRESH_BINARY)
    bright_frac = binary.mean() / 255.0
    # 固定阈值失效保护：亮像素占比过低/过高说明阈值不适配当前图像，自动改用 Otsu
    if bright_frac < 0.01 or bright_frac > 0.9:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"[warn] 固定阈值(THRESHOLD={THRESHOLD}, TOL={THRESHOLD_TOL})结果异常(亮像素占比={bright_frac:.2%})，已自动改用 Otsu")
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
    keys = ('height', 'cd_top', 'cd_mid', 'cd_bot',
            'ratio_tm', 'ratio_td', 'ratio_md',
            'angle_l_top', 'angle_l_mid', 'angle_l_bot',
            'angle_r_top', 'angle_r_mid', 'angle_r_bot',
            'ler_left', 'ler_right')
    out: dict[str, Any] = {k: None for k in keys}   # 各键随后被填入数值/角度，用 Any 放宽类型
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
    ("THRESHOLD_TOL",   "灰度容差",       "int",   0,   50,    1),
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

# 每个参数的详细说明（帮助 → 参数说明，尽量口语化 + 给调参建议）
PARAM_DESC = {
    "THRESHOLD": "【柱子有多亮才算“柱子”？】程序把图像看成黑白：灰度高于该值的像素当成柱子材料，"
                 "低于的当成背景。调到偏暗柱不完整 → 调低；四周亮噪声一大片 → 调高。"
                 "正常可不动；程序发现全图过亮/过暗时也会自动改用自适应阈值兜底。",
    "THRESHOLD_TOL": "【灰度容差 / “接近阈值也算柱子”】灰度略低于“二值化阈值”但在容差范围内的像素，"
                     "也会被当成柱子材料选中。=0 即严格大于阈值；>0 会把阈值附近较暗的过渡像素包含进来，"
                     "让柱子更完整，但太大容易把背景噪声包进来。",
    "MORPH_CLOSE_K": "【给柱子“补洞去毛”的力度】柱子内部小黑点、侧壁毛刺会被填平修整，看起来更干净。"
                     "柱体很粗壮 → 可以加大；柱与柱很近（间隔小于此值）会把两根黏成一根 → 要调小。",
    "SMOOTH_KERNEL": "【剖面曲线的平滑度】把整幅图沿水平方向“压扁”成一条亮度剖面后先平滑再找柱子分界。"
                     "数值越大曲线越平滑。误把一根柱顶部的小凹陷切成两根 → 加大；柱子彼此贴得很近分不开 → 调小。",
    "VALLEY_RATIO": "【多低的“山谷”才切开】以剖面最高峰为 1，亮度跌到该比例以下才认为是柱与柱之间的谷槽。"
                    "想要把靠在一起的柱多切几根 → 调大；一根柱老被多切一刀 → 调小。",
    "MIN_COLUMN_HEIGHT": "【太矮的不算柱】高度低于该值的亮块（刻度、文字、噪声）直接忽略。"
                         "把杂物当成了柱 → 调大；真正的短柱被丢掉了 → 调小。",
    "MIN_AVG_WIDTH": "【太窄的不算柱】平均宽度低于该值的细条（字母笔画、亮线）忽略。"
                     "看到统计里多出“文字柱” → 调大（但别超过真实柱宽）。",
    "MIN_ASPECT_RATIO": "【扁片不算柱】高度 ÷ 平均宽度 小于该值说明是横躺的亮斑而不是竖立的柱，排除。"
                        "底座的亮带被当成柱 → 调大。",
    "BOTTOM_OFFSET": "【给柱底“加裙边”】把识别出的柱底再向下多延伸几像素，让底边落在基座上、轮廓更整齐。"
                     "底部边缘参差不齐可适当调大。",
    "EDGE_MARGIN": "【四边留白】把图像四周这一宽度的边裁掉并置黑，防止黑框、标尺、白边混进识别结果。",
    "WINDOW_MARGIN": "【追踪侧壁的容错】逐行向下扫描时，允许当前行相对上一行向左右“飘”多少像素仍算同一根柱。"
                     "柱壁边缘模糊/断续 → 加大；容易串到隔壁柱子 → 调小。",
    "SCALE_PIXELS": "【比例尺占多少像素】先量出图上标尺的长度（用“直线量测”工具可直接量），把像素数填这里。"
                    "例如标尺标注 50 nm、量得 100 px，就填 100。",
    "SCALE_NM": "【标尺对应多少纳米】与上面同一段标尺的真实长度。换算比例会自动算出（见下方“1 px = xx nm”）。"
                "比例尺只影响 nm 版结果与统计表，不影响找柱子。",
    "OUTPUT_SUBDIR": "【结果放哪里】勾选 = 每张图一个同名文件夹；不勾 = 全部直接放图像文件夹根目录。",
    "DEBUG_OUTPUT": "【出图诊断】勾选后会额外保存一张“二值化中间图”（*_debug_binary.png），"
                    "可以一眼看出阈值取的是否合适，方便调参。",
}

# 生成文件的来源标记说明（帮助 → 输出文件说明 里的简短行）
FILE_TAG_HELP = (
    "文件名中的 _tool 表示“工具自动整图分析”生成，_user 表示“框选套索分析”生成；\n"
    "两类结果都放在同一张图对应的输出目录里，互不覆盖。\n"
    "· 自动(tool)：xx_tool_result.png / xx_tool_boundaries.txt / xx_tool_stats_nm.txt\n"
    "· 套索(user)：xx_user_result.png / xx_user_boundaries.txt / xx_user_stats_nm.txt"
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
    _img_x = _img_y = 0  # 类级声明，便于类型检查识别

    def __init__(self):
        super().__init__()
        self.title("PR 柱子自动识别与测量工具")
        self.geometry("1640x960")
        self.minsize(1280, 720)

        self.q = queue.Queue()
        self.running = False
        self.last_dir = core.INPUT_DIR       # 最后处理的目录（“打开输出文件夹”用）
        # 预览状态：原始图 / 工具自动 / 人工（框选套索）分别存放
        self._orig_bgr = None                # 原始灰度转 BGR，画布显示
        self._tool_bgr = None                # 工具自动识别彩色预览
        self._user_bgr = None                # 人工（框选套索）彩色预览
        self._photos = []                    # 保存 tk.PhotoImage 引用，防止被回收
        self._resize_job = None
        self._live_job = None                # 参数实时重识别的防抖任务
        self._scale = 1.0                    # 当前原图显示缩放（canvas 像素 → 原图像素）
        self._ow = 0                         # 当前原图尺寸（首次加载图像后更新）
        self._oh = 0
        self._roi = None                     # 框选矩形：整图像素坐标 (x0,y0,x1,y1)
        self._drag = None                    # 正在拖拽画框
        self._drag_id = None                 # 拖拽过程中的临时矩形
        self._m1 = None                      # 直线量测：第一点（原图像素）
        self._m2 = None                      # 直线量测：第二点（原图像素）
        self._mv_id = None                   # 量测过程中的临时连线
        self._ang = []                       # 角度量测：三个点 (A端点, 顶点V, C端点)
        self._last_img_name = None           # 当前预览的 tif 文件名（basename）
        self._last_out_tag = "_tool"         # 最近结果来源：_tool / _user
        self._last_manual_tag = "_user"      # 最近一次人工结果（框选套索 _user）
        self._last_stat_src = {"tool": None, "user": None}   # 最近加载的统计源文件
        self._analyzed = []                  # 本会话已分析的 stem（文件列表用）

        self._build_ui()
        self._refresh_files()
        self.last_dir = self._folder()   # 打开输出文件夹：默认跟随界面所选文件夹
        # 左侧识别/比例尺参数一旦变化 → 防抖 300ms 后自动对当前图重新跑一次自动识别
        for _var in list(self.spins.values()) + list(self.scale_vars.values()):
            _var.trace_add("write", self._schedule_live)
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
        # ---- 全局观感（仅调字体/标题配色，不影响任何图像坐标计算） ----
        try:
            st = ttk.Style(self)
            st.configure(".", font=("Microsoft YaHei UI", 9))
            st.configure("TLabelframe.Label",
                         font=("Microsoft YaHei UI", 9, "bold"),
                         foreground="#175a7a")
            st.configure("TCheckbutton", font=("Microsoft YaHei UI", 9))
            st.configure("TRadiobutton", font=("Microsoft YaHei UI", 9))
            st.configure("TButton", font=("Microsoft YaHei UI", 9))
            st.configure("Tool.TLabelframe.Label",
                         font=("Microsoft YaHei UI", 9, "bold"),
                         foreground="#0f7b45")
            st.configure("User.TLabelframe.Label",
                         font=("Microsoft YaHei UI", 9, "bold"),
                         foreground="#8a4b08")
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

        # ---- 已分析文件（左下角）----
        hf = ttk.LabelFrame(left, text="已分析文件（单击载入该图，可继续框选 / 量测）")
        hf.pack(fill="both", expand=True, padx=4, pady=(8, 4))
        f_body = ttk.Frame(hf)
        f_body.pack(fill="both", expand=True, padx=2, pady=2)
        self.lst_files = tk.Listbox(f_body, height=6, exportselection=False,
                                    font=("Microsoft YaHei UI", 9),
                                    relief="flat", highlightthickness=1,
                                    highlightbackground="#c9d4dc")
        self.lst_files.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=4)
        f_sb = ttk.Scrollbar(f_body, orient="vertical", command=self.lst_files.yview)
        f_sb.pack(side="left", fill="y", pady=4)
        self.lst_files.configure(yscrollcommand=f_sb.set)
        self.lst_files.bind("<<ListboxSelect>>", self._on_pick_file)

        # ---- 第二列：工具栏 + 原始图(canvas) + 双预览(tool/user) + 已分析文件列表 ----
        self.frm_mid = ttk.Frame(body)
        self.frm_mid.pack(side="left", fill="both", expand=True, padx=6)
        # row1(原始图)/row2(双预览) 各占半屏；row0=工具条、row3=文件列表固定高度
        self.frm_mid.grid_rowconfigure(1, weight=1)
        self.frm_mid.grid_rowconfigure(2, weight=1)
        self.frm_mid.grid_columnconfigure(0, weight=1)

        roibar = ttk.Frame(self.frm_mid)
        roibar.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        roibar.grid_columnconfigure(0, weight=1)

        # —— 工具单选行 ——
        tools = ttk.Frame(roibar)
        tools.grid(row=0, column=0, sticky="w")
        self.var_canvas_tool = tk.StringVar(value="roi")
        ttk.Radiobutton(tools, text="框选区域", value="roi", variable=self.var_canvas_tool,
                        command=self._on_tool_change).pack(side="left")
        ttk.Radiobutton(tools, text="直线量测", value="dist", variable=self.var_canvas_tool,
                        command=self._on_tool_change).pack(side="left", padx=(10, 0))
        ttk.Radiobutton(tools, text="角度量测", value="angle", variable=self.var_canvas_tool,
                        command=self._on_tool_change).pack(side="left", padx=(10, 0))
        btns = ttk.Frame(roibar)
        btns.grid(row=0, column=1, sticky="e")
        self.btn_roi_run = ttk.Button(btns, text="▶ 套索分析所选区域",
                                      command=self._start_roi)
        self.btn_roi_run.pack(side="right")
        self.btn_roi_clear = ttk.Button(btns, text="清除标记", command=self._clear_canvas_marks,
                                        width=8)
        self.btn_roi_clear.pack(side="right", padx=(0, 8))

        # —— 提示行 ——
        self.lbl_roi_info = ttk.Label(roibar, text="提示：在原始图上按住左键拖出矩形框选分析区域",
                                      foreground="#555555")
        self.lbl_roi_info.grid(row=1, column=0, columnspan=2, sticky="w", padx=2, pady=(2, 0))

        # —— 原始图画布（框选/量测都在上面进行）——
        self.frm_orig = ttk.LabelFrame(self.frm_mid,
                                       text="原始图像（工具在上方切换：框选区域 / 直线量测 / 角度量测）")
        self.frm_orig.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.cv_orig = tk.Canvas(self.frm_orig, bg="#f2f4f7", highlightthickness=0,
                                 cursor="crosshair")
        self.cv_orig.pack(fill="both", expand=True)
        self._bind_canvas()

        # —— 第二行：左=工具自动预览，右=人工预览，各自带保存按钮 ——
        self.frm_results = ttk.Frame(self.frm_mid)
        self.frm_results.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)
        self.frm_results.grid_columnconfigure(0, weight=1)
        self.frm_results.grid_columnconfigure(1, weight=1)
        self.frm_results.grid_rowconfigure(0, weight=1)

        self.frm_tool = ttk.LabelFrame(self.frm_results, text="工具自动识别预览（_tool）",
                                       style="Tool.TLabelframe")
        self.frm_tool.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        self.lbl_tool = tk.Label(self.frm_tool, text="（尚无自动识别结果）", bg="white",
                                 fg="#999999")
        self.lbl_tool.pack(fill="both", expand=True)
        self.btn_save_tool = ttk.Button(self.frm_tool, text="保存本图识别结果…",
                                        command=lambda: self._save_preview_img("tool"))
        self.btn_save_tool.pack(pady=(0, 4))

        self.frm_user = ttk.LabelFrame(self.frm_results, text="人工识别预览（_user）",
                                       style="User.TLabelframe")
        self.frm_user.grid(row=0, column=1, sticky="nsew", padx=(2, 0))
        self.lbl_user = tk.Label(self.frm_user, text="（尚无人工识别结果）", bg="white",
                                 fg="#999999")
        self.lbl_user.pack(fill="both", expand=True)
        self.btn_save_user = ttk.Button(self.frm_user, text="保存本图识别结果…",
                                        command=lambda: self._save_preview_img("user"))
        self.btn_save_user.pack(pady=(0, 4))

        self.frm_mid.bind("<Configure>", self._on_mid_resize)

        # ---- 第三列：统计分上下两块（上=自动 tool，下=人工 user），各自带保存按钮 ----
        # 关键：该列宽度不能随表格内容自由撑大，否则统计列数一多就会“推挤”中间图像列。
        # pack_propagate 只对 pack 子部件生效，这里子部件用 grid 摆放，
        # 因此要同时关掉 grid 传播，宽度才会真正锁死在 width=460。
        # 当内容真的放不下时，会自动把窗口整体向右加宽，而不是挤压中列。
        right = ttk.Frame(body, width=460)
        right.pack_propagate(False)
        right.grid_propagate(False)
        right.pack(side="left", fill="y", padx=(6, 0))
        right.grid_rowconfigure(0, weight=3)
        right.grid_rowconfigure(1, weight=2)
        right.grid_columnconfigure(0, weight=1)
        self._right_frame = right

        self._stat_blocks = {}
        self._stats_trees = {}
        for _i, (_k, _t) in enumerate((("tool", "上：工具自动识别统计（_tool）"),
                                       ("user", "下：人工识别统计（_user）"))):
            blk = self._make_stats_block(right, _t)
            blk["frame"].grid(row=_i, column=0, sticky="nsew", padx=2,
                              pady=(0, 2) if _i else (0, 2))
            blk["save"].configure(command=lambda k=_k: self._save_stats_txt(k))
            self._stat_blocks[_k] = blk
            self._stats_trees[_k] = blk["tree"]
        # 兼容旧的直接引用（仍指向“上：自动”面板）
        self.tree = self._stat_blocks["tool"]["tree"]
        self.var_stat_title = self._stat_blocks["tool"]["var"]
        self.var_stat_title_user = self._stat_blocks["user"]["var"]

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
    # 第三列统计面板构建（上下两块共用）
    # ===================================================================
    def _make_stats_block(self, parent, title):
        """返回 dict：frame / tree / var / save，供上下两块统计面板使用。"""
        frame = ttk.LabelFrame(parent, text=title)
        var = tk.StringVar(value="（尚无结果）")
        ttk.Label(frame, textvariable=var, foreground="#555555",
                  wraplength=420).pack(anchor="w", padx=6, pady=(2, 0))
        tcol = ttk.Frame(frame)
        tcol.pack(fill="both", expand=True, padx=4, pady=2)
        tree = ttk.Treeview(tcol, columns=("avg",), show="tree headings",
                            selectmode="none")
        tree.heading("#0", text="计算项")
        tree.heading("avg", text="")
        tree.column("#0", width=205, minwidth=150, anchor="w", stretch=True)
        tree.column("avg", width=80, minwidth=70, anchor="e", stretch=False)
        tree.tag_configure("cat", font=("Microsoft YaHei UI", 9, "bold"),
                           background="#e8f0e8")
        tree.tag_configure("odd", background="#f4f7f4")
        ys = ttk.Scrollbar(tcol, orient="vertical", command=tree.yview)
        xs = ttk.Scrollbar(tcol, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tcol.grid_rowconfigure(0, weight=1)
        tcol.grid_columnconfigure(0, weight=1)
        foot = ttk.Frame(frame)
        foot.pack(fill="x", padx=4, pady=(0, 2))
        save = ttk.Button(foot, text="保存该统计表另存为…", width=20)
        save.pack(side="left")
        ttk.Label(foot, text="列多时用底部横条滚动",
                  foreground="#888888").pack(side="left", padx=6)
        return {"frame": frame, "tree": tree, "var": var, "save": save}

    # ===================================================================
    # 左侧参数实时刷新（防抖 300ms → 重新自动识别当前图）
    # ===================================================================
    def _schedule_live(self, *_args):
        if self.running or not self._last_img_name:
            return
        # 输入尚未成为合法数值时不触发，避免打字过程中反复弹错
        for _var in list(self.spins.values()) + list(self.scale_vars.values()):
            try:
                float(_var.get().strip().replace("，", ".").replace(",", "."))
            except ValueError:
                return
        if self._live_job is not None:
            self.after_cancel(self._live_job)
        self._live_job = self.after(300, self._run_live)

    def _run_live(self):
        self._live_job = None
        if self.running or not self._last_img_name:
            return
        p = os.path.join(self._folder(), self._last_img_name)
        if not os.path.isfile(p):
            return
        self._start_jobs([{"path": p, "mode": "full", "roi": None}],
                         f"参数变化 → 自动重识别：{self._last_img_name} …")

    # ===================================================================
    # 保存按钮（预览图 / 统计表）
    # ===================================================================
    def _save_preview_img(self, kind):
        if kind == "tool":
            bgr, tag = self._tool_bgr, "_tool"
        else:
            bgr, tag = self._user_bgr, (self._last_manual_tag or "_user")
        if bgr is None or not self._last_img_name:
            messagebox.showwarning("提示", "当前没有对应的识别预览可保存。")
            return
        stem = os.path.splitext(self._last_img_name)[0]
        out_dir = os.path.join(self._folder(), stem) if self.var_subdir.get() else self._folder()
        os.makedirs(out_dir, exist_ok=True)
        dflt = os.path.join(out_dir, f"{stem}_{tag[1:]}_preview.png")
        p = filedialog.asksaveasfilename(
            title="保存识别结果预览图",
            initialdir=out_dir,
            initialfile=os.path.basename(dflt),
            defaultextension=".png",
            filetypes=[("PNG 图像", "*.png")])
        if p:
            cv2.imwrite(p, bgr)
            self.var_status.set(f"预览图已保存：{p}")

    def _save_stats_txt(self, kind):
        src = self._last_stat_src.get(kind)
        if not src or not os.path.isfile(src):
            messagebox.showwarning("提示", "该侧目前没有可保存的统计结果。")
            return
        dflt = os.path.splitext(src)[0] + "_copy.txt"
        p = filedialog.asksaveasfilename(
            title=f"保存{ '自动' if kind == 'tool' else '人工'}统计表",
            initialdir=os.path.dirname(src) or self._folder(),
            initialfile=os.path.basename(dflt),
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
        if p:
            try:
                shutil.copyfile(src, p)
                self.var_status.set(f"统计表已保存：{p}")
            except Exception as e:
                messagebox.showerror("保存失败", str(e))

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
        # 目录切换/启动时：把磁盘上已有 _tool/_user 结果的 stem 并入列表
        try:
            for s in self._scan_disk_stems():
                if s not in self._analyzed:
                    self._analyzed.append(s)
        except Exception:
            pass
        self._refresh_analyzed_list()

    def _scan_disk_stems(self):
        """扫描当前图像文件夹中已生成过 _tool/_user 结果图的 stem。"""
        folder = self._folder()
        if not os.path.isdir(folder):
            return []
        stems = []
        for p in self._list_tifs():
            stem = os.path.splitext(os.path.basename(p))[0]
            cand_dirs = [folder]
            if core.OUTPUT_SUBDIR:
                cand_dirs.append(os.path.join(folder, stem))
            found = False
            for d in cand_dirs:
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    if fn.startswith(f"{stem}_") and fn.endswith("_result.png"):
                        found = True
                        break
                if found:
                    break
            if found:
                stems.append(stem)
        return stems

    def _refresh_analyzed_list(self):
        if not hasattr(self, "lst_files"):
            return
        names = []
        for s in getattr(self, "_analyzed", []):
            if s not in names:
                names.append(s)
        self.lst_files.delete(0, "end")
        for s in names:
            self.lst_files.insert("end", s)

    def _on_pick_file(self, _evt=None):
        if self.running:
            return
        sel = self.lst_files.curselection()
        if not sel:
            return
        stem = self.lst_files.get(sel[0])
        # 找出对应 tif 名称（可能有 .tif/.tiff 两种扩展名）
        match = [n for n in (self.combo["values"] or []) if n.startswith(stem)]
        if not match:
            return
        self.var_sel.set(match[0])
        self._start_single()

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

    def _set_preview(self, tag, orig_bgr, paint_bgr, name):
        # 切换到新图像时，作废旧图上的框选/量测标记；
        # 同一张图的“自动 vs 人工”结果交替到达时保留标记。
        changed = self._last_img_name != name
        if changed:
            self._roi = None
            self._m1 = self._m2 = None
            self._ang = []
            self._user_bgr = None                # 上一张图的人工预览作废
        self._last_img_name = name
        self._last_out_tag = tag
        self._orig_bgr = orig_bgr
        if tag == "_tool":
            self._tool_bgr = paint_bgr
            if self._user_bgr is None:
                self._try_load_manual_from_disk()   # 磁盘上若已有 _user 结果则一并显示
        else:                                    # _user（框选套索）
            self._user_bgr = paint_bgr
            self._last_manual_tag = "_user"
        stem = os.path.splitext(os.path.basename(name))[0]
        if stem not in self._analyzed:
            self._analyzed.append(stem)
        self._refresh_analyzed_list()
        self._redraw_preview()
        self._refresh_stats_kind("tool")
        self._refresh_stats_kind("user")

    def _try_load_manual_from_disk(self):
        """当前图若磁盘上已有 _user 人工结果图，则读进来显示在人工预览格。"""
        if not self._last_img_name:
            return
        stem = os.path.splitext(self._last_img_name)[0]
        base = self._folder()
        cand_dirs = [base]
        if core.OUTPUT_SUBDIR:
            cand_dirs.insert(0, os.path.join(base, stem))
        for d in cand_dirs:
            p = os.path.join(d, f"{stem}_user_result.png")
            if os.path.isfile(p):
                bgr = cv2.imread(p)
                if bgr is not None:
                    self._user_bgr = bgr
                    self._last_manual_tag = "_user"
                return

    def _put_stream(self, kind):
        """把 tool/user 其中一路预览 BGR 缩放到对应 label。"""
        if kind == "tool":
            bgr = self._tool_bgr
            frm, lbl = self.frm_tool, self.lbl_tool
            note = "自动识别结果"
            frmlbl = "工具自动识别预览（_tool）"
        else:
            bgr = self._user_bgr
            frm, lbl = self.frm_user, self.lbl_user
            note = "人工识别结果"
            frmlbl = "人工识别预览（_user）"
        if bgr is None:
            lbl.configure(image="", text=f"（尚无{note}：请运行自动识别 / 框选套索）")
            frm.configure(text=frmlbl)
            return
        iw, ih = bgr.shape[1], bgr.shape[0]
        # 容器实际尺寸（label 与框架之间留少量边距）
        w = max(int(frm.winfo_width()) - 8, 80)
        h = max(int(frm.winfo_height()) - 44, 60)
        s = min(1.0, w / iw, h / ih)
        s = max(s, 0.02)
        draw = bgr
        if s < 1.0:
            draw = cv2.resize(draw, (max(int(iw * s), 1), max(int(ih * s), 1)),
                              interpolation=cv2.INTER_AREA)
        ph = self._bgr_to_photo(draw) or tk.PhotoImage(width=1, height=1)
        self._photos.append(ph)
        lbl.configure(image=ph, text="")
        frm.configure(text=f"{frmlbl}  {self._last_img_name or ''}  ({iw}×{ih})")

    def _redraw_preview(self):
        self._resize_job = None
        self.cv_orig.delete("all")
        self._photos = []
        if self._orig_bgr is None:
            self._placeholder()
            self.lbl_tool.configure(image="", text="（尚无自动识别结果）")
            self.lbl_user.configure(image="", text="（尚无人工识别结果）")
            return
        self.update_idletasks()
        ow, oh = self._orig_bgr.shape[1], self._orig_bgr.shape[0]
        self._ow, self._oh = ow, oh
        cw = max(self.cv_orig.winfo_width() - 4, 60)
        ch = max(self.cv_orig.winfo_height() - 4, 60)
        s = min(1.0, cw / ow, ch / oh)
        s = max(s, 0.02)
        self._scale = s
        draw = self._orig_bgr
        if s < 1.0:
            draw = cv2.resize(draw, (max(int(ow * s), 1), max(int(oh * s), 1)),
                              interpolation=cv2.INTER_AREA)
        ph = self._bgr_to_photo(draw) or tk.PhotoImage(width=1, height=1)
        self._photos.append(ph)
        x = 0
        y = 0
        self._img_x = x
        self._img_y = y
        self.cv_orig.create_image(x, y, anchor="nw", image=ph)
        self.frm_orig.configure(
            text=f"原始图像（工具在上方切换：直线量测 / 角度量测 / 框选区域）  {self._last_img_name or ''}"
                 f"  ({ow}×{oh})")
        self._put_stream("tool")
        self._put_stream("user")
        self._redraw_marks()

    def _placeholder(self):
        w = max(self.cv_orig.winfo_width(), 200)
        h = max(self.cv_orig.winfo_height(), 100)
        self.cv_orig.create_text(w // 2, h // 2,
                                 text="（先在上方选择一张 tif 并“分析选中图像”；\n"
                                      "    也可载入后：框选区域 做人工识别）",
                                 fill="#999999", font=("Microsoft YaHei UI", 10),
                                 justify="center")

    def _on_mid_resize(self, _evt=None):
        if self._orig_bgr is not None and self._resize_job is None:
            self._resize_job = self.after(120, self._redraw_preview)

    # ===================================================================
    # 第二列画布交互：框选区域(roi) / 直线量测(dist) / 角度量测(angle)
    # 直线与角度量测打点时带“水平/竖直磁吸”：当新点与上一点连线接近
    # 水平或竖直（±5°内）时，自动吸附成完全水平/竖直，方便对准柱壁等特征。
    # ===================================================================
    def _bind_canvas(self):
        self.cv_orig.bind("<ButtonPress-1>", self._cv_press)
        self.cv_orig.bind("<B1-Motion>", self._cv_move)
        self.cv_orig.bind("<ButtonRelease-1>", self._cv_release)

    def _cv_to_orig(self, ex, ey):
        """canvas 显示坐标 → 原图像素坐标（越界自动裁剪到图像内）。"""
        ox = min(max(int((ex - self._img_x) / self._scale), 0), self._ow - 1)
        oy = min(max(int((ey - self._img_y) / self._scale), 0), self._oh - 1)
        return ox, oy

    def _orig_to_cv(self, x, y):
        """原图像素坐标 → canvas 显示坐标（含居中偏移）。"""
        return x * self._scale + self._img_x, y * self._scale + self._img_y

    def _tool(self):
        return self.var_canvas_tool.get()

    def _on_tool_change(self):
        """切换工具：只保留当前工具所需的标记。"""
        self._drag = None
        t = self._tool()
        if t == "roi":
            self._m1 = self._m2 = None
            self._ang = []
        elif t == "dist":
            self._roi = None
            self._ang = []
        elif t == "angle":
            self._roi = None
            self._m1 = self._m2 = None
        self._redraw_marks()

    def _clear_canvas_marks(self):
        self._roi = None
        self._m1 = self._m2 = None
        self._ang = []
        self._drag = None
        self._redraw_marks()

    # ---- 水平/竖直磁吸 ----
    def _snap_point(self, base, target):
        """base→target 连线与水平/竖直夹角 ≤5° 时吸附成完全水平/竖直。

        返回 (snapped_point, mode)，mode ∈ {'H','V',None}。
        """
        bx, by = base
        tx, ty = target
        dx, dy = tx - bx, ty - by
        if dx == 0 and dy == 0:
            return target, None
        ang = math.atan2(abs(dy), abs(dx))          # 0~90°（与水平线的夹角）
        tol = math.radians(5.0)
        if ang < tol:
            return (tx, by), "H"                     # 吸附为水平
        if ang > math.pi / 2 - tol:
            return (bx, ty), "V"                     # 吸附为竖直
        return target, None

    def _marker(self, x, y, fill, outline=None, radius=None):
        s = self._scale
        r = radius if radius is not None else max(int(3 * s), 2)
        cx, cy = self._orig_to_cv(x, y)
        self.cv_orig.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 outline=outline or fill, fill=fill, tags="ovl")

    def _pending_preview(self, base, e, color):
        """画布移动时预览 base → 光标 的连线；接近水平/竖直时显示吸附参考线。"""
        self._redraw_marks()
        s = self._scale
        tx, ty = self._cv_to_orig(e.x, e.y)
        sp, mode = self._snap_point(base, (tx, ty))
        sx, sy = sp
        if mode == "H":                              # 吸附参考：过 snapped 点的水平线
            x0, y0 = self._orig_to_cv(0, sy)
            x1, y1 = self._orig_to_cv(self._ow, sy)
            self.cv_orig.create_line(x0, y0, x1, y1,
                                     fill="#00aa66", width=1, dash=(6, 4), tags="ovl")
            note = "水平 已吸附"
        elif mode == "V":
            x0, y0 = self._orig_to_cv(sx, 0)
            x1, y1 = self._orig_to_cv(sx, self._oh)
            self.cv_orig.create_line(x0, y0, x1, y1,
                                     fill="#00aa66", width=1, dash=(6, 4), tags="ovl")
            note = "竖直 已吸附"
        else:
            note = None
        bx, by = base
        ax, ay = self._orig_to_cv(bx, by)
        cx, cy = self._orig_to_cv(sx, sy)
        self.cv_orig.create_line(ax, ay, cx, cy,
                                 fill=color, width=2, dash=(3, 3), tags="ovl")
        self._marker(sx, sy, "#55ff88", outline="#00804d")
        if note:
            tx, ty = self._orig_to_cv(sx, sy)
            self.cv_orig.create_text(tx, max(ty - 6, self._img_y + 6), text=note,
                                     fill="#00804d", anchor="s",
                                     font=("Microsoft YaHei UI", 9, "bold"),
                                     tags="ovl")

    # ---- 鼠标事件 ----
    def _cv_press(self, e):
        if self.running or self._orig_bgr is None:
            return
        t = self._tool()
        if t == "roi":
            self._drag = (e.x, e.y)
            return
        p = self._cv_to_orig(e.x, e.y)
        if t == "dist":
            if self._m1 is None:
                self._m1, self._m2 = p, None
            elif self._m2 is None:
                self._m2, _ = self._snap_point(self._m1, p)
            else:                                  # 已测完一组 → 开始下一组
                self._m1, self._m2 = p, None
            self._redraw_marks()
            return
        if t == "angle":                            # A端点 → 顶点V → C端点
            if len(self._ang) < 3:
                if self._ang:
                    sp, _ = self._snap_point(self._ang[-1], p)
                    p = sp
                self._ang.append(p)
            else:
                self._ang = [p]
            self._redraw_marks()
            return

    def _cv_move(self, e):
        if self.running or self._orig_bgr is None:
            return
        t = self._tool()
        if t == "roi":
            if self._drag is None:
                return
            self._redraw_marks()                    # 清除旧预览框
            x0, y0 = self._drag
            self.cv_orig.create_rectangle(x0, y0, e.x, e.y,
                                          outline="#ff3333", width=2, dash=(5, 3),
                                          tags="ovl")
            return
        if t == "dist":
            if self._m1 is not None and self._m2 is None:
                self._pending_preview(self._m1, e, "#0099ff")
            return
        if t == "angle":
            if len(self._ang) == 1:
                self._pending_preview(self._ang[0], e, "#cc6600")
            elif len(self._ang) == 2:
                self._pending_preview(self._ang[1], e, "#0099cc")
            return

    def _cv_release(self, e):
        t = self._tool()
        if t == "roi":
            if self._drag is None:
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
            return

    # ---- 覆盖图元绘制 ----
    def _clear_overlays(self):
        self.cv_orig.delete("ovl")

    def _redraw_marks(self):
        """按当前工具把框选矩形 / 量测标记重画到原图画布上。"""
        if self._orig_bgr is None:
            return
        self._clear_overlays()
        t = self._tool()
        if t == "roi" and self._roi is not None:
            x0, y0, x1, y1 = self._roi
            x0, y0 = self._orig_to_cv(x0, y0)
            x1, y1 = self._orig_to_cv(x1, y1)
            self.cv_orig.create_rectangle(x0, y0, x1, y1,
                                          outline="#ff0000", width=2, dash=(5, 3),
                                          tags="ovl")
        elif t == "dist":
            self._draw_dist()
        elif t == "angle":
            self._draw_angle()
        self._update_canvas_info()

    def _draw_dist(self):
        if self._m1 is None:
            return
        ax, ay = self._m1
        self._marker(ax, ay, "#33aaff", outline="#0066ff")
        if self._m2 is not None:
            bx, by = self._m2
            x0, y0 = self._orig_to_cv(ax, ay)
            x1, y1 = self._orig_to_cv(bx, by)
            self.cv_orig.create_line(x0, y0, x1, y1,
                                     fill="#0066ff", width=2, tags="ovl")
            self._marker(bx, by, "#33aaff", outline="#0066ff")
            px = math.hypot(bx - ax, by - ay)
            mx = (x0 + x1) / 2
            my = max(min((y0 + y1) / 2 - 8,
                         self._oh * self._scale + self._img_y - 10),
                     self._img_y + 10)
            self.cv_orig.create_text(mx, my,
                                     text=f"L={px:.2f} px", fill="#0044cc",
                                     font=("Microsoft YaHei UI", 9, "bold"),
                                     tags="ovl")

    def _angle_result(self):
        """三点角 ∠A·V·C（0~180°）。返回 (deg, len1_px, len2_px) 或 None。"""
        if len(self._ang) < 3:
            return None
        (ax, ay), (vx, vy), (cx, cy) = self._ang[:3]
        ux, uy = ax - vx, ay - vy
        wx, wy = cx - vx, cy - vy
        lu, lw = math.hypot(ux, uy), math.hypot(wx, wy)
        if lu == 0 or lw == 0:
            return None
        cos = max(-1.0, min(1.0, (ux * wx + uy * wy) / (lu * lw)))
        return (math.degrees(math.acos(cos)), lu, lw)

    def _draw_angle(self):
        n = len(self._ang)
        if n == 0:
            return
        ax, ay = self._ang[0]
        self._marker(ax, ay, "#ffcc66", outline="#cc6600")
        if n >= 2:
            vx, vy = self._ang[1]
            ax_, ay_ = self._orig_to_cv(ax, ay)
            vx_, vy_ = self._orig_to_cv(vx, vy)
            self.cv_orig.create_line(ax_, ay_, vx_, vy_,
                                     fill="#cc6600", width=2, tags="ovl")
            self._marker(vx, vy, "#ff6666", outline="#cc0000",
                         radius=max(int(4 * self._scale), 3))
            if n >= 3:
                cx, cy = self._ang[2]
                cx_, cy_ = self._orig_to_cv(cx, cy)
                self.cv_orig.create_line(vx_, vy_, cx_, cy_,
                                         fill="#0088cc", width=2, tags="ovl")
                self._marker(cx, cy, "#66ccff", outline="#0066cc")
                res = self._angle_result()
                if res is not None:
                    deg, lu, lw = res
                    # 在顶点附近沿夹角平分线放置角度标注
                    (ax, ay), (vx, vy), (cx, cy) = self._ang[:3]
                    ux, uy = ax - vx, ay - vy
                    wx, wy = cx - vx, cy - vy
                    lu2, lw2 = math.hypot(ux, uy), math.hypot(wx, wy)
                    dx = ux / lu2 + wx / lw2
                    dy = uy / lu2 + wy / lw2
                    dl = math.hypot(dx, dy)
                    if dl < 1e-6:                   # 夹角接近 180°，取垂直方向
                        dx, dy = -uy, ux
                        dl = math.hypot(dx, dy) or 1.0
                    s = self._scale
                    rad = 34 * s
                    tx = min(max(vx * s + dx / dl * rad + self._img_x,
                                 self._img_x + 8),
                             self._ow * s + self._img_x - 8)
                    ty = min(max(vy * s + dy / dl * rad + self._img_y,
                                 self._img_y + 8),
                             self._oh * s + self._img_y - 8)
                    self.cv_orig.create_text(tx, ty, text=f"∠={deg:.1f}°",
                                             fill="#cc0000",
                                             font=("Microsoft YaHei UI", 9, "bold"),
                                             tags="ovl")

    def _update_canvas_info(self):
        """根据当前工具与已绘制内容，更新工具条左侧的提示/结果文字。"""
        t = self._tool()
        if t == "dist":
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
                self.lbl_roi_info.configure(text="直线量测：A 已定，请单击 B（接近水平/竖直自动吸附）",
                                            foreground="#aa5500")
            else:
                self.lbl_roi_info.configure(
                    text="提示：单击 A、B 两点量测距离；接近水平/竖直自动吸附",
                    foreground="#555555")
        elif t == "angle":
            n = len(self._ang)
            res = self._angle_result() if n >= 3 else None
            if res is not None:
                deg, lu, lw = res
                nm1, nm2 = self._px_to_nm(lu), self._px_to_nm(lw)
                txt = (f"角度量测：∠={deg:.1f}°   边长 "
                       f"{lu:.1f}/{lw:.1f} px")
                if nm1 is not None and nm2 is not None:
                    txt += f"（{nm1:.1f}/{nm2:.1f} nm）"
                self.lbl_roi_info.configure(text=txt, foreground="#cc0000")
            elif n == 2:
                self.lbl_roi_info.configure(text="角度量测：顶点 V 已定，请单击 C 点确定另一条边",
                                            foreground="#aa5500")
            elif n == 1:
                self.lbl_roi_info.configure(text="角度量测：A 点已定，请单击顶点 V",
                                            foreground="#aa5500")
            else:
                self.lbl_roi_info.configure(
                    text="提示：角度量测按 A端点 → 顶点V → C端点 依次单击三个点",
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
        core.INPUT_DIR = os.path.normpath(folder)      # pyright: ignore[reportAttributeAccessIssue]
        core.OUTPUT_SUBDIR = bool(self.var_subdir.get())  # pyright: ignore[reportAttributeAccessIssue]
        core.DEBUG_OUTPUT = bool(self.var_debug.get())  # pyright: ignore[reportAttributeAccessIssue]
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
            self.q.put(("img", out_tag, orig, paint, name))
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
        # 批量分析时，旧图的框选/量测标记不应带到下一张图
        if len(jobs) > 1:
            self._roi = None
            self._m1 = self._m2 = None
            self._ang = []
            self._redraw_marks()
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
        if self._roi is None or self._orig_bgr is None:
            messagebox.showwarning("提示", "请先在原始图像上按住左键拖出矩形分析区域。")
            return
        name = self._last_img_name
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
                    _tb = traceback.format_exc().strip().splitlines()
                    _why = _tb[-1] if _tb else "未知错误"
                    fails.append(f"{bname}: {_why}")
        self.q.put(("done", ok, total, fails))

    def _set_running(self, on):
        self.running = on
        state = "disabled" if on else "normal"
        for b in (self.btn_run1, self.btn_run_all, self.btn_open,
                  self.btn_roi_run, self.btn_roi_clear):
            b.configure(state=state)
        self.combo.configure(state="disabled" if on else "readonly")
        self.lst_files.configure(state="disabled" if on else "normal")

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

    def _refresh_stats_kind(self, kind):
        """kind: 'tool'(自动) 或 'user'(人工)，各自读取磁盘统计并填充自己的面板。"""
        stem = os.path.splitext(os.path.basename(self._last_img_name or ""))[0]
        blk = self._stat_blocks[kind]
        var = blk["var"]
        if not stem:
            var.set("尚未分析图像")
            self._last_stat_src[kind] = None
            return
        base = self._folder()
        out_dir = base if not core.OUTPUT_SUBDIR else os.path.join(base, stem)
        if kind == "tool":
            cand = ["_tool"]
            src_name = "工具自动"
        else:
            tag = getattr(self, "_last_manual_tag", None) or "_user"
            # 人工侧统一读 _user 结果
            cand = ["_user"] if tag != "_tool" else ["_tool"]
            src_name = "人工"
        nm_txt = None
        for t in cand:
            p = os.path.join(out_dir, f"{stem}{t}_boundaries_nm.txt")
            if os.path.isfile(p):
                nm_txt = p
                break
        if not nm_txt:
            self._last_stat_src[kind] = None
            var.set(f"{stem}: 尚无{src_name}统计")
            return
        try:
            data = core.analyze_boundaries_file(nm_txt)
            self._populate_stats_tree(kind, data)
            self._last_stat_src[kind] = nm_txt
            var.set(f"当前：{stem}（{src_name}）  ← {os.path.basename(nm_txt)}")
        except Exception as e:
            self._last_stat_src[kind] = None
            var.set(f"统计表读取失败：{e}")

    def _fit_right_to_content(self, min_right=460, max_right=820):
        """统计列数太多时，优先向右扩大窗口，而不是挤压中间图像列。"""
        if not getattr(self, "_right_frame", None):
            return
        self.update_idletasks()
        right = self._right_frame
        try:
            # 上下两块树中较宽者的实际需求（列宽 + 滚动条/边距余量）
            tree_req = 0
            for _t in getattr(self, "_stats_trees", {}).values():
                tree_req = max(tree_req, _t.winfo_reqwidth())
        except Exception:
            return
        desired = max(min_right, tree_req + 34)
        desired = min(desired, max_right)
        cur = right.winfo_width()
        if desired <= cur:
            return
        delta = desired - cur
        # 通过扩大窗口来“向右扩”，而不是占用中列空间
        scr_w = self.winfo_screenwidth()
        x = self.winfo_x()
        y = self.winfo_y()
        w = self.winfo_width()
        h = self.winfo_height()
        if x + w + delta > scr_w - 10:       # 防止超出屏幕右边缘
            delta = max(0, scr_w - 10 - (x + w))
            desired = cur + delta
            if delta <= 0:
                return
        right.configure(width=desired)
        self.geometry(f"{w + delta}x{h}+{x}+{y}")

    def _populate_stats_tree(self, kind, data):
        blk = self._stat_blocks[kind]
        tree, var = blk["tree"], blk["var"]
        tree.delete(*tree.get_children())
        cols, descs, rows, means = data["cols"], data["desc"], data["rows"], data["means"]
        if not cols:
            var.set("没有识别到 PR 柱，无统计")
            return
        col_ids = [f"p{i}" for i in range(len(cols))] + ["avg"]
        tree.configure(columns=col_ids)
        for cid, label in zip(col_ids[:-1], cols):
            tree.heading(cid, text=label)
            tree.column(cid, width=66, minwidth=58, anchor="e", stretch=False)
        tree.heading("avg", text="全部柱平均")
        tree.column("avg", width=88, minwidth=80, anchor="e", stretch=False)
        tree.heading("#0", text="计算项")

        groups = [
            ("尺寸与 CD（nm）", (0, 4)),
            ("CD 比值（无量纲）", (4, 7)),
            ("三段侧壁角（°，与+x轴，可>90°）", (7, 13)),
            ("边缘粗糙度 LER 3σ（nm）", (13, 15)),
        ]
        toggle = 0
        for gname, (s, e) in groups:
            pid = tree.insert("", "end", text=gname, values=[""] * len(col_ids),
                              tags=("cat",))
            for idx in range(s, e):
                vals = [self._fmt_stat(descs[idx], v) for v in rows[idx]]
                vals.append(self._fmt_stat(descs[idx], means[idx]))
                tag = ("odd",) if toggle % 2 else ()
                toggle += 1
                tree.insert(pid, "end", text=descs[idx], values=vals, tags=tag)
            toggle += 1
        self._fit_right_to_content()

    # ===================================================================
    # 帮助窗口
    # ===================================================================
    def _show_params_help(self):
        lines = [
            "软件是怎么找柱子的？（看懂这一小段，参数就好理解了）\n"
            "  ① 把“像柱子的亮块”从背景里挑出来 —— 由【二值化阈值】决定多亮算柱子；\n"
            "  ② 给亮块“补洞、修毛边” —— 由【闭运算核】控制补修力度；\n"
            "  ③ 沿水平方向把剖面压扁，在亮度跌下去的“山谷”处把挨在一起的柱子逐根切开\n"
            "     —— 由【投影平滑核】与【谷值比例】控制切几根、怎么切；\n"
            "  ④ 丢掉“矮的 / 窄的 / 扁的”杂物 —— 由【最小柱高 / 最小平均宽 / 最小高宽比】过滤；\n"
            "  ⑤ 贴齐柱底、避开图框 —— 由【底部延伸 / 边缘忽略】微调；【窗口外扩】管侧壁追踪。\n\n"
            "给新图调参的最快路径：\n"
            "  · 第一件事先填比例尺（若标尺单位不是默认值）；\n"
            "  · 识别不准时勾选“调试：另存二值化中间图”，对照黑白图确认柱子是否完整、是否粘连；\n"
            "  · 之后按下面每条的“何时调大/调小”对症调整，一次只动一个参数。\n\n"
            "────────────────────────────────────────\n"
            "各参数含义（默认值已按样例调好，多数图可直接用）：\n"
        ]
        for attr, name, kind, lo, hi, inc in MAIN_PARAMS:
            lines.append(f"■ {name}  [{attr}]\n"
                         f"   默认 {getattr(core, attr)}，可调范围 {lo}~{hi}，步长 {inc}\n"
                         f"   {PARAM_DESC.get(attr, '')}\n")
        lines.append(f"■ 比例尺换算（SCALE_PIXELS / SCALE_NM）\n"
                     f"   默认 {core.SCALE_PIXELS} px 对应 {core.SCALE_NM} nm → "
                     f"1 px = {core.SCALE_NM / core.SCALE_PIXELS:.4f} nm。\n"
                     f"   只影响 nm 版结果图坐标轴刻度、boundaries_nm / stats_nm 的数值，不影响找柱子。\n\n"
                     f"■ 输出方式\n"
                     f"   · 输出到“同名子文件夹”：每张 tif 的结果放进其同名子文件夹。\n"
                     f"   · 调试中间图：把二值化结果另存一张 png，便于确认阈值。\n"
                     f"   · 两类结果互不覆盖：_tool（自动整图）、_user（框选套索）。\n")
        self._show_doc("调节参数含义", "".join(lines))

    def _show_about(self):
        text = ("PR 柱子自动识别与测量工具\n"
                "作者：E924744 Kang An\n\n"
                "版本更新说明：\n"
                "· 自动整图识别 + 左侧参数实时改动后自动重识别\n"
                "· 框选套索（_user）人工识别\n"
                "· 直线量测、角度量测，接近水平/竖直时自动磁吸\n"
                "· 双路预览 + 右侧上下两块统计表，可分别保存\n"
                "· 已分析文件列表支持继续编辑\n"
                "· “打开输出文件夹”直接打开输出文件夹位置\n\n"
                "第二列三个工具（原始图上单选，自动带水平/竖直磁吸）：\n"
                "  直线量测  — 单击 A、B 两点，量测直线像素长度并按比例尺换算 nm；\n"
                "              拖动时连线接近水平或竖直会自动“吸住”，便于对准柱壁/基底；\n"
                "  角度量测  — 依次单击 A端点 → 顶点V → C端点 三个点，显示夹角与两边长度；\n"
                "  框选区域  — 左键拖出矩形 → 点“▶ 套索分析所选区域”在框内自动定位 PR 柱。\n\n"
                "预览与统计分两路：\n"
                "  工具自动识别 = _tool（左列改参数后自动实时重识别）；\n"
                "  人工识别 = _user（框选套索，点上方“▶ 套索分析所选区域”生成）。\n"
                "下方双预览各自带“保存本图识别结果”按钮；第三列上下两块统计表各有“保存该统计表另存为”按钮；\n"
                "左列下方“已分析文件”单击可载入该图预览并继续操作（批量识别会全部列入）。\n\n"
                "本文件是单文件版：分析核心已内置，不依赖其他 .py，\n"
                "直接 python pr2_gui.py 运行，或用 PyInstaller 只打包本文件即可。")
        box = self._show_doc("关于", text)
        box.configure(state="normal")
        box.tag_add("author", "2.0", "2.end+1c")
        box.tag_configure("author", font=("Microsoft YaHei UI", 12, "bold"))
        box.configure(state="disabled")

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
        return box

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
                    _, tag, orig, paint, name = item
                    self._set_preview(tag, orig, paint, name)
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
            if not os.path.isfile(probe):
                lines.append(f"probe image not found: {probe}（跳过算法自检）")
            else:
                img, binary = core.preprocess_image(probe)
                lines.append(f"preprocess ok: shape={img.shape}")
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
