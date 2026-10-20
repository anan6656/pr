# -*- coding: utf-8 -*-
"""PR profile analyzer v6.3 — 图像识别模块。

包含：Unicode 安全的图像读写（支持 png/jpg/bmp/tif 等常见格式）、
预处理、binary 粗定位、几何提议、动态规划精修与整图/ROI 检测管线。
"""

import csv
import glob
import json
import math
import os
from datetime import datetime

import cv2
import numpy as np

from pr_1_config import (ALPHA,
    AXIS_MARGIN_B, AXIS_MARGIN_L,
    AUTO_CD_ESTIMATE_PX, AUTO_CD_TOLERANCE_PX,
    BOTTOM_PULL_WEAK,
    DEBUG_OUTPUT, GEOMETRIC_EDGE_SMOOTH,
    GEOMETRIC_MIN_EDGE_RESPONSE, GEOMETRIC_MIN_RUN_PX,
    GEOMETRIC_PROFILE_SMOOTH, GEOMETRIC_ROW_END_RATIO,
    GEOMETRIC_ROW_START_RATIO, GEOMETRIC_TRACE_DISTANCE_PENALTY,
    GEOMETRIC_TRACE_STEP_RATIO, INPUT_DIR, INPUT_GLOBS,
    MAX_TRACE_GAP, MIN_ASPECT_RATIO, MIN_AVG_WIDTH, MIN_COLUMN_HEIGHT,
    PANEL_W, PILLAR_POLARITY, PR_COLORS, REFINE_RADIUS_PX,
    LINE_THICKNESS, OUTPUT_SUBDIR, STEP_HELP,
    read_image_scale, validate_scale,
    ENABLE_TOP_V_NOTCH_FIX, ENABLE_ASYMMETRIC_HEAD_TRIM,
    TOP_EXTEND_GAIN_GATE, HEAD_TRIM_SCAN_ROWS,
    TOP_RESCUE_PULL, TOP_RESCUE_GATE,
    ENABLE_WIDE_TOP_SHRINK,
    ENABLE_FOOT_EXTEND_EVIDENCE, FOOT_DROP_GRAY_GATE,
    ENABLE_GRADIENT_FOOT_CLIP, GRADIENT_FOOT_CLIP_MAX,
    BATCH_FAILURE_LOG, ENABLE_BIG_PILLAR_MODE,
)
from pr_3_contour_calc import (analyze_columns, export_boundaries, render_combined,
                             render_main_image, write_stats_file,
                             _detection_meta_from_aux)

# 常见图片格式（GUI 文件对话框与批量扫描共用）
IMAGE_FILE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
                         ".jpe", ".jp2", ".webp", ".pbm", ".pgm", ".ppm",
                         ".sr", ".ras", ".dib")
IMAGE_FILE_GLOBS = tuple(f"*{ext}" for ext in IMAGE_FILE_EXTENSIONS)


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
    """兼容 Windows 中文路径；支持 png/jpg/bmp/tif 等所有 cv2 能解码的格式。"""
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

    # 柱状垂直延续性证据：柱是垂直结构，相隔 lag 行的“亮于行内中位”
    # 掩码仍然对齐（IoU 高）；颗粒衬底、多孔基底与底部信息栏的亮斑是
    # 各向同性颗粒，错位后 IoU 掉到背景水平（≈0.25~0.30）。据此给每个
    # 候选主段打“柱状证据”分：高证据段优先于更长的基底纹理段。
    lag = max(8, min(20, int(round(h * 0.03))))
    med_row = np.percentile(residual, 55, axis=1)
    mask_b = residual > med_row[:, None]
    hh = max(1, h - lag)
    inter = np.logical_and(mask_b[:hh], mask_b[lag:]).sum(axis=1).astype(np.float32)
    union = np.logical_or(mask_b[:hh], mask_b[lag:]).sum(axis=1).astype(np.float32)
    vert = np.zeros(h, dtype=np.float32)
    vert[:hh] = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)
    vert = smooth_1d(vert, 5)

    def _pillar_score(run):
        a, b = run
        m = float(np.mean(vert[a:min(b, h - 1) + 1])) if b >= a else 0.0
        return float(np.clip((m - 0.30) / 0.20, 0.0, 1.0))

    strong = [r for r in runs if r[0] < int(h * 0.70) and _pillar_score(r) >= 0.5]
    if strong:
        # 柱状证据优先：并列时取更长、更靠上的段。
        main = max(strong, key=lambda r: (_pillar_score(r), r[1] - r[0] + 1, -r[0]))
    else:
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

    # 柱顶向上扩展：主段可能从柱身对比最强处才开始，圆头/T-top 顶部
    # 对比度偏低而落在段外。从主段顶向上，行活动度不显著低于背景且
    # 垂直延续性高于基底水平时并入；连续 6 行不满足即停，防止并入
    # 顶部文件名等文字行。
    extend_limit = max(0, top - int(h * 0.30))
    quiet = 0.12 * peak
    miss, y = 0, top - 1
    while y >= extend_limit and miss < 6:
        if activity[y] >= quiet and vert[y] >= 0.32:
            top = y
            miss = 0
        else:
            miss += 1
        y -= 1

    # 主体底部在柱脚与基座融合处会快速失去横向周期结构。
    # end_thr 以主段内部峰值（而非全局峰值）为基准：如 A55 底部
    # 颗粒纹理把全局峰值抬得很高（1.9 vs 柱区 0.85），固定比例阈值
    # （0.37×全局 peak）会把柱体下半段（活动度 0.65~0.69）误判为
    # 融合区截掉，导致柱脚丢失。robust_floor 同步放宽到 0.85 倍，
    # 因为主段内柱体的活动度通常高于全图中位数（含背景噪声行）。
    seg = activity[top:int(main[1]) + 1]
    main_peak = float(seg.max()) if seg.size else peak
    end_thr = main_peak * float(np.clip(GEOMETRIC_ROW_END_RATIO, 0.12, 0.90))
    robust_floor = float(np.median(activity[activity > 0])) * 1.15 \
        if np.any(activity > 0) else 0.0
    end_thr = min(end_thr, max(robust_floor * 0.85, main_peak * 0.12))
    in_main = np.arange(top, int(main[1]) + 1, dtype=np.int32)
    good_end = in_main[activity[in_main] >= end_thr]
    bottom = int(good_end[-1]) if good_end.size else int(main[1])
    if bottom <= top:
        bottom = int(main[1])

    # 柱脚 V 谷截断：衬底颗粒活动度与柱区相当时（A67-4/5 主段被
    # Otsu 连到 y479），主段峰之后的显著 V 形谷是柱体与衬底的真实
    # 分界。谷须满足：不高于前方 6 行最小值的 1.02 倍、低于段峰
    # 0.62、其后 25 行回升超过谷值 1/0.85 倍且达到段峰 0.55（排除
    # 柱内爬升段的浅凹）。
    limit = min(int(main[1]), top + int(h * 0.42))
    seg = activity[top:limit + 1]
    if seg.size:
        seg_max = float(seg.max())
        y = top + int(np.argmax(seg)) + 3
        while y < limit and y + 25 < h:
            win_before = float(np.min(activity[max(0, y - 6):y]))
            after_max = float(np.max(activity[y:y + 25]))
            if (activity[y] <= win_before * 1.02 and activity[y] < seg_max * 0.62
                    and after_max >= seg_max * 0.55 and activity[y] < after_max * 0.85):
                bottom = min(bottom, y)
                break
            y += 1

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
    # 间隙两极分化（小间隙=柱内纹理缺口，大间隙=真实柱间距）时收紧
    # 合并阈值，避免把真实柱间隙合并成连体柱（A67-4/5）；间隙单一
    # 分布时保持保守合并（A55 的柱内缺口仍需合并）。
    gaps = [runs[i][0] - runs[i - 1][1] - 1 for i in range(1, len(runs))] if len(runs) > 1 else []
    tight_gap = max(3, int(round(0.06 * typical)))
    small = [g for g in gaps if g <= tight_gap]
    big = [g for g in gaps if g > tight_gap]
    if big and small and min(big) >= 2.0 * max(small):
        merge_gap = tight_gap
    else:
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
    # 弱证据拆分子柱折扣：拆分缝的帽/身双区谷深弱（<10 灰度级）说明
    # 缝可能是纹理假缝（A66 宽帽柱被误拆），一致性项打 0.9 折，让
    # "不拆的整柱提案"在仲裁中胜出；A67-4 真缝谷深 >=15 不受影响。
    weak_split = any(
        c.get("split_child") and float(c.get("split_evidence", 255.0)) < 10.0
        for c in columns
    )
    cons_term = consistency * (0.9 if weak_split else 1.0)
    return float(
        count_fit
        + 1.45 * width_fit
        + 1.20 * cons_term
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


def _split_merged_geometric_columns(cols, gray, filtered, strict=False):
    """geometric_seed 路径的连体柱二次拆分。

    binary 全败时几何路径可能把 2~3 根真实柱并成一个宽段（间隙在
    归一化剖面里不降到 Otsu 阈值以下）。用无 CLOSE 的 CLAHE Otsu
    mask 在整柱带内逐行统计段内空隙位置：真实柱间隙会在多数行稳定
    出现；从候选缝中选一组使子列宽度最均匀的组合。
    """
    if len(cols) < 2:
        return cols
    widths = sorted(float(c["right"] - c["left"] + 1) for c in cols)
    med_w = widths[len(widths) // 2]
    # 连体判定：柱列宽度离散时（连体+单柱混合），以最小宽为基准。
    # A67-4: [46,167,106,151,42] 真实单柱 ~42-46，连体 106-167。
    base_w = max(widths[0], 8.0)
    merged = [c for c in cols if (c["right"] - c["left"] + 1) >= 1.9 * base_w]
    if not merged:
        return cols
    try:
        enhanced = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(
            np.asarray(filtered, dtype=np.uint8))
        _, raw_mask = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        return cols
    h, w = raw_mask.shape
    out = []
    for col in cols:
        if col not in merged:
            out.append(col)
            continue
        L, R = int(col["left"]), int(col["right"])
        rows = col.get("ranges") or []
        if len(rows) < 12 or R - L + 1 < 24:
            out.append(col)
            continue
        y_top = int(rows[0][0])
        y_bot = int(rows[-1][0])
        ya = max(0, y_top + max(2, int(round((y_bot - y_top + 1) * 0.035))))
        yb_ = min(h - 1, y_bot - 1)
        votes = np.zeros(R - L + 1, dtype=np.float32)
        nrows = 0
        for y in range(ya, yb_ + 1):
            runs = _run_segments(raw_mask[y, L:R + 1] > 0, min_len=3)
            if len(runs) < 2:
                continue
            nrows += 1
            for s0, s1 in _run_segments(raw_mask[y, L:R + 1] == 0, min_len=3):
                votes[s0:s1 + 1] += 1.0
        if nrows < 8:
            out.append(col)
            continue
        need = max(8.0, nrows * (0.7 if strict else 0.4))
        hot = [L + x for x, v in enumerate(votes) if v >= need]
        groups = []
        for x in hot:
            if groups and x - groups[-1][-1] <= 3:
                groups[-1].append(x)
            else:
                groups.append([x])
        # 候选缝：距两端至少 0.12 段宽（避免切出残条）
        margin = int((R - L) * 0.12)
        cands = [(g[0], g[-1]) for g in groups
                 if g[0] - L >= margin and R - g[-1] >= margin]
        if not cands:
            out.append(col)
            continue
        # 真缝必须"双区成谷"：帽区（顶部 15%~35% 高度带）与柱身区
        # （中部 45%~75% 高度带）在原始灰度列均值上都要有暗谷。
        # A66 宽帽柱内部的纹理假缝只在部分行的 CLAHE mask 里出现暗段，
        # 帽区/柱身亮度平坦无谷（67-70/96-100 vs 背景 50）；A67-4 真
        # 实柱缝贯穿帽与身（谷深 >=15 灰度级）。任一区无谷即废弃该缝。
        band_h = y_bot - y_top + 1
        cap_rows = gray[y_top + int(band_h * 0.15):y_top + int(band_h * 0.35) + 1, :]
        body_rows = gray[y_top + int(band_h * 0.45):y_top + int(band_h * 0.75) + 1, :]

        def _band_dip(band_rows, seam_lo, seam_hi):
            """缝 x 在指定行带内的谷深（灰度级）。无谷返回 0。"""
            if band_rows.shape[0] < 3:
                return 0.0
            prof = np.convolve(
                band_rows[:, L:R + 1].astype(np.float32).mean(axis=0),
                np.ones(5) / 5.0, mode="same")
            off = max(3, int((R - L + 1) * 0.10))
            s0, s1 = seam_lo - L, seam_hi - L
            lo_a, hi_b = max(0, s0 - off), min(len(prof) - 1, s1 + off)
            if lo_a >= hi_b:
                return 0.0
            peak = max(float(np.max(prof[lo_a:s0 + 1])) if s0 > lo_a else 0.0,
                       float(np.max(prof[s1:hi_b + 1])) if hi_b > s1 else 0.0)
            if peak <= 0:
                return 0.0
            dip = peak - float(np.min(prof[s0:s1 + 1]))
            return dip if dip > 0 else 0.0

        seam_need = 6.0
        valid_cands = []
        seam_evidence = {}
        for seam_lo, seam_hi in cands:
            cap_dip = _band_dip(cap_rows, seam_lo, seam_hi)
            body_dip = _band_dip(body_rows, seam_lo, seam_hi)
            seam_evidence[(seam_lo, seam_hi)] = (cap_dip, body_dip)
            # 缝宽/谷深联合判据：真实柱间隙是"窄缝+强谷"（A67-4 缝宽
            # 6px 谷深>=15）。A66 假缝是"宽热区(37px)+弱谷(12.8)"，
            # 纹理空隙弥散不成锋利缝。缝越宽所需谷深越高，按宽度插值。
            seam_w = seam_hi - seam_lo + 1
            need_s = seam_need + max(0.0, seam_w - 4) * 0.25
            if cap_dip >= need_s and body_dip >= need_s:
                valid_cands.append((seam_lo, seam_hi))
        cands = valid_cands
        if not cands:
            out.append(col)
            continue
        # 目标子列宽：优先用非连体、非贴边候选的中位宽（真实单柱的
        # 宽度）。A66 [67,140,135] 里 67 是贴边残柱、140/135 是连体，
        # 若直接用全体中位宽 135 当 target，0.45*target≈61 的下限会
        # 拒绝 52~56px 的真实子列，导致拆分失败退回连体。
        body_ws = sorted(
            float(c["right"] - c["left"] + 1) for c in cols
            if c not in merged
            and int(c.get("left", 0)) > 1
            and int(c.get("right", w - 1)) < w - 2
        )
        if body_ws:
            target = max(body_ws[len(body_ws) // 2], 12.0)
        else:
            # 全是连体/贴边时按连体宽的一半估计（2 柱粘连最常见）。
            target = max(0.5 * med_w, 12.0)
        best = None
        n = len(cands)

        def _try(combo):
            cuts = [L - 1] + [int((s0 + s1) // 2) for s0, s1 in combo] + [R + 1]
            ws = [cuts[i + 1] - cuts[i] - 1 for i in range(len(cuts) - 1)]
            if min(ws) < max(8, target * 0.45):
                return None
            return float(np.std(ws))

        import itertools
        for k in (1, 2, 3):
            if k > n:
                continue
            for combo in itertools.combinations(cands, k):
                ok = True
                for i in range(1, len(combo)):
                    if combo[i][0] - combo[i - 1][1] < max(6, target * 0.4):
                        ok = False
                        break
                if not ok:
                    continue
                sc = _try(combo)
                if sc is not None and (best is None or sc < best[0]):
                    best = (sc, combo)
            if best:
                break
        if not best:
            out.append(col)
            continue
        seams = best[1]
        # 子列完整性守卫：缝必须把连体段分成"每一段都像柱"的组合。
        # A67-4 x[262~367] 的假缝 338~344（柱内暗纹理，谷深 23.8）会
        # 切出 26px 碎条；真缝组合（如 A67-4 x[81~247] 缝 162~167）的
        # 子列宽度都接近 target。若任一子列宽 < 0.55*target 且组合里
        # 存在更宽的子列（宽窄比 > 1.8），视为碎条缝，放弃该组合。
        def _combo_ok(combo_):
            cuts_ = [L - 1] + [int((s0 + s1) // 2) for s0, s1 in combo_] + [R + 1]
            ws_ = [cuts_[i + 1] - cuts_[i] - 1 for i in range(len(cuts_) - 1)]
            return min(ws_) * 1.8 >= max(ws_)
        if not _combo_ok(seams):
            out.append(col)
            continue
        # 所选缝的双区谷深（min 取帽/身较弱一侧），随子列传递给置信度
        # 仲裁：弱证据拆分子柱在竞争时打折（A66 防护，见 _detection_confidence）。
        child_dip = min(
            min(seam_evidence.get(s, (0.0, 0.0))) for s in seams)
        cuts = [L - 1] + [int((s0 + s1) // 2) for s0, s1 in seams] + [R + 1]
        subs = []
        for i in range(len(cuts) - 1):
            lo, hi = cuts[i] + 1, cuts[i + 1] - 1
            sub_rows = []
            for y, a, b in rows:
                a2, b2 = max(int(a), lo), min(int(b), hi)
                if a2 <= b2:
                    sub_rows.append((int(y), a2, b2))
            if len(sub_rows) >= max(8, int(len(rows) * 0.35)):
                subs.append(sub_rows)
        if len(subs) < 2:
            out.append(col)
            continue
        for sub_rows in subs:
            out.append({
                "left": int(min(a for _, a, _ in sub_rows)),
                "right": int(max(b for _, _, b in sub_rows)),
                "center_x": int(round(np.mean([(a + b) * 0.5 for _, a, b in sub_rows]))),
                "y_top": sub_rows[0][0], "y_bottom": sub_rows[-1][0],
                "width": float(np.median([b - a for _, a, b in sub_rows])),
                "ranges": sub_rows, "contour": build_contour(sub_rows),
                "y_proj": col.get("y_proj"),
                "split_child": True,
                "split_evidence": float(child_dip),
            })
    if len(out) != len(cols):
        for i, c in enumerate(out):
            c["label"] = i + 1
            c["color"] = PR_COLORS[i % len(PR_COLORS)]
        # 递归一次：拆分结果里可能仍有连体段（A66 的 135 = 2x65）。
        # 仅当新一轮最小宽明显大于本轮 base（说明本轮 base 只是残柱，
        # 宽度基准不可靠）时才递归，避免把单柱再拆碎（A67-4）。
        try:
            rec = _split_merged_geometric_columns(out, gray, filtered, strict=True)
            if len(rec) > len(out):
                return rec
        except Exception:
            pass
        return out
    return cols


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
                overlap = max(y_top, ht) <= min(y_bottom, hb)
                if ((y_bottom - y_top + 1 < max(MIN_COLUMN_HEIGHT, 12) and hb - ht >= MIN_COLUMN_HEIGHT - 1)
                        or not overlap):
                    # 候选主区异常浅或与原图 hint 完全错位：信任原图
                    # hint（含 pad 余量）。
                    y_top, y_bottom = max(0, ht - pad), min(h - 1, hb + pad)
                else:
                    # 正常时仅上扩顶部；底部保持候选自身结果，避免把
                    # 原图 hint 里的衬底颗粒区（A67-4/5）接进追踪带。
                    y_top = max(0, min(y_top, ht) - 2)
                    y_bottom = min(h - 1, max(y_bottom, y_top + max(MIN_COLUMN_HEIGHT, 12) - 1))
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


def _pitch_estimate(columns):
    """邻柱中心距中位估计（不足 3 根或异常时返回 None，调用方退回常数）。"""
    cs = sorted(0.5 * (c["left"] + c["right"]) for c in columns or [] if c.get("ranges"))
    if len(cs) < 3:
        return None
    gaps = [cs[i + 1] - cs[i] for i in range(len(cs) - 1) if cs[i + 1] - cs[i] > 1.0]
    if not gaps:
        return None
    return float(np.median(gaps))


def _pillar_evidence(columns, gray, sign, image_width=None):
    """逐柱对照同一行内外灰度，单根柱和 ROI 也执行极性检查。

    image_width: 传入时用于识别贴边裁切柱。贴边残柱的"外区"取样
    常落在图边界的暗角/亮带上（A67-4 右残柱 contrast 8 vs 阈值
    10.7），对比度天然偏低，按完整柱阈值会误杀；因此贴边柱的
    contrast 阈值放宽到 0.6 倍，support 下限也相应放宽。
    """
    accepted = []
    residual = gray.astype(np.float32) - cv2.medianBlur(gray, 5).astype(np.float32)
    noise_floor = float(np.median(np.abs(residual))) * 1.4826
    w_img = int(gray.shape[1]) if image_width is None else int(image_width)
    for col in columns or []:
        contrasts = []
        rows = col.get("ranges", [])
        # 浅柱（柱排带高相对柱距过小，如 A67-2 低对比密集柱排）：外区
        # 稍远就落进相邻柱/亮带（柱距 ~55px），对比被污染。外区收窄到
        # 6px、对比/support 阈值放宽（同贴边柱），避免整排被误杀。
        # 浅带阈值 26px 随柱距自适应：0.45×pitch（A67-2 pitch≈55→25），
        # 夹在 [18,26]——现有图柱距小则 26 支配，行为不变；大柱距图上
        # 阈值按比例放开。
        pitch_est = _pitch_estimate(columns)
        shallow_lim = 26 if pitch_est is None else max(18, min(26, 0.45 * pitch_est))
        shallow_band = (len(rows) >= 2
                        and (int(rows[-1][0]) - int(rows[0][0])) < shallow_lim)
        for y, left, right in rows[::max(1, len(rows) // 32)]:
            y, left, right = int(y), int(round(left)), int(round(right))
            width = right - left
            if width < 3:
                continue
            inset = max(1, width // 4)
            gap_lim = 6 if shallow_band else 12
            gap = max(2, min(gap_lim, width // 4))
            inside = gray[y, left + inset:right - inset + 1]
            outside = np.concatenate((gray[y, max(0, left - gap):max(0, left - 1)],
                                      gray[y, min(gray.shape[1], right + 2):right + gap + 1]))
            if inside.size and outside.size:
                contrasts.append(sign * (float(np.median(inside)) - float(np.median(outside))))
        contrast = float(np.median(contrasts)) if contrasts else 0.0
        support = float(np.mean(np.asarray(contrasts) > 1.0)) if contrasts else 0.0
        edge_clipped = (
            int(col.get("left", 0)) <= 1
            or int(col.get("right", w_img - 1)) >= w_img - 2
        )
        # 相反极性不能通过更高的柱数/宽度评分抵消。
        # 拆分子柱与母柱同源（几何追踪已验证柱体），且外区可能落在
        # 相邻子柱体内导致对比度失真，仅要求非反相。
        if col.get("split_child"):
            if contrast > -0.5 * max(1.5, 1.8 * noise_floor):
                item = dict(col)
                item["polarity_contrast"] = contrast
                item["polarity_support"] = support
                accepted.append(item)
            continue
        if edge_clipped:
            # 贴边裁切残柱：外侧背景不可靠，阈值放宽。
            contrast_need = 0.6 * max(1.5, 1.8 * noise_floor)
            support_need = 0.45
        elif shallow_band:
            # 浅柱排：外区易污染（A67-2），放宽同贴边柱。
            contrast_need = 0.6 * max(1.5, 1.8 * noise_floor)
            support_need = 0.5
        else:
            contrast_need = max(1.5, 1.8 * noise_floor)
            support_need = 0.65
        if contrast > contrast_need and support >= support_need:
            item = dict(col)
            item["polarity_contrast"] = contrast
            item["polarity_support"] = support
            accepted.append(item)
    return accepted


def _binary_seed_columns(mask, gray, roi_mode=False, y_hint=None):
    """binary 先确定柱身、顶部、底部；逐行轮廓只在邻柱间的单元内生长。"""
    h, w = mask.shape
    if h < MIN_COLUMN_HEIGHT or w < 5:
        return []
    top, bottom, _ = _main_row_activity(gray, prefer_upper=not roi_mode)
    if y_hint is not None:
        try:
            ht, hb = int(y_hint[0]), int(y_hint[1])
            pad = max(6, int(round(h * 0.05)))
            overlap = max(top, ht) <= min(bottom, hb)
            if ((bottom - top < MIN_COLUMN_HEIGHT - 1 and hb - ht >= MIN_COLUMN_HEIGHT - 1)
                    or not overlap):
                # filtered 主区异常浅（如 A55 只剩柱顶亮线），或与原图
                # hint 完全错位（A67-5 filtered 落在衬底颗粒上）：信任
                # 原图 hint。
                top, bottom = max(0, ht - pad), min(h - 1, hb + pad)
            else:
                # 正常时仅把顶部上扩到原图柱顶；底部保持自身结果，
                # 避免把原图 hint 里的衬底颗粒区（A67-4/5）接进来。
                top = max(0, min(top, ht) - 2)
                bottom = min(h - 1, max(bottom, top + MIN_COLUMN_HEIGHT - 1))
        except Exception:
            pass
    if bottom - top < MIN_COLUMN_HEIGHT - 1:
        return []
    trim = max(2, (bottom - top) // 6)
    band = mask[top + trim:max(top + trim + 1, bottom - trim + 1)]
    projection = smooth_1d(np.mean(band > 0, axis=0), 5)
    segments = _run_segments(projection >= 0.5, min_len=5)
    cols = []
    # 两阶段过滤：先收集全部宽度/纵横比达标的候选，再决定宽扁豁免。
    # 宽扁柱排（如 A58 66px 高 × 85px 宽、A689 38px 高 × 70px 宽）的
    # 纵横比天然低于 1.2；但只有当 ≥2 个非贴边候选宽度彼此一致
    # （说明是一排同类真实柱）才放宽，单根或宽度混杂（如 A66 的
    # 134px 粘连柱混在 68px 柱旁）仍按原阈值拒绝。
    def _aspect_ok(width_, rows_, edge_):
        if len(rows_) / max(width_, 1) >= MIN_ASPECT_RATIO:
            return True
        return (not edge_) and _flat_exempt and             len(rows_) / max(width_, 1) >= _flat_limit

    prelim = []
    _prelim_cells = []
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
            # 贴边 cell（cell_l==0 / cell_r==w-1）的整行守卫失效，
            # 补一条"run 几乎占满 cell"判定：衬底亮带会横贯整个贴边
            # cell（A66 PR1 的横线伪影、A67-5 的碎片段）。
            cell_w = cell_r - cell_l + 1
            if (a == cell_l and cell_l > 0) or (b == cell_r and cell_r < w - 1)                     or (cell_l <= 1 and a <= 1 and (b - a) >= cell_w * 0.9)                     or (cell_r >= w - 2 and b >= w - 2 and (b - a) >= cell_w * 0.9):
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
        if width < (3 if edge else MIN_AVG_WIDTH):
            continue
        prelim.append((width, rows, edge, center, left, right))
        _prelim_cells.append((cell_l, cell_r))

    _flat_exempt = False
    _flat_limit = MIN_ASPECT_RATIO
    # 宽扁豁免：宽度一致性检查覆盖全部宽度达标候选（含贴边），
    # 否则 A66 的 134/132px 粘连柱会因贴边邻柱不参与而误判为一致组。
    # 豁免下限只按非贴边成员的最低纵横比计算。
    width_set = [wd for wd, _, _, _, _, _ in prelim if wd >= MIN_AVG_WIDTH]
    body_items = [(wd, rr) for wd, rr, eg, _, _, _ in prelim if not eg]
    if len(body_items) >= 2 and len(width_set) >= 2:
        med_w = float(np.median(width_set))
        if all(abs(wd - med_w) <= max(6.0, med_w * 0.18) for wd in width_set):
            min_ar = min(len(rr) / max(wd, 1) for wd, rr in body_items)
            # 宽扁但仍在柱形范围（≥0.45）且高度不小于最低柱高：
            if MIN_ASPECT_RATIO > min_ar >= 0.45 and min(len(rr) for _, rr in body_items) >= MIN_COLUMN_HEIGHT:
                _flat_exempt = True
                _flat_limit = min_ar * 0.92

    for _pi, (width, rows, edge, center, left, right) in enumerate(prelim):
        if not _aspect_ok(width, rows, edge):
            continue
        # 侧壁行间中值平滑：衬底界面亮带会让个别行的 run 左边界抖到
        # cell 深处（A66 PR1 轮廓的横向线条伪影），中值滤波抑制单行
        # 抖动。
        if len(rows) >= 9:
            arr = np.asarray(rows, dtype=np.float32)
            k = 9
            pad = k // 2
            a_pad = np.pad(arr[:, 1], (pad, pad), mode="edge")
            b_pad = np.pad(arr[:, 2], (pad, pad), mode="edge")
            a_win = np.lib.stride_tricks.sliding_window_view(a_pad, k)
            b_win = np.lib.stride_tricks.sliding_window_view(b_pad, k)
            arr[:, 1] = np.median(a_win, axis=1)
            arr[:, 2] = np.median(b_win, axis=1)
            rows = [(int(y), int(a), int(b)) for y, a, b in arr]
        cell_l, cell_r = _prelim_cells[_pi]
        col = {"ranges": rows, "left": min(a for _, a, _ in rows),
               "right": max(b for _, _, b in rows), "center_x": int(round(center)),
               "y_top": rows[0][0], "y_bottom": rows[-1][0], "width": width,
               "contour": build_contour(rows), "cell": (cell_l, cell_r)}
        cols.append(col)
    return cols


def _constrained_edge_path(response, seed, radius, lower=0, upper=None, pull=0.0):
    """窄带动态规划：梯度证据 + binary 距离 + 行间连续性，抛物线亚像素定位。

    pull>0 时对向上偏移（offsets<0，top 搜索中行号减小方向）追加定向
    代价 pull*|offset|/radius：柱顶上方局部横向特征（噪声带/亮线）要
    把整条路径集体上跳，需要比真实顶边更强的梯度证据。默认 0 保持
    原行为逐位不变。
    """
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
    if pull > 0:
        unary = unary + pull * np.maximum(0.0, -offsets) / max(radius, 1)
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


def _wall_path_curved(response, seed, radius, lower=0, upper=None, kappa=0.03):
    """pair-state Viterbi：一阶转移上追加曲率正则 κ·(x_t-2x_{t-1}+x_{t-2})²。

    圆帽拐角/锥形过渡处一阶 DP 被"柱身延伸"unary 拽直（A63 帽沿）；
    二阶项让逐行渐转比突然转折更省，配合 _arbitrate_wall 逐行仲裁
    采纳（证据更强且粗糙度不增才换）。unary/亚像素/valid 语义与
    _constrained_edge_path 完全一致；短路径退回一阶实现。
    """
    response = np.asarray(response, dtype=np.float32)
    n = response.shape[0]
    if n < 3 or kappa <= 0:
        return _constrained_edge_path(response, seed, radius, lower, upper)
    seed = np.asarray(seed, dtype=np.float32)
    width = response.shape[1]
    lo = np.broadcast_to(np.maximum(0, lower), (n,)).astype(float)
    hi = np.broadcast_to(np.minimum(width - 1, width - 1 if upper is None else upper), (n,)).astype(float)
    seed = np.clip(seed, lo, hi)
    offsets = np.arange(-radius, radius + 1)
    xs = np.clip(np.rint(seed).astype(int)[:, None] + offsets,
                 np.ceil(lo).astype(int)[:, None], np.floor(hi).astype(int)[:, None])
    values = np.maximum(response[np.arange(n)[:, None], xs], 0)
    scale = max(1.0, float(np.percentile(values.max(axis=1), 75)))
    unary = -np.clip(values / scale, 0, 3) + 0.15 * (offsets / max(radius, 1)) ** 2
    ii = offsets[:, None, None]
    jj = offsets[None, :, None]
    kk = offsets[None, None, :]
    T1 = 0.07 * np.minimum((jj - ii) ** 2, 36) + 0.10 * np.abs(jj - ii)   # (cur,prev)
    T1 = T1[:, :, 0]                                                       # 显式 2D (cur,prev)
    K2 = kappa * (ii - 2 * jj + kk) ** 2                                   # (cur,prev,prev2)
    # D(i,j)：cur=i、prev=j 的最优代价；行 1 全对初始化（行 0 无二阶约束）。
    D = unary[0][None, :] + unary[1][:, None] + T1
    bps = np.zeros((n, len(offsets), len(offsets)), dtype=np.int8)          # 每行 (j,i)->k
    for t in range(2, n):
        # E[j,k,i] = D_prev(j,k) + K2(i,j,k)：D_prev[j,k] 即 D[j,k]
        #（t-1 行 cur=j、prev=k）；对 prev2=k 取 argmin。
        E = D[:, :, None] + K2.transpose(1, 2, 0)                          # (j,k,i)
        bk = E.argmin(axis=1)                                              # (j,i)
        G = np.take_along_axis(E, bk[:, None, :], axis=1)[:, 0, :]         # (j,i)
        D = unary[t][:, None] + T1 + G.T                                   # (i,j)
        bps[t] = bk
    # 回溯：末端两层 argmin，再沿 bps 逆推前两行状态；亚像素/valid 与
    # 一阶实现同式（抛物线内插 + 强度门控）。
    j_end, i_end = np.unravel_index(int(np.argmin(D)), D.shape)
    states = np.zeros(n, dtype=int)
    states[n - 1] = int(i_end)
    states[n - 2] = int(j_end)
    for t in range(n - 1, 1, -1):
        states[t - 2] = int(bps[t, states[t - 1], states[t]])
    path = np.zeros(n, dtype=float)
    strength = np.zeros(n, dtype=float)
    for y in range(n):
        x = int(xs[y, states[y]])
        strength[y] = max(0.0, float(response[y, x]))
        dx = 0.0
        if 0 < x < width - 1 and strength[y] > 0:
            a, b, c = map(float, response[y, x - 1:x + 2])
            den = a - 2 * b + c
            if den < -1e-6 and b >= a and b >= c:
                dx = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
        path[y] = np.clip(x + dx, max(lo[y], seed[y] - radius), min(hi[y], seed[y] + radius))
    valid = strength > max(0.8, 0.10 * scale)
    path[~valid] = seed[~valid]
    return path, strength, valid


def _arbitrate_wall(old_path, old_st, new_path, new_st, eps=0.05, min_run=3, max_ratio=0.6):
    """逐行仲裁：新路证据更强且粗糙度不增的行才采纳，孤立段丢弃。

    返回 (path, strength, valid, adopted_ratio)。9 图侧壁证据强且已平滑，
    adopt 集为空 → 输出与旧路径逐元素相同（bit-exact 保证）。
    """
    n = len(old_path)
    if n < 5 or len(new_path) != n:
        return old_path, old_st, None, 0.0
    d2o = np.abs(np.diff(old_path, 2, prepend=[old_path[0], old_path[0]]))
    d2n = np.abs(np.diff(new_path, 2, prepend=[new_path[0], new_path[0]]))
    adopt = (new_st - old_st > eps) & (d2n <= d2o + 1e-9)
    # 连续段 <min_run 的孤立采纳视为噪声。
    i = 0
    while i < n:
        if adopt[i]:
            j = i
            while j + 1 < n and adopt[j + 1]:
                j += 1
            if j - i + 1 < min_run:
                adopt[i:j + 1] = False
            i = j + 1
        else:
            i += 1
    ratio = float(np.mean(adopt))
    if ratio == 0.0:
        return old_path, old_st, None, 0.0
    if ratio > max_ratio:
        return new_path, new_st, None, ratio
    out_p, out_s = np.array(old_path, dtype=float), np.array(old_st, dtype=float)
    out_p[adopt] = new_path[adopt]
    out_s[adopt] = new_st[adopt]
    return out_p, out_s, adopt, ratio


def _rebuild_cap_profile(rows, gray, sign, image_width):
    """帽区弧形顶重建：按行内灰度半高交叉收缩顶部轮廓宽度。

    DP 侧壁在柱顶被柱身侧壁证据拽成直壁，圆帽柱（A63）顶部行宽
    明显大于真实帽形。对帽区逐行取行内灰度剖面：峰为行内最大平
    滑灰度，半高基准用沟底灰度（柱带外暗区中位）。收缩只在行宽
    仍显著小于柱身中位宽的行生效（帽区），且不把行宽收缩到低于
    0.3*中位宽（防噪声）。
    """
    if len(rows) < 8:
        return rows
    h, w = gray.shape[:2]
    widths = np.asarray([b - a for _, a, b in rows], dtype=float)
    med_w = float(np.median(widths))
    if med_w < 8:
        return rows
    # 帽区终点：从顶往下第一处宽度 >= 0.92*med 的行；跳过孤立突增行
    # （横线伪影：宽度骤增后骤减，reach 若落在其上会提前终结帽区收缩）。
    reach = None
    for i, wd in enumerate(widths):
        if wd < 0.92 * med_w:
            continue
        nxt = widths[i + 1] if i + 1 < len(widths) else wd
        prv = widths[i - 1] if i > 0 else wd
        if wd - nxt > 0.12 * wd or wd - prv > 0.25 * med_w:
            continue
        reach = i
        break
    if not reach:
        return rows
    # 沟底基准：柱带外左右两侧的暗区中位（找不到时用全图 10 分位）。
    y0, y1 = rows[0][0], rows[-1][0]
    left_ref = gray[y0:y1 + 1, :max(2, int(rows[0][1]) - 4)]
    right_ref = gray[y0:y1 + 1, min(w - 2, int(rows[-1][2]) + 5):]
    refs = [float(np.median(r)) for r in (left_ref, right_ref) if r.size]
    trench = min(refs) if refs else float(np.percentile(gray, 10))
    if not np.isfinite(trench):
        return rows
    out = []
    for i, (y, a, b) in enumerate(rows):
        if i > reach or (b - a) >= 0.92 * med_w:
            out.append((int(y), int(a), int(b)))
            continue
        lo = int(np.clip(a - 6, 0, w - 1))
        hi = int(np.clip(b + 7, 1, w))
        seg = np.convolve(
            gray[y, lo:hi].astype(np.float32), np.ones(3) / 3.0, mode="same")
        peak = float(np.max(seg))
        if peak <= trench + 8:
            out.append((int(y), int(a), int(b)))
            continue
        half = trench + (peak - trench) * 0.5
        above = np.where(seg > half)[0]
        if len(above) < 2:
            out.append((int(y), int(a), int(b)))
            continue
        na = lo + int(above[0])
        nb = lo + int(above[-1])
        # 半高收缩下限：不低于 0.3*med 且不比原行宽得离谱。
        if (nb - na) < max(4.0, 0.3 * med_w) or nb - na > (b - a):
            out.append((int(y), int(a), int(b)))
            continue
        out.append((int(y), na, nb))
    return out


def _horizontal_artifact_limit(gray, columns, sign, max_extra=6):
    """检测柱顶上方的水平亮带伪影，返回顶部搜索允许的最高 y（含少量余量）。

    SEM 图常在柱顶上方出现贯穿视野的曝光亮线（A63/A68D）：binary 会把
    亮线连进柱体，顶部 DP 被拽到亮线处产生假顶。亮线特征是同一行在
    多根柱之间的沟道位置也保持高灰度（真实柱顶上方的沟道是暗的）。
    若不存在这种"跨柱亮带"，返回 0 表示不限制。
    """
    if gray is None or not columns:
        return 0
    g = np.asarray(gray, dtype=np.float32)
    h, w = g.shape[:2]
    if h < 16 or w < 32:
        return 0
    # 沟道参考行带：取柱体 y 范围的中部，宽度取柱间沟道。
    tops = [int(c.get("y_top", 0)) for c in columns]
    bottoms = [int(c.get("y_bottom", 0)) for c in columns]
    y_body0, y_body1 = int(np.median(tops)), int(np.median(bottoms))
    if y_body1 - y_body0 < 8:
        return 0
    channel_rows = g[y_body0 + (y_body1 - y_body0) // 3:
                     y_body0 + 2 * (y_body1 - y_body0) // 3]
    channel_dark = float(np.percentile(channel_rows, 25))
    # 亮带候选：柱顶上方到 min(top)-? 的行；亮带行沟道灰度显著高于沟底。
    scan_hi = min(min(tops) for _ in columns)
    scan_lo = max(0, scan_hi - 40)
    if scan_hi <= scan_lo + 3:
        return 0
    bright_need = channel_dark + max(8.0, 0.35 * (float(np.percentile(g, 99)) - channel_dark))
    artifact_rows = []
    for y in range(scan_lo, scan_hi + max_extra):
        row = g[y]
        # 只统计柱体 x 范围之外的沟道像素，避免柱体自身高亮干扰。
        mask = np.ones(w, dtype=bool)
        for c in columns:
            l = max(0, int(c.get("left", 0)) - 2)
            r = min(w, int(c.get("right", w - 1)) + 3)
            mask[l:r] = False
        if mask.sum() < w * 0.08:
            continue
        if float(np.median(row[mask])) >= bright_need:
            artifact_rows.append(y)
    if not artifact_rows:
        return 0
    # 亮带必须连续成段（>=3 行）且贴近柱顶才算伪影；返回亮带底部+余量。
    runs = _run_segments(np.isin(np.arange(h), artifact_rows), min_len=3)
    if not runs:
        return 0
    bottom = max(r[1] for r in runs)
    top_min = min(tops)
    if bottom <= top_min + 2:
        return bottom + max_extra
    return 0


def _shrink_len(widths, med):
    """宽顶渐降段长度：从首行起连续不增的行数（2~6）。"""
    n = 1
    while n < 6 and n + 1 < len(widths) and widths[n + 1] <= widths[n] + 0.5:
        n += 1
    return max(2, min(6, n))


def med_w_ok(widths, med):
    """宽顶渐降判据总门。"""
    if widths[0] < 0.72 * med or widths[0] >= med:
        return False
    drop = widths[0] - min(widths[1:7]) if len(widths) > 6 else widths[0] - widths[-1]
    return drop >= 0.12 * med


def _trim_head_artifact_rows(rows, gray):
    """修复帽顶前几行被水平亮带拉宽的轮廓行（A63/A68D 头部横线）。

    亮带行在帽顶上方或帽顶处让 DP 侧壁外扩，形成"首行宽、次行骤窄"
    的横向线条。判据（前 5 行内）：
    1. 行宽 > 参考宽 1.35 倍且差额 >= 4px（单行突跳，原有判据）；
    2. 行宽比后续 3 行最小宽高出 max(4.0, 0.25*ref)（渐进展宽，横线
       被部分吸收时逐行衰减的形态）。
    修复动作：默认把命中行的边界收缩到参考宽度；若该行两侧沟道灰度
    也接近行内峰值（横带特征：真实柱顶中间亮两侧暗，横带整行皆亮），
    则直接删除该行（收缩会留下平顶台阶），最多删 2 行且删后行数 >= 8。
    """
    if len(rows) < 6:
        return rows
    widths = np.asarray([b - a for _, a, b in rows], dtype=float)
    # 后续正常宽度：帽顶向下首个"平台"行宽（宽度首次稳定：与下一行差
    # <=2px），而非固定第 4~10 行中位——帽形柱（A63）前 10 行宽度仍在
    # 爬升，取其中位会把伪影行误当成参考。
    ref = 0.0
    # 平台行宽必须显著窄于首行（<0.85*首行宽），否则前几行整体都是
    # 横线平台（PEB105C：前 4 行 46px，柱身 20px），平台搜索会把伪影
    # 平台当成参考导致不触发。
    w_first = float(widths[0])
    for i in range(1, min(14, len(widths) - 1)):
        if widths[i] >= 0.85 * w_first:
            continue
        if abs(widths[i + 1] - widths[i]) <= 2.0 and widths[i] >= 6:
            ref = float(min(widths[i], widths[i + 1]))
            break
    if ref < 6:
        deeper = widths[6:14] if len(widths) >= 12 else widths[len(widths)//2:]
        cand = float(np.median(deeper)) if len(deeper) else 0.0
        if cand < 0.9 * w_first:
            ref = cand
    if ref < 6:
        return rows
    # v6.2.1 P1-B2：宽顶渐降（横线帽残留，PEB110C_-0.06um PR5）。
    # 首行宽 >=0.72*med 且 <med，其后数行单调不增且总降幅 >=0.12*med：
    # 顶部吸在亮帽上沿斜切下来；真实 T-top 帽沿宽 > med 不触发，窄顶不触发。
    if ENABLE_WIDE_TOP_SHRINK:
        w_med_all = float(np.median(widths))
        if len(widths) >= 8 and med_w_ok(widths, w_med_all):
            tail_w = widths[_shrink_len(widths, w_med_all)]
            for k in range(_shrink_len(widths, w_med_all)):
                y_, a_, b_ = rows[k]
                na_ = a_ + (widths[k] - tail_w) * 0.5
                nb_ = b_ - (widths[k] - tail_w) * 0.5
                if nb_ - na_ >= max(4.0, tail_w * 0.5):
                    rows[k] = (int(y_), int(round(na_)), int(round(nb_)))
    h, w = gray.shape[:2]
    trimmed = 0
    removed = 0
    out = list(rows)
    scan = min(HEAD_TRIM_SCAN_ROWS, len(rows) - 1)
    i = 0
    while i < scan:
        wi = out[i][2] - out[i][1]
        nxt = [out[j][2] - out[j][1] for j in range(i + 1, min(i + 4, len(out)))]
        narrow_next = min(nxt) if nxt else ref
        spike = wi > ref * 1.35 and wi - ref >= 4.0
        gradual = wi - narrow_next >= max(4.0, 0.25 * ref) and wi > ref
        if not (spike or gradual):
            i += 1
            continue
        y, a, b = out[i]
        # 横带判定：行 x 范围外扩 3px 处的沟道灰度接近行内峰灰度。
        band_row = False
        lo = int(np.clip(a - 3, 0, w - 1))
        hi = int(np.clip(b + 4, 1, w))
        seg = gray[y, lo:hi].astype(np.float32)
        if seg.size:
            peak = float(np.max(seg))
            left_g = float(np.median(gray[y, max(0, int(a) - 6):max(1, int(a) - 1)]))                 if int(a) - 1 > 0 else peak
            right_g = float(np.median(gray[y, min(w - 1, int(b) + 2):min(w, int(b) + 7)]))                 if int(b) + 2 < w else peak
            band_row = peak > 0 and min(left_g, right_g) >= 0.6 * peak
        if band_row and removed < 2 and len(out) - 1 >= 8:
            del out[i]
            removed += 1
            scan = min(scan, len(out) - 1)
            continue
        # v6.2.1 P1-A：按左右边缘与下一行的差分决定收缩侧，避免对单侧
        # 外扩行做对称误切（缺口成因之一）。
        ny, na_, nb_ = out[min(i + 1, len(out) - 1)]
        dl = a - na_
        dr = nb_ - b
        if ENABLE_ASYMMETRIC_HEAD_TRIM and dl > dr + 2.0:
            na, nb = a + (wi - ref), b
        elif ENABLE_ASYMMETRIC_HEAD_TRIM and dr > dl + 2.0:
            na, nb = a, b - (wi - ref)
        else:
            cut = (wi - ref) * 0.5
            na, nb = a + cut, b - cut
        if nb - na >= max(4.0, ref * 0.5):
            out[i] = (int(y), int(round(na)), int(round(nb)))
            trimmed += 1
        i += 1
    return out if (trimmed or removed) else rows


def _smooth_top_notches(rows, med_w):
    """回填柱顶 V 形凹陷行（头部缺口修复，v6.2.1 P1-A）。

    顶部各行左右边缘由侧壁 DP 独立估计，个别行会比两侧邻行都窄，
    经 build_contour 连线后形成 V 形缺口。真实圆顶单调变宽、T-top
    平台后单调变窄，"比两侧都窄"的凹陷在两种真实形态中都不物理。
    仅对这类 V 形行做左右边缘线性插值回填（保持窄顶形态，不补宽）。
    """
    if not ENABLE_TOP_V_NOTCH_FIX or len(rows) < 6 or med_w < 8:
        return rows
    scan = min(10, max(4, len(rows) // 5))
    need = max(2.0, 0.06 * med_w)
    out = [tuple(r) for r in rows]
    fixed = 0
    i = 1
    while i < scan and i + 1 < len(out):
        w0 = out[i - 1][2] - out[i - 1][1]
        w1 = out[i][2] - out[i][1]
        w2 = out[i + 1][2] - out[i + 1][1]
        if w1 >= min(w0, w2) - need or fixed >= 4:
            i += 1
            continue
        y, a, b = out[i]
        _, pa, pb = out[i - 1]
        _, na, nb = out[i + 1]
        frac = (y - out[i - 1][0]) / max(1, out[i + 1][0] - out[i - 1][0])
        ja = pa + (na - pa) * frac
        jb = pb + (nb - pb) * frac
        # 插值后不得比两侧邻行更宽（保持窄顶，只去凹陷）
        if jb - ja > max(w0, w2):
            i += 1
            continue
        out[i] = (int(y), int(round(ja)), int(round(jb)))
        fixed += 1
        i += 1
    return out if fixed else rows


def _adaptive_scales(cols, h, w):
    """跨柱尺寸基准（通用化：不同放大倍数/分辨率图的相对量参考）。

    med_w：柱宽中位；pitch：邻柱中心距中位；med_h：柱高中位。
    各调用方以 max(旧绝对值, k×基准) 形式消费——现有小图上绝对项
    支配（k×基准 ≤ 旧值），行为逐位不变；大尺寸图上比例项接管。
    """
    ws = [float(c["right"] - c["left"] + 1) for c in cols if c.get("ranges")]
    cs = sorted(0.5 * (c["left"] + c["right"]) for c in cols if c.get("ranges"))
    hs = [float(c["y_bottom"] - c["y_top"] + 1) for c in cols if c.get("ranges")]
    med_w = ws[len(ws) // 2] if ws else 0.0
    pitch = (cs[len(cs) // 2] - cs[0]) / max(1, len(cs) - 1) if len(cs) >= 2 else med_w
    med_h = hs[len(hs) // 2] if hs else 0.0
    return {"med_w": med_w, "pitch": max(pitch, med_w), "med_h": med_h, "h": h, "w": w}


def _dedupe_refined_columns(cols):
    """精修后去重：同一根柱的贴边残柱+窄条碎片（A671318-01M-4）。

    真实密排柱相邻 x 区间几乎不重叠；横向重叠 >=0.60*min(宽) 时判
    同一根柱，删窄者，等宽时删行数少者。须在两端排除之前调用。
    """
    if len(cols) < 2:
        return cols
    out = []
    for c in sorted(cols, key=lambda c: c["center_x"]):
        if out:
            m = out[-1]
            ov = min(m["right"], c["right"]) - max(m["left"], c["left"]) + 1
            w1 = m["right"] - m["left"] + 1
            w2 = c["right"] - c["left"] + 1
            if ov > 0 and ov >= 0.60 * min(w1, w2):
                if w2 < w1 or (w2 == w1 and
                               (c["y_bottom"] - c["y_top"]) <= (m["y_bottom"] - m["y_top"])):
                    continue
                out[-1] = c
                continue
        out.append(c)
    return out


def _merge_overlapping_columns(cols, gray):
    """合并"过分裂"的同一根柱（A671318-01M-4：PR1+PR2、PR4+PR5）。

    _dedupe_refined_columns 的 0.60×min宽 重叠门只够抓碎片；本函数处理
    轻微搭接（重叠 >=5%×min宽）的同柱对：y 顶底对齐（差<=12px）、
    重叠带内灰度无真实柱缝谷（谷深 <12，真缝 >=15）、合并后不超
    2×中位宽。宁漏合不误合：任一门不过即保持两列。
    """
    if len(cols) < 2 or gray is None:
        return cols
    ws_ = sorted(float(c["right"] - c["left"] + 1) for c in cols)
    med_w = ws_[len(ws_) // 2]
    g = np.asarray(gray, dtype=np.float32)

    def _seam_dip(m, c):
        # 重叠带 x 范围在柱身行带（合并柱 30%~70% 高度）的灰度列均值谷深。
        ovl = max(int(m["left"]), int(c["left"]))
        ovr = min(int(m["right"]), int(c["right"]))
        yt = min(int(m["y_top"]), int(c["y_top"]))
        yb = max(int(m["y_bottom"]), int(c["y_bottom"]))
        band = g[max(0, yt + int((yb - yt + 1) * 0.30)):max(1, yt + int((yb - yt + 1) * 0.70)) + 1, :]
        if band.shape[0] < 3 or ovr <= ovl:
            return 0.0
        prof = np.convolve(band.astype(np.float32).mean(axis=0),
                           np.ones(5) / 5.0, mode="same")
        off = max(3, int((ovr - ovl + 1) * 0.5))
        lo_a, hi_b = max(0, ovl - off), min(len(prof) - 1, ovr + off)
        peak = max(float(np.max(prof[lo_a:ovl + 1])) if ovl > lo_a else 0.0,
                   float(np.max(prof[ovr:hi_b + 1])) if hi_b > ovr else 0.0)
        dip = peak - float(np.min(prof[ovl:ovr + 1]))
        return dip if dip > 0 else 0.0

    out = []
    for c in sorted(cols, key=lambda c: c["center_x"]):
        if out:
            m = out[-1]
            ov = min(m["right"], c["right"]) - max(m["left"], c["left"]) + 1
            w1 = m["right"] - m["left"] + 1
            w2 = c["right"] - c["left"] + 1
            if (ov > 0 and ov >= 0.05 * min(w1, w2)
                    and abs(int(m["y_top"]) - int(c["y_top"])) <= 12
                    and abs(int(m["y_bottom"]) - int(c["y_bottom"])) <= 12
                    and (max(m["right"], c["right"]) - min(m["left"], c["left"]) + 1) <= 2.0 * med_w
                    and _seam_dip(m, c) < 12.0):
                rows = {}
                for y, s, e in list(m["ranges"]) + list(c["ranges"]):
                    q = rows.get(int(y))
                    rows[int(y)] = (min(q[0], s), max(q[1], e)) if q else (s, e)
                nr = [(y, rows[y][0], rows[y][1]) for y in sorted(rows)]
                m["ranges"] = nr
                m["left"] = min(m["left"], c["left"])
                m["right"] = max(m["right"], c["right"])
                m["y_top"] = min(m["y_top"], c["y_top"])
                m["y_bottom"] = max(m["y_bottom"], c["y_bottom"])
                m["center_x"] = 0.5 * (m["left"] + m["right"])
                m["contour"] = build_contour(nr)
                continue
        out.append(c)
    return out


def _repair_foot_width_drop(rows, gray):
    """修复底部宽度骤降缺口（A641572 PR5/PR6 底部缺角）。

    柱脚处 DP 侧壁被衬底暗斑拽进柱内，形成"上一行宽、下一行骤窄"
    的阶梯缺口。真实锥形/缩脚是渐变的；判据：骤降 >0.38×前宽且
    >12px、位于底部 25% 高度内，且被丢弃段与保留段灰度差 <12
    （无真实边界证据）→ 用上方平台 (l,r) 延展到柱底。
    """
    if len(rows) < 12 or gray is None:
        return rows
    g = np.asarray(gray, dtype=np.float32)
    widths = np.asarray([b - a for _, a, b in rows], dtype=float)
    drop_i = None
    for j in range(len(rows) - 1):
        drop = widths[j] - widths[j + 1]
        # 骤降像素下限随柱宽自适应（max 保证现有小图上 12.0 支配不变）。
        drop_floor = max(12.0, 0.15 * widths[j])
        if drop > 0.38 * widths[j] and drop > drop_floor                 and j >= int(len(rows) * 0.75):
            drop_i = j
            break
    if drop_i is None:
        return rows
    y = int(rows[drop_i + 1][0])
    wide_l, wide_r = int(rows[drop_i][1]), int(rows[drop_i][2])
    nar_l, nar_r = int(rows[drop_i + 1][1]), int(rows[drop_i + 1][2])
    kept = float(np.median(g[y, nar_l + 2:max(nar_l + 3, nar_r - 1)]))         if nar_r - nar_l > 5 else float(np.median(g[y, nar_l:nar_r + 1]))
    ok = True
    if nar_l - 2 > wide_l:
        dl = float(np.median(g[y, wide_l:nar_l - 1]))
        if abs(dl - kept) >= FOOT_DROP_GRAY_GATE:
            ok = False
    if wide_r > nar_r + 2:
        dr = float(np.median(g[y, nar_r + 1:wide_r]))
        if abs(dr - kept) >= FOOT_DROP_GRAY_GATE:
            ok = False
    if not ok:
        return rows
    out = list(rows[:drop_i + 1])
    for k in range(drop_i + 1, len(rows)):
        out.append((int(rows[k][0]), wide_l, wide_r))
    return out


def _repair_foot_ramp_collapse(rows, gray):
    """修复底部"缓坡+骤降"型缺口（A641572-01C-3 右壁底部缺小口）。

    底部若干行每行仅收窄 2~12px（低于 _repair_foot_width_drop 的
    骤降门），累计却塌掉 10px+，DP 把真实斜坡内壁误当边界。判据：
    自底向上找最末连续缓窄段，若 (1) 起点平台宽度显著大于底行宽、
    (2) 平台宽接近柱身中位宽、(3) 被丢弃区与柱内保留段灰度差中位
    <10（无真实边界证据），则用起点平台 (l,r) 延展到柱底。
    """
    if len(rows) < 12 or gray is None:
        return rows
    g = np.asarray(gray, dtype=np.float32)
    widths = np.asarray([b - a for _, a, b in rows], dtype=float)
    n = len(rows)
    # 只搜底部 1/8 窗口：缺口是柱脚局部现象，窗口过大时会把正常
    # 锥脚渐变段并入缓窄段，导致 collapse 门失真。
    lo = max(int(n * 0.75), n - 1 - n // 8)
    # 自底向上找最末"缓窄段"起点：允许 <=2px 回弹（底部平台宽度
    # 常有 1~2px 抖动，严格单调会在起点前提前停住）。
    k = n - 1
    while k - 1 >= lo and -2.0 <= widths[k - 1] - widths[k] <= 12.0:
        k -= 1
    if k >= n - 1 or k < lo:
        return rows
    if widths[k] - widths[n - 1] < max(8.0, 0.12 * widths[k]):
        return rows
    if widths[k] < 0.6 * float(np.median(widths)):
        return rows
    diffs = []
    for tt in range(k + 1, n):
        y = int(rows[tt][0])
        wl, wr = int(rows[tt][1]), int(rows[tt][2])
        pl, pr = int(rows[k][1]), int(rows[k][2])
        kept = float(np.median(g[y, pl + 2:max(pl + 3, pr - 1)]))
        # 收窄方向（行窄于平台）：被丢弃区是 [wr, pr)——DP 把这段
        # 柱体/裙边拽出轮廓；延展方向（行宽于平台）：被丢弃区是
        # (pr, wr]——平台比真实壁窄。两个方向都要采灰度证据。
        if wr < pr - 2:
            diffs.append(float(np.median(g[y, max(wr + 1, pr - 12):pr])) - kept)
        if pl - 2 > wl:
            diffs.append(float(np.median(g[y, wl:max(wl + 1, pl - 1)])) - kept)
        if wr > pr + 2:
            diffs.append(float(np.median(g[y, pr + 1:wr])) - kept)
    # 真实锥脚：被丢弃区是衬底，与柱身差通常 >=14；缺口（DP 把柱脚
    # 裙边/柱体拽出轮廓）：丢弃区是柱体材料或亮裙边，与柱身差 <14
    # （A641 pr1 裙边比柱身亮 ~12）。暗于柱身 >=10 判真实沟道拒绝。
    if diffs:
        med = float(np.median(diffs))
        if med <= -FOOT_DROP_GRAY_GATE or med >= 14.0:
            return rows
    plat = rows[max(0, k - 2):k + 1]
    pl = int(np.median([r[1] for r in plat]))
    pr = int(np.median([r[2] for r in plat]))
    return rows[:k + 1] + [(int(y), pl, pr) for y, _, _ in rows[k + 1:]]


def _extend_rows_checked(rows, gray, new_foot_y):
    """foot 延展证据校验（P1-C 防拖尾，PEB110C_BF PR2）。

    底部对齐向下延展时：(1) 延展行宽取柱脚行宽 rows[-1]，不用柱身
    中部平台宽 rows[len//2]——平台宽大于柱脚时矩形外扩会造出拖尾；
    (2) 逐行灰度证据校验，命中任一即停：
        a. 延展行柱内灰度中位 vs 柱身下部基线差 >12（已非柱体）；
        b. 与衬底灰度（柱脚下方 3~10px）差 <8（已是衬底）。
    返回校验后的 rows（可能短于 new_foot_y）；一行都未延展返回 None。
    """
    if gray is None or len(rows) < 4:
        return None
    g = np.asarray(gray, dtype=np.float32)
    h = g.shape[0]
    n = len(rows)
    seg = rows[max(0, n - max(3, n // 4)):]
    vals = []
    for y, a, b in seg:
        if b - a > 3:
            vals.append(float(np.median(g[int(y), int(a) + 2:int(b) - 1])))
    if not vals:
        return None
    base = float(np.median(vals))
    fy = int(rows[-1][0])
    fl, fr = int(rows[-1][1]), int(rows[-1][2])
    if fr - fl < 4:
        return None
    sub_vals = []
    for dy in range(3, 11):
        yy = fy + dy
        if 0 <= yy < h:
            sub_vals.append(float(np.median(g[yy, fl + 1:fr])))
    sub = float(np.median(sub_vals)) if sub_vals else None
    out = list(rows)
    for y in range(fy + 1, int(new_foot_y) + 1):
        if y >= h:
            break
        m = float(np.median(g[y, fl + 1:fr]))
        if abs(m - base) > 25.0:
            break
        if sub is not None and abs(m - sub) < 8.0:
            break
        out.append((int(y), fl, fr))
    if len(out) == len(rows):
        return None
    return out


def _fill_row_gaps(cols, max_gap_ratio=0.20, min_gap=4):
    """断行填补（v7.1.0）：柱体纵向应连续，追踪中途丢失的行按邻行线性插值补齐。

    几何行追踪在柱身对比度局部降低时会丢行，ranges 出现 y 跳跃，表现为
    "柱子中间断开"（A671318-01M-4）。此处按跳跃跨度做线性插值补齐；
    缺口超过 max(4, 20%×柱高) 视为真实结构中断，不补。
    """
    for c in cols:
        rows = c.get("ranges")
        if not rows or len(rows) < 4:
            continue
        try:
            arr = [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
        except Exception:
            continue
        height = arr[-1][0] - arr[0][0] + 1
        limit = max(int(min_gap), int(round(max_gap_ratio * height)))
        out = [arr[0]]
        filled = 0
        for i in range(len(arr) - 1):
            y0, l0, r0 = arr[i]
            y1, l1, r1 = arr[i + 1]
            step = int(round(y1)) - int(round(y0))
            if 1 < step <= limit:
                for k in range(1, step):
                    t = float(k) / float(step)
                    out.append((y0 + k, l0 + (l1 - l0) * t, r0 + (r1 - r0) * t))
                filled += step - 1
            out.append(arr[i + 1])
        if filled:
            c["ranges"] = out
            c["contour"] = build_contour(out)
            c["gap_filled"] = filled
    return cols


def _merge_fragment_columns(cols, gray=None, width_ratio=0.45, max_gap=3):
    """碎片柱合并（v7.1.0）。

    柱体对比度局部降低时追踪会把一根柱切成若干窄条（A671318-01M-4 出现
    2~4px 宽的碎片，表现为"柱子中间断开"）。宽度 < ratio×柱群中位宽、且与
    邻柱几乎无间隙的窄条并入最近的邻柱：x 取并集、y 取包络、逐行取外沿。
    """
    if len(cols) < 2:
        return cols
    widths = sorted(float(c["right"] - c["left"] + 1) for c in cols)
    med_w = widths[len(widths) // 2]
    if med_w <= 0:
        return cols

    def _rows_map(c):
        return {int(r[0]): (float(r[1]), float(r[2])) for r in c.get("ranges", [])}

    changed = True
    guard = 0
    while changed and guard < 4:
        changed, guard = False, guard + 1
        cols = sorted(cols, key=lambda c: c["center_x"])
        out = []
        for c in cols:
            w = float(c["right"] - c["left"] + 1)
            if not out or w >= width_ratio * med_w:
                out.append(c)
                continue
            prev = out[-1]
            gap = float(c["left"]) - float(prev["right"])
            if gap > max_gap:
                out.append(c)
                continue
            # 合并：y 包络 + 逐行取外沿
            m1, m2 = _rows_map(prev), _rows_map(c)
            ys = sorted(set(m1) | set(m2))
            rows = []
            for y in ys:
                a1, b1 = m1.get(y, (None, None))
                a2, b2 = m2.get(y, (None, None))
                vals_a = [v for v in (a1, a2) if v is not None]
                vals_b = [v for v in (b1, b2) if v is not None]
                if not vals_a or not vals_b:
                    continue
                rows.append((y, min(vals_a), max(vals_b)))
            if len(rows) < 4:
                out.append(c)
                continue
            prev["ranges"] = rows
            prev["left"] = int(min(r[1] for r in rows))
            prev["right"] = int(max(r[2] for r in rows))
            prev["y_top"] = int(rows[0][0])
            prev["y_bottom"] = int(rows[-1][0])
            prev["center_x"] = 0.5 * (prev["left"] + prev["right"])
            prev["contour"] = build_contour(rows)
            prev["merged_fragment"] = True
            changed = True
        cols = out
    return cols


def _split_wide_columns_by_seam(cols, gray, width_factor=1.7, min_contrast=8.0):
    """宽柱暗缝拆分（v7.1.0）。

    相邻柱粘连时会被追踪成一根宽柱（A671318-01M-2 出现 93~106px 宽柱，
    而柱群中位约 35px）。宽度 > factor×中位宽的柱，在其 x 区间内沿柱高统计
    列中位灰度，寻找纵向暗缝（两侧为柱体、缝内明显变暗且缝宽 2~10px），
    在缝中心把柱一分为二。找不到可靠暗缝则保持原样。
    """
    if gray is None or len(cols) < 2:
        return cols
    g = np.asarray(gray, dtype=np.float32)
    h = g.shape[0]
    widths = sorted(float(c["right"] - c["left"] + 1) for c in cols)
    med_w = widths[len(widths) // 2]
    if med_w <= 0:
        return cols
    out = []
    for c in cols:
        L, R = int(c["left"]), int(c["right"])
        w = R - L + 1
        y0, y1 = int(c["y_top"]), int(c["y_bottom"])
        if w <= width_factor * med_w or y1 - y0 < 8:
            out.append(c)
            continue
        ys0, ys1 = max(0, y0 + 1), min(h - 1, y1)
        if ys1 <= ys0:
            out.append(c)
            continue
        # 列剖面：沿柱高的中位灰度（暗缝在该剖面上是局部低谷）
        prof = np.median(g[ys0:ys1 + 1, L:R + 1], axis=0)
        if prof.size < 8:
            out.append(c)
            continue
        body = float(np.median(prof))
        sm = cv2.GaussianBlur(prof.reshape(1, -1), (1, 5), 0).ravel() if prof.size >= 5 else prof
        # 峰（柱体）之间找最深谷，谷两侧要有柱体亮峰
        peaks = [i for i in range(1, len(sm) - 1)
                 if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > body]
        if len(peaks) < 2:
            out.append(c)
            continue
        best = None
        for i in range(len(peaks) - 1):
            a, b = peaks[i], peaks[i + 1]
            if b - a < 4 or b - a > max(10, int(med_w * 0.9)):
                continue
            seg = sm[a:b + 1]
            valley = float(seg.min())
            depth = min(float(sm[a]), float(sm[b])) - valley
            if depth < min_contrast:
                continue
            if best is None or depth > best[0]:
                best = (depth, a + int(np.argmin(seg)))
        if best is None:
            out.append(c)
            continue
        seam = L + best[1]
        if seam - L < 3 or R - seam < 3:
            out.append(c)
            continue
        rows = [(float(r[0]), float(r[1]), float(r[2])) for r in c.get("ranges", [])]
        left_rows = [(y, a, min(b, float(seam))) for y, a, b in rows if b > a]
        right_rows = [(y, max(a, float(seam) + 1), b) for y, a, b in rows if b > a]
        left_rows = [r for r in left_rows if r[2] - r[1] >= 2]
        right_rows = [r for r in right_rows if r[2] - r[1] >= 2]
        if len(left_rows) < 4 or len(right_rows) < 4:
            out.append(c)
            continue
        for idx, sub in enumerate((left_rows, right_rows)):
            nc = dict(c)
            nc["ranges"] = sub
            nc["left"] = int(min(r[1] for r in sub))
            nc["right"] = int(max(r[2] for r in sub))
            nc["y_top"] = int(sub[0][0])
            nc["y_bottom"] = int(sub[-1][0])
            nc["center_x"] = 0.5 * (nc["left"] + nc["right"])
            nc["contour"] = build_contour(sub)
            nc["split_from_wide"] = True
            nc.pop("ranges_subpix", None)
            out.append(nc)
    return sorted(out, key=lambda c: c["center_x"])


def _drop_structurally_invalid(cols, gray, max_h_ratio=0.40, max_width_cv=0.40):
    """无效结构剔除（v7.1.0，针对 A575082 via 单孔）。

    PR 柱阵的高度不超过图像高度的约 40%，且柱身宽度沿高度不应剧烈变化。
    同时满足"过高（>40% 图高）+ 宽度 CV 大（>0.40）"的"柱"实际是把衬底或
    通孔长结构误当柱（A575082：9 根柱高 222~236px、柱身宽 1~55px 剧变）。
    若剔除后无柱且原柱数 >1，保留高度最小、宽度 CV 最小的一根
    （单个通孔的情形）。
    """
    if not cols:
        return cols
    g = np.asarray(gray) if gray is not None else None
    h = int(g.shape[0]) if g is not None else 0
    if h <= 0:
        return cols
    keep, dropped = [], []
    for c in cols:
        rr = np.asarray(c.get("ranges", []), dtype=float)
        if rr.size == 0:
            continue
        height = float(rr[-1, 0] - rr[0, 0] + 1)
        widths = rr[:, 2] - rr[:, 1]
        wcv = float(widths.std() / max(widths.mean(), 1e-9))
        if height > max_h_ratio * h:
            dropped.append((height, wcv, c))
        else:
            keep.append(c)
    if dropped and not keep:
        dropped.sort(key=lambda t: (t[0], t[1]))
        keep = [dropped[0][2]]
    return keep


def _align_bottoms_to_consensus(cols, gray):
    """柱底共识延伸（v7.1.0）：同一排柱的底边应落在同一衬底界面。

    多个收缩守卫（衬底亮带回收、底部离群收缩、梯度脚尾裁剪）叠加时，个别柱
    底会被提到明显高于同伴的位置（"底部靠上"，A4BY067/A4CD536）。此处把高于
    柱群共识底边 max(3px, 10%×柱高) 以上的柱底，用 _extend_rows_checked 向
    下延展到共识位置——该函数自带逐行灰度证据校验，不会拖尾进衬底。
    """
    if gray is None or len(cols) < 2:
        return cols
    g = np.asarray(gray, dtype=np.float32)
    h = g.shape[0]
    valid = [c for c in cols if c.get("ranges") and len(c["ranges"]) >= 4]
    if len(valid) < 2:
        return cols
    bottoms = [int(c["y_bottom"]) for c in valid]
    heights = [int(c["y_bottom"]) - int(c["y_top"]) for c in valid]
    med_b = float(np.median(bottoms))
    med_h = float(np.median(heights)) if heights else 0.0
    tol = max(3.0, 0.10 * med_h)
    sane = [b for b in bottoms if b >= med_b - tol]
    target = float(np.median(sane)) if len(sane) >= 2 else med_b
    for c in valid:
        yb = int(c["y_bottom"])
        if yb >= h - 2 or yb >= target - tol:
            continue
        rows = c.get("ranges")
        new_rows = _extend_rows_checked(rows, g, target)
        if new_rows and len(new_rows) > len(rows):
            c["ranges"] = new_rows
            c["y_bottom"] = int(new_rows[-1][0])
            c["bottom_source"] = "bottom_consensus_ext"
            c["contour"] = build_contour(new_rows)
    return cols


def _shrink_bottom_outliers(cols):
    """柱底聚集收缩（P3.4，TBS CM2 AEX2055 DEV+）。

    overlay 区柱底无结构锚时 column_gradient 会下探 50px（DEV+ 三柱
    ybot=187 vs 主簇 132-137）。同图 >=4 柱且存在 >=3 根底边聚集簇
    （6px 窗）时，超出簇顶 +15px 的离群柱收缩到簇中位。仅当离群柱
    收缩后高度仍 >= MIN_COLUMN_HEIGHT 才生效。
    """
    if len(cols) < 4:
        return cols
    feet = sorted(int(c["y_bottom"]) for c in cols)
    best = []
    for anchor in feet:
        cl = [v for v in feet if abs(v - anchor) <= 6]
        if len(cl) > len(best):
            best = cl
    if len(best) < 3:
        return cols
    target = int(round(float(np.median(best))))
    for c in cols:
        yb = int(c["y_bottom"])
        if yb <= target + 15 or yb <= max(best) + 15:
            continue
        new_rows = [r for r in c["ranges"] if int(r[0]) <= target]
        if len(new_rows) < MIN_COLUMN_HEIGHT:
            continue
        c["ranges"] = new_rows
        c["coarse_ranges"] = [r for r in c.get("coarse_ranges", new_rows)
                              if int(r[0]) <= target] or new_rows
        c["y_bottom"] = target
        c["bottom_source"] = "bottom_cluster_shrink"
        c["contour"] = build_contour(new_rows)
        if "ranges_subpix" in c and c["ranges_subpix"]:
            c["ranges_subpix"] = [r for r in c["ranges_subpix"] if int(r[0]) <= target]
    return cols


def _clip_substrate_band_overreach(cols, gray):
    """衬底亮带越谷回收（P3.3，TBS LCH AEX4337）。

    梯形柱底部精修时 bottom (column_gradient) 会越过柱底暗谷、
    吸到下方衬底亮带（LCH 三柱 ybot 143-150 深入亮带 8-13px）。
    守卫：在 [ybot-28, ybot] 行中值窗内找最深暗谷，要求
    (a) 窗内峰谷差 >=60；(b) 谷行 <= ybot-4；(c) 底行比谷暗 >=30；
    (d) 谷带向下延伸后 4+ 行内出现亮带（>=谷+45）且亮带起点到 ybot
    连续保持 >=谷+30；(e) 底行深入亮带 >=8px。满足则把底部收缩到
    谷带末行。line_pair/flat_block 合成柱自带底边，跳过。
    """
    if gray is None or not cols:
        return cols
    g = np.asarray(gray, dtype=np.float32)
    h = g.shape[0]
    for c in cols:
        if c.get("merged_from") in ("line_pair", "flat_block"):
            continue
        yb = int(c["y_bottom"])
        L = int(c["left"])
        R = int(c["right"])
        if yb >= h - 2 or yb - int(c["y_top"]) < 24:
            continue
        prof = np.median(g[:, max(0, L):R + 1], axis=1)
        y0 = max(0, yb - 28)
        win = [(y, float(prof[y])) for y in range(y0, yb + 1)]
        mx = max(v for _, v in win)
        mn_y, mn_v = min(win, key=lambda t: t[1])
        if mx - mn_v < 60.0 or mn_y > yb - 4 or float(prof[yb]) - mn_v < 30.0:
            continue
        vb = mn_y
        while vb + 1 <= yb and float(prof[vb + 1]) < mn_v + 15.0:
            vb += 1
        below = [float(prof[y]) for y in range(vb + 1, yb + 1)]
        if len(below) < 4 or max(below) < mn_v + 45.0:
            continue
        cand = [k for k, v in enumerate(below) if v >= mn_v + 45.0]
        if not cand:
            continue
        k0 = cand[0]
        if not all(v >= mn_v + 30.0 for v in below[k0:]):
            continue
        if len(below) - k0 < 12:
            continue
        # 收缩 ranges 到谷带末行
        new_rows = [r for r in c["ranges"] if int(r[0]) <= vb]
        if len(new_rows) < MIN_COLUMN_HEIGHT:
            continue
        c["ranges"] = new_rows
        c["coarse_ranges"] = [r for r in c.get("coarse_ranges", new_rows)
                              if int(r[0]) <= vb] or new_rows
        c["y_bottom"] = vb
        c["bottom_source"] = "valley_clip_p33"
        c["contour"] = build_contour(new_rows)
        if "ranges_subpix" in c and c["ranges_subpix"]:
            c["ranges_subpix"] = [r for r in c["ranges_subpix"] if int(r[0]) <= vb]
        try:
            import numpy as _np
            proj = _np.zeros(h, dtype=_np.float32)
            ys_ = [int(r[0]) for r in new_rows]
            proj[ys_] = [float(r[2] - r[1]) for r in new_rows]
            c["y_proj"] = proj
        except Exception:
            pass
    return cols


def _clip_gradient_foot_tails(cols, gray=None):
    """gradient-foot tail clip (P1-C v2, PEB110C_BF PR2).

    Real pillar feet sit on one flat substrate plane. Most columns get
    evidence-based bottoms (base_reference etc.); a few lock onto the
    bright substrate band via column_gradient, adding a constant-width
    rectangular tail. When >=3 evidence feet cluster within 3px and a
    gradient foot exceeds the cluster median by >GRADIENT_FOOT_CLIP_MAX
    with constant-width rows in the overhang, clip to the median.
    """
    if not ENABLE_GRADIENT_FOOT_CLIP or len(cols) < 3:
        return cols
    ev = [c for c in cols if c.get("ranges")
          and str(c.get("bottom_source", "")) not in ("column_gradient", "column_gradient_2nd")]
    grad = [c for c in cols if c.get("ranges")
            and str(c.get("bottom_source", "")) in ("column_gradient", "column_gradient_2nd")]
    if len(ev) < 3 or not grad:
        return cols
    feet = sorted(int(c["y_bottom"]) for c in ev)
    # 稳健簇判定：中位 ± MAD 门。base_reference 各柱独立参照柱外亮带，
    # 抖动 ±3~4px（PEB110C_BF 实测 278~289 全距 11）；拖尾超出 10px+
    # 时其值必落进 MAD 离群区。要求 ≥70% 证据底在 med±4 内才视为齐平。
    med_f = float(np.median(feet))
    mad_f = float(np.median(np.abs(np.asarray(feet, dtype=float) - med_f)))
    if mad_f > 4.0 or sum(1 for f in feet if abs(f - med_f) <= 4.0) < max(3, int(round(0.7 * len(feet)))):
        return cols
    # 裁剪目标取簇最大值（保守）：证据柱间 ±3px 衬底起伏属正常，
    # 只裁掉明显超出全部证据柱的部分（PEB110C_BF: max=286, 拖尾 296）。
    cut_f = float(feet[-1])
    g = np.asarray(gray, dtype=np.float32) if gray is not None else None
    for col in grad:
        rows = col["ranges"]
        foot = int(col["y_bottom"])
        if foot - cut_f <= GRADIENT_FOOT_CLIP_MAX:
            continue
        cut = int(round(cut_f))
        # 柱身灰度证据（P1-C v6，AEX2055JN_BF col4 误杀教训）：恒宽超出段
        # 灰度接近柱身（差 <8）时是真实深柱脚，不裁；明显暗于柱身
        # （差 >=8，衬底亮带特征 PEB105C col3）才裁。
        if g is not None:
            a0 = int(col["left"])
            b0 = int(col["right"])
            if b0 - a0 > 12:
                mid_r = rows[len(rows) // 2]
                body = float(np.median(g[int(mid_r[0]), a0 + 6:b0 - 5]))
                seg = [float(np.median(g[int(y), a0 + 4:b0 - 3]))
                       for y, _, _ in rows if int(y) > cut]
                if seg and float(np.median(seg)) > body - 8.0:
                    continue
        if cut <= int(rows[0][0]) + MIN_COLUMN_HEIGHT - 1 or cut <= rows[0][0] + 7:
            continue
        over = [r for r in rows if int(r[0]) > cut]
        if not over:
            continue
        ows = [int(r[2]) - int(r[1]) for r in over]
        if max(ows) - min(ows) > 2:
            continue
        at_cut = [r for r in rows if int(r[0]) == cut]
        if not at_cut:
            continue
        w_cut = int(at_cut[0][2]) - int(at_cut[0][1])
        if min(ows) < w_cut:
            continue
        new_rows = [r for r in rows if int(r[0]) <= cut]
        if len(new_rows) < 8:
            continue
        col["ranges"] = new_rows
        col["y_bottom"] = int(new_rows[-1][0])
        col["contour"] = build_contour(new_rows)
        col["bottom_source"] = "gradient_foot_clip"
    return cols


def _foot_transition_consensus(cols, g, med):
    """亮带衬底下的逐柱跃迁脚共识（A63/A68D：DP 底集体锁进衬底亮带）。

    判别门（沟道=柱外 x 区，全部满足才启用）：
    1. 足线下方 [med+2,med+9] 沟道灰度比上方 [med-14,med-8] 高 >=10
       （衬底亮带阶跃）；2. 上方带不暗于柱身带（above-body>6，排除
       A67-4/A641 凹陷沟槽边）；3. >=70% 柱在 [med-10,med+10] 内找到
       柱内灰度跃迁脚，且中位 |跃迁脚-DP底| >=4（A58 无跃迁候选不触发）。
    返回 [(col, new_foot_or_None)]，不满足返回 None。
    """
    h, w = g.shape
    mi = int(med)
    if mi < 20 or mi > h - 14:
        return None
    mask = np.ones(w, dtype=bool)
    for c in cols:
        if not c.get("ranges"):
            continue
        mask[max(0, int(c["left"]) - 2):min(w, int(c["right"]) + 3)] = False
    if mask.sum() < w * 0.05:
        return None
    ch = g[:, mask]
    above = float(np.median(ch[mi - 14:mi - 8]))
    bright = float(np.median(ch[mi + 2:mi + 10]))
    body = float(np.median(ch[max(0, mi - 30):max(1, mi - 16)]))
    if bright - above < 10.0 or above - body <= 6.0:
        return None
    cands = []
    for col in cols:
        rows = col.get("ranges") or []
        nf = None
        if len(rows) >= 8:
            L, R = int(col["left"]), int(col["right"])
            if R - L >= 6:
                base = float(np.median([np.median(g[y, L:R + 1])
                                        for y, a, b in rows[:max(4, len(rows) // 3)]]))
                for y in range(max(2, mi - 10), min(h - 2, mi + 10)):
                    m1 = float(np.median(g[y, L:R + 1]))
                    m2 = float(np.median(g[y + 1, L:R + 1]))
                    if m1 - base > 8.0 and m2 - base > 8.0:
                        nf = y - 1
                        break
        cands.append(nf)
    ok = [f for f in cands if f is not None]
    if len(ok) < max(3, int(round(0.7 * len(cols)))):
        return None
    devs = [abs(int(c["y_bottom"]) - f) for c, f in zip(cols, cands) if f is not None]
    if float(np.median(devs)) < 4.0:
        return None
    return list(zip(cols, cands))


def _merge_flat_block_pillars(cols, gray):
    """宽扁亮块模式（P3.2，THS CMOS V1 AEX2948）：亮衬底上的宽扁亮块阵列。

    特征：图像上半区存在整幅亮行带（块+缝混合行中值 >105），其上方
    紧邻大面积真空暗区（行中值 <70）；块体亮（>115）、缝隙暗（列
    中值 <105）。在亮带前半子带（块顶区，衬底亮线尚未把缝列拉亮）
    按列中值找竖直暗缝（w>=8），缝间亮段（w>=15）即宽扁块；逐块
    合成候选（merged_from="flat_block"）交 refine/证据门精修。
    防误触门：子带块列 vs 缝列中值差 >=15、缝数 >=2、块数 >=2、
    上方 14 行真空占比 >=0.5、带高 >= max(12, 0.06h)。
    """
    if gray is None:
        return cols
    g = np.asarray(gray, dtype=np.float32)
    h, w = g.shape
    med = np.median(g, axis=1)
    runs = []
    inr = False
    for y in range(int(0.08 * h), int(0.55 * h)):
        v = med[y] > 105
        if v and not inr:
            inr = True
            s = y
        elif not v and inr:
            inr = False
            runs.append((s, y - 1))
    if inr:
        runs.append((s, int(0.55 * h) - 1))
    runs = [(a, b) for a, b in runs if b - a + 1 >= max(12, int(0.06 * h))]
    new_cols = list(cols)
    for a, b in runs:
        above = med[max(0, a - 14):a]
        if above.size == 0 or float(np.mean(above < 70)) < 0.5:
            continue
        # 前半子带 = 块顶区；整带会混入衬底亮线导致缝列被拉亮
        sa, sb = a, a + max(12, (b - a + 1) // 2) - 1
        colmed = np.median(g[sa:sb + 1], axis=0)
        dark = colmed < 105
        segs = []
        inr = False
        for x in range(w):
            if dark[x] and not inr:
                inr = True
                s0 = x
            elif not dark[x] and inr:
                inr = False
                segs.append((s0, x - 1))
        if inr:
            segs.append((s0, w - 1))
        darks = [(u, v) for u, v in segs if v - u + 1 >= 8]
        if len(darks) < 2:
            continue
        blocks = []
        prev = 0
        for u, v in darks:
            if u - prev >= 15:
                blocks.append((prev, u - 1))
            prev = v + 1
        if w - 1 - prev >= 15:
            blocks.append((prev, w - 1))
        if len(blocks) < 2:
            continue
        bmed = float(np.median([colmed[u2:v2 + 1].mean() for u2, v2 in blocks]))
        dmed = float(np.median([colmed[u2:v2 + 1].mean() for u2, v2 in darks]))
        if bmed - dmed < 15.0:
            continue
        # 块底：子带之后首个行中值 >150 的行（衬底亮线）上沿；
        # 从 sb+1 起扫，跳过块顶边缘亮线（y=a 附近同样 >150）
        bl, br = blocks[len(blocks) // 2]
        ybot = b
        for y in range(sb + 1, min(h, b + 30)):
            if float(np.median(g[y, bl:br + 1])) > 150.0:
                ybot = y - 1
                break
        ytop = max(0, a - 2)
        if ybot - ytop + 1 < MIN_COLUMN_HEIGHT:
            continue
        for L, R in blocks:
            ys = np.arange(ytop, ybot + 1)
            new_rows = [(int(y), int(L), int(R)) for y in ys]
            col = {
                "left": int(L), "right": int(R),
                "center_x": int(round((L + R) * 0.5)),
                "y_top": int(ys[0]), "y_bottom": int(ys[-1]),
                "ranges": new_rows,
                "contour": build_contour(new_rows),
                "merged_from": "flat_block",
            }
            if any(abs(col["center_x"] - int(c["center_x"])) <= max(
                    6, min(R - L, int(c["right"]) - int(c["left"])) // 4)
                   for c in new_cols):
                continue
            new_cols.append(col)
    return sorted(new_cols, key=lambda c: c["center_x"])


def _merge_line_pair_pillars(cols, mask, gray):
    """大柱特写模式（P3.1，TBS 270/300）：SE 边缘亮线对合成宽柱。

    30kx/20kx 特写中宽柱（w 85-200px）内部灰度均质、仅两侧 SE 边缘
    亮线进入二值图，且线宽 <MIN_AVG_WIDTH 被宽门拒掉 → coarse 全空。
    从二值图直接提取竖线（列覆盖率 >=0.55、宽 <=30），两线水平间距
    50-300px、y 覆盖重叠 >=70%，且线间内区灰度 std <28（均质柱体）
    → 合成宽柱候选 [A.left, B.right]（追加到 cols，让 refine 精修）。
    """
    if not ENABLE_BIG_PILLAR_MODE or mask is None or gray is None:
        return cols
    m = np.asarray(mask) > 0
    g = np.asarray(gray, dtype=np.float32)
    h, w = m.shape
    cover = m.mean(axis=0)
    segs = []
    inr = False
    for x in range(w):
        if cover[x] >= 0.55 and not inr:
            inr = True
            s = x
        elif cover[x] < 0.55 and inr:
            inr = False
            segs.append((s, x - 1))
    if inr:
        segs.append((s, w - 1))
    segs = [(a, b) for a, b in segs if 2 <= b - a + 1 <= 30]
    if len(segs) < 2:
        return cols
    # 每条线的 y 覆盖行（用 5%~95% 分位截掉边框/底带噪声行）
    def _ycover(a, b):
        ys = np.where(m[:, a:b + 1].any(axis=1))[0]
        if len(ys) < 12:
            return None
        lo, hi = int(np.percentile(ys, 5)), int(np.percentile(ys, 95))
        return (lo, hi) if hi - lo >= 12 else None
    covers = [_ycover(a, b) for a, b in segs]

    def _has_far_neighbor(idx_i, idx_j):
        # chain-middle test: a far wall line [50,300]px beyond either
        # end means this pair sits inside a wall chain (trench pair).
        for k in range(len(segs)):
            if k in (idx_i, idx_j) or not covers[k]:
                continue
            if 50 <= segs[idx_i][0] - segs[k][1] <= 300:
                return True
            if 50 <= segs[k][0] - segs[idx_j][1] <= 300:
                return True
        return False
    # 贪心配对：每条线只属于一个对，按 gap 从小到大
    pairs = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            gap = segs[j][0] - segs[i][1]
            # gap 下限 110：线对间距须达到大柱宽度量级（TBS300 沟槽
            # 对 gap=85 被排除；大柱对 gap=149/158 保留）
            if 50 <= gap <= 300 and covers[i] and covers[j]:
                pairs.append((gap, i, j))
    pairs.sort()
    used = set()
    new_cols = list(cols)
    for gap, i, j in pairs:
        if i in used or j in used:
            continue
        (la, ra), (lb, rb) = segs[i], segs[j]
        L, R = la, rb
        if R - L + 1 < 60:
            continue
        ta, ba = covers[i]
        tb, bb = covers[j]
        ov = min(ba, bb) - max(ta, tb) + 1
        span = max(ba, bb) - min(ta, tb) + 1
        if span <= 0 or ov / span < 0.70:
            continue
        y0, y1 = max(ta, tb), min(ba, bb)
        if y1 - y0 < 12:
            continue
        il, ir = L + 12, R - 12
        if ir - il < 8:
            continue
        inner = g[y0:y1 + 1, il:ir + 1]
        if float(np.std(inner)) > 28.0:
            continue
        # 大柱内部在二值图中不亮（SE 边缘线模式）：内区 mask 覆盖率低
        if float(m[y0:y1 + 1, il:ir + 1].mean()) > 0.35:
            continue
        # 外侧延续性：真实柱侧壁线的外侧 12px 与内区灰度一致（还是柱体
        # 或同质平台）；沟槽对的外侧是柱壁亮线，灰度差大。
        outs = []
        if L - 12 >= 0:
            outs.append(float(np.median(g[y0:y1 + 1, max(0, L - 12):L - 2])))
        if R + 12 < w:
            outs.append(float(np.median(g[y0:y1 + 1, R + 2:min(w, R + 12)])))
        if outs:
            inmed = float(np.median(inner))
            # v7: reject only when inner is darker than outside by
            # >=10 AND the pair is mid-chain (trench). A lone dark
            # cone (TBS270, no neighbor lines) passes.
            dark_out = inmed <= float(np.median(outs)) - 10.0
            if dark_out and _has_far_neighbor(i, j):
                continue
        ys = np.arange(y0, y1 + 1)
        new_rows = [(int(y), int(L), int(R)) for y in ys]
        col = {
            "left": int(L), "right": int(R),
            "center_x": int(round((L + R) * 0.5)),
            "y_top": int(ys[0]), "y_bottom": int(ys[-1]),
            "ranges": new_rows,
            "contour": build_contour(new_rows),
            "merged_from": "line_pair",
        }
        if any(abs(col["center_x"] - int(c["center_x"])) <= max(6, (R - L) // 4)
               for c in new_cols):
            continue
        new_cols.append(col)
        used.add(i)
        used.add(j)
    return sorted(new_cols, key=lambda c: c["center_x"])


def _align_column_tops_bottoms(cols, gray):
    """跨柱底部 y 对齐（A68D024：衬底亮带使逐柱独立判定参差 166-179）。

    仅当 ≥70% 柱底落在中位数 ±3px 内（多数柱底齐平）才启用，防真实
    台阶衬底被误夹紧。对偏离 >max(3, 2.5×MAD) 的柱：在其 x 范围、
    [med-10, med+10] 窗口用柱内灰度跃迁（相对柱身基线上升）找底；
    跃迁行距 med ≤6px 才采纳。新底更小则截断，更大用上方平台延展。
    """
    if len(cols) < 3 or gray is None:
        return cols
    feet = [int(c["y_bottom"]) for c in cols if c.get("ranges")]
    if len(feet) < 3:
        return cols
    med = float(np.median(feet))
    mad = float(np.median(np.abs(np.asarray(feet) - med)))
    inliers = sum(1 for f in feet if abs(f - med) <= 3.0)
    if inliers < max(3, int(round(0.7 * len(feet)))):
        return cols
    dev_need = max(3.0, 2.5 * mad)
    g = np.asarray(gray, dtype=np.float32)
    h = g.shape[0]
    # 亮带衬底共识（A63/A68D）：DP 底集体锁进衬底亮带时，多数柱"齐平"
    # 是假的——先尝试用逐柱跃迁脚整体贴底，命中则替代下面的离群柱逻辑。
    consensus = _foot_transition_consensus(cols, g, med)
    if consensus is not None:
        for col, nf in consensus:
            rows = col.get("ranges") or []
            if nf is None or nf == int(col["y_bottom"]) or len(rows) < 8:
                continue
            if nf < int(rows[0][0]) + MIN_COLUMN_HEIGHT - 1:
                continue
            if nf < int(col["y_bottom"]):
                new_rows = [(y, a, b) for y, a, b in rows if y <= nf]
            else:
                # P1-C: 柱脚行宽 + 灰度证据校验，防矩形拖尾
                new_rows = (_extend_rows_checked(rows, gray, nf)
                            if ENABLE_FOOT_EXTEND_EVIDENCE else None)
                if new_rows is None:
                    # v6.5 共识延展兜底：共识脚由 >=3 根柱交叉验证，非单柱
                    # 拖尾；灰度证据校验失败（A68D PR6 真实深脚）时按
                    # 柱身平台宽度补行，深度受柱高比例上限约束。
                    gap = nf - int(col["y_bottom"])
                    if gap <= max(20, int(0.35 * (nf - rows[0][0]))):
                        plat = rows[max(0, len(rows) // 2)]
                        pl, pr_ = int(plat[1]), int(plat[2])
                        new_rows = list(rows) + [
                            (int(y), pl, pr_) for y in range(int(col["y_bottom"]) + 1, nf + 1)]
                        col["bottom_source"] = "foot_consensus_ext"
                    else:
                        continue
            if len(new_rows) < 8:
                continue
            col["ranges"] = new_rows
            col["y_top"] = int(new_rows[0][0])
            col["y_bottom"] = int(new_rows[-1][0])
            col["contour"] = build_contour(new_rows)
            col["bottom_source"] = "foot_consensus"
        return cols
    changed = False
    for col in cols:
        rows = col.get("ranges") or []
        if len(rows) < 8:
            continue
        foot = int(col["y_bottom"])
        if abs(foot - med) <= dev_need:
            continue
        L, R = int(col["left"]), int(col["right"])
        if R - L < 6:
            continue
        base = float(np.median([np.median(g[y, L:R + 1])
                                for y, a, b in rows[:max(4, len(rows) // 3)]]))
        y_lo = max(2, int(med) - 10)
        y_hi = min(h - 2, int(med) + 10)
        new_foot = None
        for y in range(max(y_lo, rows[0][0] + 4), y_hi):
            m1 = float(np.median(g[y, L:R + 1]))
            m2 = float(np.median(g[y + 1, L:R + 1]))
            if m1 - base > 8.0 and m2 - base > 8.0:
                new_foot = y - 1
                break
        if new_foot is None or abs(new_foot - med) > 6.0:
            continue
        if new_foot == foot:
            continue
        if new_foot < foot:
            new_rows = [(y, a, b) for y, a, b in rows if y <= new_foot]
        elif ENABLE_FOOT_EXTEND_EVIDENCE:
            # P1-C: 柱脚行宽 + 灰度证据校验，防矩形拖尾
            new_rows = _extend_rows_checked(rows, gray, new_foot)
            if new_rows is None:
                continue
        else:
            plat = rows[max(0, len(rows) // 2)]
            pl, pr_ = int(plat[1]), int(plat[2])
            new_rows = list(rows)
            for y in range(foot + 1, new_foot + 1):
                new_rows.append((int(y), pl, pr_))
        if len(new_rows) < 8:
            continue
        col["ranges"] = new_rows
        col["y_top"] = int(new_rows[0][0])
        col["y_bottom"] = int(new_rows[-1][0])
        col["contour"] = build_contour(new_rows)
        changed = True
    return cols



def _refine_binary_columns(columns, filtered, sign, gray=None):
    h, w = filtered.shape
    gx = cv2.Scharr(filtered, cv2.CV_32F, 1, 0) / 32.0 * sign
    gy = cv2.Scharr(filtered, cv2.CV_32F, 0, 1) / 32.0 * sign
    result = []
    typical_width = float(np.median([_column_width_px(c) for c in columns])) if columns else 20.0
    artifact_top_limit = _horizontal_artifact_limit(gray, columns, sign)
    for original in columns:
        col = dict(original)
        coarse = np.asarray(col["ranges"], dtype=float)
        width = float(np.median(coarse[:, 2] - coarse[:, 1]))
        radius = min(REFINE_RADIUS_PX, max(3, int(max(width, typical_width) * 0.32)))
        y0, y1 = int(coarse[0, 0]), int(coarse[-1, 0])
        x0 = max(0, int(np.median(coarse[:, 1]) + width * 0.2))
        x1 = min(w - 1, int(np.median(coarse[:, 2]) - width * 0.2))
        cap_points = []
        limits = []
        top_strength = 1.0
        for side, seed_y, direction in (("top", y0, 1), ("bottom", y1, -1)):
            xs = np.arange(x0, max(x0 + 1, x1 + 1))
            if side == "top":
                # v6.5：首轮顶窗放宽（5r/2,26 -> 4r,32），真实顶边距种子
                # 更远时首轮即可命中，减少 binary_fallback 头部缺失。
                up_r, dn_r = min(4 * radius, 32), min(radius, 5)
            else:
                up_r, dn_r = min(radius, 5), min(2 * radius, 13)
            if side == "top" and artifact_top_limit > 0:
                up_r = max(2, min(up_r, seed_y - artifact_top_limit))
            lo, hi = max(0, seed_y - up_r), min(h - 1, seed_y + dn_r)
            local = (direction * gy[lo:hi + 1, xs]).T
            pull = 0.6 if side == "top" else 0.0
            pos, strengths, valid = _constrained_edge_path(local, np.full(len(xs), seed_y - lo), radius, pull=pull)
            for _ext in range(3):
                if side != "top" or not len(valid) or not np.count_nonzero(valid):
                    break
                med_pos = float(np.median((pos + lo)[valid]))
                if med_pos - lo > 2 or lo <= 0:
                    break
                prev_med_pos = med_pos
                # 收益门：扩窗后路径强度须显著更高才接受；否则上一轮
                # 结果可能只是吸向柱顶上方的弱横向特征（横线伪影）。
                prev_strength = float(np.median(strengths[valid]))
                up_r = min(up_r * 2, 40)
                lo = max(0, seed_y - up_r)
                local = (direction * gy[lo:hi + 1, xs]).T
                pos2, strengths2, valid2 = _constrained_edge_path(local, np.full(len(xs), np.clip(seed_y - lo, 0, hi - lo)), radius, pull=pull)
                if np.count_nonzero(valid2):
                    new_strength = float(np.median(strengths2[valid2]))
                    med_pos2 = float(np.median((pos2 + lo)[valid2]))
                    moved_up = (prev_med_pos - med_pos2) >= 2.0
                    # v6.2.1 P1-B：门限降到 TOP_EXTEND_GAIN_GATE；扩窗后路径
                    # 整体上移且强度不降也算真实顶边证据，放行（去横线残留）。
                    if (new_strength < TOP_EXTEND_GAIN_GATE * prev_strength
                            and not (moved_up and new_strength >= 0.95 * prev_strength)):
                        break  # 扩窗无收益：保留上一轮，阻断吸附
                pos, strengths, valid = pos2, strengths2, valid2
            for x, y, strength, ok in zip(xs, pos + lo, strengths, valid):
                cap_points.append({"side": side, "x": float(x), "y": float(y),
                                   "strength": float(strength), "valid": bool(ok), "source": "column_gradient"})
            # v6.5 亮带守卫复核：钳位把顶窗压得过窄（up_r<6）时，钳位 DP
            # 可能错过真顶；在不越过亮带底的前提下以 artifact_top_limit 为
            # 上界再跑一次宽窗，取有效列更多且中位强度更高者。
            if (side == "top" and artifact_top_limit > 0 and up_r < 6
                    and seed_y - artifact_top_limit > up_r):
                lo_w = max(0, artifact_top_limit)
                local_w = (direction * gy[lo_w:hi + 1, xs]).T
                pos_w, strengths_w, valid_w = _constrained_edge_path(
                    local_w, np.full(len(xs), seed_y - lo_w), radius, pull=0.6)
                if (np.count_nonzero(valid_w) > np.count_nonzero(valid)
                        and float(np.median(strengths_w)) >= float(np.median(strengths))):
                    lo = lo_w
                    pos, strengths, valid = pos_w, strengths_w, valid_w
                    cap_points = [cp for cp in cap_points
                                  if not (cp["side"] == side and cp.get("source") == "column_gradient")]
                    for x, y, strength, ok in zip(xs, pos + lo, strengths, valid):
                        cap_points.append({"side": side, "x": float(x), "y": float(y),
                                           "strength": float(strength), "valid": bool(ok),
                                           "source": "column_gradient"})
                    cap_points = [cp for cp in cap_points
                                  if not (cp["side"] == side and cp.get("source") == "column_gradient")]
                    for x, y, strength, ok in zip(xs, pos + lo, strengths, valid):
                        cap_points.append({"side": side, "x": float(x), "y": float(y),
                                           "strength": float(strength), "valid": bool(ok),
                                           "source": "column_gradient"})
            valid_y = (pos + lo)[valid]
            source = "column_gradient"
            if side == "top":
                top_strength = float(np.median(strengths))
            # v6.4 bottom 弱证据二道闸：底边证据弱时以小 pull 重跑一次，
            # 惩罚向上吸附（基底条带/亮带），防底边被抬高导致 CD 塌缩；
            # 强证据底边不重跑，行为与旧版逐位一致。
            if (side == "bottom" and BOTTOM_PULL_WEAK > 0
                    and float(np.median(strengths)) < max(1.5, 0.15 * top_strength)):
                pos_b, strengths_b, valid_b = _constrained_edge_path(
                    local, np.full(len(xs), seed_y - lo), radius, pull=BOTTOM_PULL_WEAK)
                if np.count_nonzero(valid_b) >= max(2, len(xs) // 3):
                    pos, strengths, valid = pos_b, strengths_b, valid_b
                    for x, y, strength, ok in zip(xs, pos + lo, strengths, valid):
                        cap_points.append({"side": side, "x": float(x), "y": float(y),
                                           "strength": float(strength), "valid": bool(ok),
                                           "source": "column_gradient_pulled"})
                    valid_y = (pos + lo)[valid]
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
            if (side == "bottom" and source == "column_gradient" and len(valid_y) > 0
                    and float(np.median(strengths)) < max(1.5, 0.15 * top_strength)
                    and len(valid) >= 3 and bool(valid[-3:].all())
                    and float(np.min(pos[-3:])) >= (hi - lo) - 1.5):
                dn2 = min(2 * radius, 13)
                c2 = float(np.median(valid_y))
                lo2, hi2 = max(0, int(round(c2)) - dn2), min(h - 1, int(round(c2)) + dn2)
                if hi2 - lo2 >= 4:
                    loc2 = (direction * gy[lo2:hi2 + 1, xs]).T
                    p2, s2, v2 = _constrained_edge_path(
                        loc2, np.full(len(xs), float(np.clip(c2 - lo2, 0, hi2 - lo2))), radius)
                    vy2 = (p2 + lo2)[v2]
                    if (int(np.count_nonzero(v2)) >= max(2, len(xs) // 3)
                            and float(np.median(s2)) >= 1.15 * float(np.median(strengths))
                            and abs(float(np.median(vy2)) - c2) > 1.0):
                        valid_y = vy2
                        source = "column_gradient_2nd"
            # v6.5 顶部弱证据两段式：顶边 DP 全败（binary_fallback）时，
            # 以弱向上粘滞 pull 再搜一次；仅当新顶比二值种子更高（y 更小）
            # 且强度不显著变差才采纳，防止下坠假顶。
            if (side == "top" and TOP_RESCUE_PULL > 0 and source == "binary_fallback"):
                pos_t, strengths_t, valid_t = _constrained_edge_path(
                    local, np.full(len(xs), seed_y - lo), radius, pull=TOP_RESCUE_PULL)
                if (int(np.count_nonzero(valid_t)) >= max(2, len(xs) // 3)
                        and float(np.median(strengths_t)) >= TOP_RESCUE_GATE * float(np.median(strengths))):
                    med_t = float(np.median((pos_t + lo)[valid_t]))
                    med_cur = float(np.median(valid_y)) if len(valid_y) else float(seed_y)
                    if med_t < med_cur - 1.0:
                        pos, strengths, valid = pos_t, strengths_t, valid_t
                        valid_y = (pos + lo)[valid]
                        source = "column_gradient"
                        for x, y, strength, ok in zip(xs, pos + lo, strengths, valid):
                            cap_points.append({"side": side, "x": float(x), "y": float(y),
                                               "strength": float(strength), "valid": bool(ok),
                                               "source": "column_gradient_top_rescue"})
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
        left, ls, lv = _constrained_edge_path(gx[ys], seeds_l, radius, cell_l, np.floor(middle - 1))
        right, rs, rv = _constrained_edge_path(-gx[ys], seeds_r, radius, np.ceil(middle + 1), cell_r)
        # v6.4 曲率正则二阶 DP + 逐行仲裁：圆帽拐角/锥形过渡处一阶 DP 被
        # 柱身延伸拽直（A63 帽沿）。整段计算 curved 路径后交仲裁器：
        # 仅"证据更强且粗糙度不增"的成段行被采纳，帽区外两路结果几乎
        # 相同，输出与一阶 DP 保持 bit-exact。
        try:
            cl, sl, _vl = _wall_path_curved(gx[ys], seeds_l, radius, cell_l, np.floor(middle - 1))
            new_left, new_ls, new_lv, _ = _arbitrate_wall(left, ls, cl, sl)
            if new_lv is not None:
                left, ls, lv = new_left, new_ls, new_lv
            cr, sr, _vr = _wall_path_curved(-gx[ys], seeds_r, radius, np.ceil(middle + 1), cell_r)
            new_right, new_rs, new_rv, _ = _arbitrate_wall(right, rs, cr, sr)
            if new_rv is not None:
                right, rs, rv = new_right, new_rs, new_rv
        except Exception:
            pass  # 二阶 DP/仲裁异常时保留一阶结果
        for coords, seeds, valid in ((left, seeds_l, lv), (right, seeds_r, rv)):
            clipped = (seeds <= 0.5) | (seeds >= w - 1.5)
            if np.mean(clipped) < .1:
                clipped[:] = False
            coords[clipped] = seeds[clipped]
            valid[clipped] = False
        bad = right <= left + 1
        left[bad], right[bad] = seeds_l[bad], seeds_r[bad]
        lv[bad] = rv[bad] = False
        rows = [(int(y), float(a), float(b)) for y, a, b in zip(ys, left, right)]
        sub_rows = [(int(y), float(a), float(b)) for y, a, b in zip(ys, left, right)]

        def _flatten_wall(seq):
            seq = np.asarray(seq, dtype=np.float32)
            n = len(seq)
            if n < 12:
                return seq
            i = 0
            while i < n:
                j = i
                while j + 1 < n and abs(seq[j + 1] - seq[i]) <= 2.0:
                    j += 1
                run = j - i + 1
                if 1 <= run <= 12 and i > 0 and j < n - 1:
                    lo = float(np.median(seq[max(0, i - 4):i]))
                    hi = float(np.median(seq[j + 1:min(n, j + 5)]))
                    steep_in = abs(seq[i] - seq[i - 1]) > 12.0 or i == 1
                    steep_out = abs(seq[min(j + 1, n - 1)] - seq[j]) > 12.0 or j == n - 2
                    if (abs(lo - hi) > 1.0 and min(abs(seq[i] - lo), abs(seq[j] - hi)) > 12.0
                            and steep_in and steep_out):
                        seq[i:j + 1] = np.linspace(lo, hi, run)
                        i = j + 1
                        continue
                i = j + 1
            return seq

        points = cap_points
        if len(rows) >= 9:
            arr = np.asarray(rows, dtype=np.float32)
            med_l = float(np.median(arr[:, 1]))
            med_r = float(np.median(arr[:, 2]))
            med_st_l = float(np.median(ls)) if len(ls) else 0.0
            med_st_r = float(np.median(rs)) if len(rs) else 0.0
            bad_l = [i for i in range(len(ys))
                     if (ls[i] < med_st_l / 3.0
                         and abs(arr[i, 1] - med_l) > max(8.0, 0.25 * radius))]
            bad_r = [i for i in range(len(ys))
                     if (rs[i] < med_st_r / 3.0
                         and abs(arr[i, 2] - med_r) > max(8.0, 0.25 * radius))]

            def _interp_bad(bad, colv, arr_):
                good = [i for i in range(len(arr_)) if i not in bad]
                if not good:
                    return
                for i in bad:
                    lo_g = max((g for g in good if g < i), default=None)
                    hi_g = min((g for g in good if g > i), default=None)
                    if lo_g is None:
                        arr_[i] = arr_[hi_g]
                    elif hi_g is None:
                        arr_[i] = arr_[lo_g]
                    else:
                        w = (i - lo_g) / (hi_g - lo_g)
                        arr_[i] = arr_[lo_g] * (1 - w) + arr_[hi_g] * w
            _interp_bad(bad_l, 1, arr[:, 1])
            _interp_bad(bad_r, 2, arr[:, 2])

            def _mad_spike_rows(w_seq, k=3.0):
                """v6.4 逐行宽度 MAD 突刺标记：|w−median| > 3×(1.4826·MAD)+1px。

                单调台阶豁免：突刺段若与前后行构成单调台阶且累计幅度显著
                （真实 footing/锥度），保留不剔。
                """
                w_seq = np.asarray(w_seq, dtype=float)
                n = len(w_seq)
                if n < 9:
                    return np.zeros(n, bool)
                med = float(np.median(w_seq))
                mad = float(np.median(np.abs(w_seq - med)))
                scale = 1.4826 * mad if mad > 1e-9 else (float(np.std(w_seq)) or 1.0)
                bad = np.abs(w_seq - med) > k * scale + 1.0
                if not np.any(bad):
                    return bad
                # 连片段检查：与前后行构成单调且累计幅度显著 → 豁免
                i = 0
                while i < n:
                    if not bad[i]:
                        i += 1
                        continue
                    j = i
                    while j + 1 < n and bad[j + 1]:
                        j += 1
                    before = w_seq[max(0, i - 1)]
                    after = w_seq[min(n - 1, j + 1)]
                    seg = w_seq[i:j + 1]
                    chain = np.r_[before, seg, after]
                    monotonic = (np.all(np.diff(chain) >= -1.0)
                                 or np.all(np.diff(chain) <= 1.0))
                    if monotonic and abs(after - before) >= max(0.05 * abs(med), 2.0):
                        bad[i:j + 1] = False
                    i = j + 1
                return bad

            spike_rows = _mad_spike_rows(arr[:, 2] - arr[:, 1])
            if np.any(spike_rows):
                spike_idx = [int(i) for i in np.where(spike_rows)[0]]
                _interp_bad(spike_idx, 1, arr[:, 1])
                _interp_bad(spike_idx, 2, arr[:, 2])
            arr[:, 1] = _flatten_wall(arr[:, 1])
            arr[:, 2] = _flatten_wall(arr[:, 2])
            rows = [(int(y), int(a), int(b)) for y, a, b in arr]
        for side, coords, strength, valid in (("left", left, ls, lv), ("right", right, rs, rv)):
            for y, x, st, ok in zip(ys, coords, strength, valid):
                points.append({"side": side, "x": float(x), "y": float(y), "strength": float(st),
                               "valid": bool(ok), "source": "row_gradient" if ok else "binary_fallback"})
        if gray is not None and len(rows) >= 8:
            rows = _rebuild_cap_profile(rows, np.asarray(gray), sign, w)
            rows = _trim_head_artifact_rows(rows, np.asarray(gray))
            rows = _smooth_top_notches(rows, float(np.median([b - a for _, a, b in rows])))
            rows = _repair_foot_width_drop(rows, np.asarray(gray))
            rows = _repair_foot_ramp_collapse(rows, np.asarray(gray))
        sub_out = sub_rows if (len(sub_rows) == len(rows)
                               and all(int(r[0]) == int(s[0]) for r, s in zip(rows, sub_rows))) else None
        col.update(ranges=rows, coarse_ranges=original["ranges"], points=points,
                   ranges_subpix=sub_out,
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
    gray_hint = _main_row_activity(gray, prefer_upper=not roi_mode)[:2]
    trials = []
    winner = None
    best_score = -1e9

    def _run_variant_set(oriented, sign, accept=None):
        # 单一极性下的 3 变体 × 2 hint 种子试跑（原 bright 循环体提取，
        # 供 dark 兜底复用）；accept=None 时与原仲裁行为逐位一致。
        nonlocal winner, best_score
        enhanced = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(oriented)
        xx = np.linspace(-1, 1, w, dtype=np.float32)
        column_mean = np.mean(oriented, axis=0)
        slope = float(np.dot(column_mean - column_mean.mean(), xx) / max(float(np.dot(xx, xx)), 1e-6))
        corrected = cv2.normalize(oriented.astype(np.float32) - slope * xx[None, :],
                                  None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for name, source in (("filtered_otsu", oriented), ("clahe_otsu", enhanced), ("illumination_otsu", corrected)):
            _, mask = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            for hint_tag, hint in (("yhint", gray_hint), ("raw", None)):
                cols = _binary_seed_columns(mask, oriented, roi_mode, y_hint=hint)
                cols = _pillar_evidence(_apply_cd_hint(cols, w), gray, sign, image_width=w)
                score = _detection_confidence(cols, oriented)
                trials.append({"variant": name if hint_tag == "raw" else name + "+yhint",
                               "polarity": "bright" if sign == 1 else "dark",
                               "count": len(cols), "score": float(score)})
                if cols and score > best_score and (accept is None or accept(cols, score)):
                    winner, best_score = (cols, mask, sign, name), score

    for sign in sign_options:
        oriented = filtered if sign == 1 else 255 - filtered
        _run_variant_set(oriented, sign)
        # 亮柱弱仲裁兜底（A671318-01M-5：衬底偏暗，中间暗方柱在亮柱
        # Otsu mask 中无投影段，候选从未诞生）。bright 全场弱分时补跑
        # 一轮 dark trial；胜出须过结构门，防止暗沟道噪声翻盘。
        if PILLAR_POLARITY == "bright" and best_score < 4.0:
            bright_count = len(winner[0]) if winner else 0
            bright_score = best_score

            # v6.5：bright 胜者常含贴边伪柱（M-5: 4 根中 2 根贴边/窄条），
            # 直接比较总柱数会把 2 根真暗柱拒掉。改为对比非贴边主体数，
            # 并在 bright 中位柱宽与自动 CD 估计明显冲突（CD hint 冲突）时
            # 收紧分数门，让结构更合理的暗极性候选翻盘。
            def _body_count(cols_):
                return sum(1 for c in cols_
                           if int(c.get("left", 0)) > 1
                           and int(c.get("right", w - 1)) < w - 2)

            def _cd_hint_conflict(cols_):
                # v6.5：无用户 CD 先验时改用 bright 胜者的宽度一致性判断——
                # bright 柱宽离散大（伪柱混入，M-5: 45/23/20/35）而 dark 候选
                # 宽度集中（58/56）时，视为结构冲突，允许暗柱数少于亮柱。
                if not cols_ or not (winner[0] if winner else []):
                    return False
                def _spread(cs):
                    ws = np.asarray([float(c.get("right", 0)) - float(c.get("left", 0)) + 1
                                     for c in cs], dtype=np.float32)
                    if ws.size == 0:
                        return 1e9
                    med = float(np.median(ws))
                    return float(np.median(np.abs(ws - med))) / max(2.0, med * 0.22)
                dark_spread = _spread(cols_)
                bright_spread = _spread(list(winner[0]))
                return dark_spread < 0.5 * bright_spread

            def _overlap_conflict(dc):
                # v6.5.1：亮/暗两种柱体解释若高度重叠则自相矛盾——同一
                # x 区间不可能既是亮柱又是暗柱。度量：bright 胜者中
                # 非贴边柱的柱身被 dark 候选覆盖的平均比例 >50% 时拒绝
                # dark（A55R144：dark 实为沟道/衬底，把 bright 真柱覆盖
                # 50.8%/70.5%；M-5：bright 伪柱仅被覆盖 15.9%/43.5%，
                # 均值 29.7%，真暗柱解释自洽，不受影响）。
                bw = [b for b in (winner[0] if winner else [])
                      if int(b.get("left", 0)) > 1
                      and int(b.get("right", w - 1)) < w - 2]
                if not bw or not dc:
                    return False
                covs = []
                for b in bw:
                    bl, br = int(b["left"]), int(b["right"])
                    bwid = br - bl + 1
                    if bwid <= 0:
                        continue
                    ov = 0
                    for c in dc:
                        seg = (min(br, int(c["right"])) - max(bl, int(c["left"])) + 1)
                        if seg > 0:
                            ov += seg
                    covs.append(min(1.0, ov / float(bwid)))
                return bool(covs) and float(np.mean(covs)) > 0.5

            def _accept_dark(dc, ds):
                if _overlap_conflict(dc):
                    return False
                score_need = bright_score + (0.15 if _cd_hint_conflict(dc) else 0.25)
                if ds <= score_need:
                    return False
                if len(dc) < 2:
                    return False
                if _cd_hint_conflict(dc):
                    if len(dc) < max(2, sum(
                            1 for c in (winner[0] if winner else [])
                            if int(c.get("left", 0)) > 1
                            and int(c.get("right", w - 1)) < w - 2)):
                        return False
                elif len(dc) < bright_count:
                    return False
                if any(int(c["left"]) <= 1 or int(c["right"]) >= w - 2 for c in dc):
                    return False
                return True

            _run_variant_set(255 - filtered, -1, accept=_accept_dark)
        # v6.5 Otsu 阈值偏移重试：全场弱分（best_score<4.0）时对胜出极性
        # 补跑 otsu×0.88 / ×1.12 两档阈值（M-5：衬底偏暗使标准 Otsu 把
        # 真柱并进衬底，降阈值可分离），纳入既有 trials 仲裁。
        if best_score < 4.0 and winner is not None:
            w_sign = winner[2]
            w_src = winner[3] if len(winner) > 3 else None
            oriented_w = filtered if w_sign == 1 else 255 - filtered
            otsu_v, _ = cv2.threshold(oriented_w, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for k in (0.88, 1.12):
                lvl = float(otsu_v) * k
                _, mask_k = cv2.threshold(oriented_w, lvl, 255, cv2.THRESH_BINARY)
                mask_k = cv2.morphologyEx(mask_k, cv2.MORPH_CLOSE,
                                          np.ones((3, 3), np.uint8))
                for hint_tag, hint in (("yhint", gray_hint), ("raw", None)):
                    cols_k = _binary_seed_columns(mask_k, oriented, roi_mode, y_hint=hint)
                    cols_k = _pillar_evidence(_apply_cd_hint(cols_k, w), gray, w_sign, image_width=w)
                    score_k = _detection_confidence(cols_k, oriented_w)
                    trials.append({"variant": ("otsu%.2f" % k) + ("+yhint" if hint_tag == "yhint" else ""),
                                   "polarity": "bright" if w_sign == 1 else "dark",
                                   "count": len(cols_k), "score": float(score_k)})
                    if cols_k and score_k > best_score:
                        winner, best_score = (cols_k, mask_k, w_sign, "otsu_offset"), score_k
        # 几何提议总是参与仲裁（-0.25 罚分）：binary 全败时兜底；
        # binary 只有碎片段时（A67-5，宽 8/23/18/23 的假柱）由置信度
        # 仲裁让位给几何路径。
        # filtered 提案可能丢失弱对比残柱（A66 右缘 x612），再用原图
        # gray 跑一次并合并：按 x 中心近邻去重，保留宽度更大者。
        proposals, _ = _detect_columns_geometric(oriented, roi_mode=roi_mode, y_hint=gray_hint)
        proposals_g, _ = _detect_columns_geometric(gray, roi_mode=roi_mode, y_hint=gray_hint)
        merged_props = []
        for cand in (proposals or []) + (proposals_g or []):
            dup = None
            for m in merged_props:
                mc = 0.5 * (m["left"] + m["right"])
                cc = 0.5 * (cand["left"] + cand["right"])
                if abs(mc - cc) <= max(6.0, 0.35 * max(m["right"] - m["left"],
                                                       cand["right"] - cand["left"])):
                    dup = m
                    break
            if dup is None:
                merged_props.append(cand)
            else:
                if (cand["right"] - cand["left"]) > (dup["right"] - dup["left"]):
                    merged_props[merged_props.index(dup)] = cand
        # 包含关系去重（A67-4：gray 提案的单柱 x81-102 与 filtered 提案
        # 的连体 x81-247 中心距超近邻阈值而共存，拆分后子柱与单柱重叠）。
        # 窄柱 x 范围被宽柱覆盖 ≥80% 时删窄者；互含时保留宽者。
        merged_props.sort(key=lambda c: c["left"])
        dedup = []
        for c in merged_props:
            drop = False
            for m in dedup:
                ov = max(0, min(c["right"], m["right"]) - max(c["left"], m["left"]) + 1)
                wc = c["right"] - c["left"] + 1
                wm = m["right"] - m["left"] + 1
                if ov >= 0.8 * min(wc, wm):
                    if wc > wm:
                        dedup[dedup.index(m)] = c
                    drop = True
                    break
            if not drop:
                dedup.append(c)
        merged_props = dedup
        proposals = merged_props or None
        proposals = _split_merged_geometric_columns(proposals or [], gray, filtered)
        gcols = _pillar_evidence(_apply_cd_hint(proposals or [], w), gray, sign)
        if gcols:
            gmask = np.zeros_like(gray)
            for col in gcols:
                cv2.drawContours(gmask, [col["contour"]], -1, 255, -1)
            gscore = _detection_confidence(gcols, oriented) - 0.25
            trials.append({"variant": "geometric_seed", "polarity": "bright" if sign == 1 else "dark", "count": len(gcols), "score": float(gscore)})
            if gscore > best_score:
                winner, best_score = (gcols, gmask, sign, "geometric_seed"), gscore
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
    # P3.2: 宽扁亮块模式（V1 AEX2948）——亮衬底宽扁块阵列合成候选
    cols = _merge_flat_block_pillars(cols, gray)
    # P3.2: flat_block 图中，远离块带的胜者柱（水印/角标）为伪柱，剔除
    _fb = [c for c in cols if c.get("merged_from") == "flat_block"]
    if _fb:
        _band_bot = max(c["y_bottom"] for c in _fb)
        cols = [c for c in cols if c.get("merged_from") == "flat_block"
                or int(c["y_top"]) <= _band_bot + 20]
    # P3.1: 大柱特写模式——SE 边缘线对合成宽柱（refine 前追加候选）
    cols = _merge_line_pair_pillars(cols, binary, gray)
    cols = sorted(_refine_binary_columns(cols, filtered, sign, gray=gray), key=lambda c: c["center_x"])
    # P3.1 v7: drop coarse wall-stub columns fully covered by a
    # merged line_pair wide pillar (TBS270 stubs inside w96).
    _lp = [c for c in cols if c.get("merged_from") == "line_pair"]
    if _lp:
        cols = [c for c in cols if c.get("merged_from") == "line_pair"
                or not any(c["left"] >= p["left"] and c["right"] <= p["right"]
                           for p in _lp)]
    cols = _dedupe_refined_columns(cols)
    cols = _merge_overlapping_columns(cols, gray)
    cols = _align_column_tops_bottoms(cols, gray)
    # P1-C v2: gradient-foot tail clip after alignment
    cols = _clip_gradient_foot_tails(cols, gray)
    # P3.3: 衬底亮带越谷回收（LCH 梯形柱底部下探守卫）
    cols = _clip_substrate_band_overreach(cols, gray)
    # P3.4: 柱底聚集收缩（DEV+ overlay 区柱底下探限制）
    cols = _shrink_bottom_outliers(cols)
    # v7.1.0: 断行填补（柱体纵向连续性）→ 柱底共识延伸（底边对齐衬底界面），
    # 修正"柱中间断开"与"底部偏上"；两者都在收缩守卫之后执行。
    cols = _fill_row_gaps(cols)
    # v7.1.0: 先按暗缝拆分粘连宽柱，再合并被切碎的窄条碎片柱。
    cols = _split_wide_columns_by_seam(cols, gray)
    cols = _merge_fragment_columns(cols, gray)
    # v7.1.0: 剔除"过高 + 宽窄剧变"的无效结构（误把衬底/通孔长结构当柱）。
    cols = _drop_structurally_invalid(cols, gray)
    cols = _align_bottoms_to_consensus(cols, gray)
    # S3 超大柱宽守卫：柱宽 >2×默认精修半径且边缘支持弱（DP 半径封顶
    # 32px 够不到真实侧壁）时，按柱宽比例放大半径对该柱重跑一次精修。
    # 仅当重跑后 edge_support 提高才采纳（防误触 A66 粘连柱——其
    # edge_support 达标不触发）。
    if cols:
        ws_ = sorted(float(c["right"] - c["left"] + 1) for c in cols)
        med_w_ = ws_[len(ws_) // 2]
        if med_w_ > 2 * REFINE_RADIUS_PX:
            weak = [c for c in cols
                    if float(c.get("edge_support", 1.0)) < 0.35
                    and (c["right"] - c["left"] + 1) > 2 * REFINE_RADIUS_PX]
            if weak:
                weak_ids = {id(c) for c in weak}
                others = [c for c in cols if id(c) not in weak_ids]
                radius_big = min(int(0.4 * med_w_), max(3, int(0.2 * w)))
                redone = _refine_binary_columns([dict(c) for c in weak], filtered, sign, gray=gray)
                # 精修可能改变 left/center_x，按输入顺序或就近中心匹配，
                # 避免坐标漂移后 next() 抛 StopIteration（A4BS588 实测）。
                if len(redone) == len(weak):
                    redone = [c for c, o in zip(redone, weak)
                              if float(c.get("edge_support", 0.0)) > float(
                                  o.get("edge_support", 0.0))]
                else:
                    kept = []
                    for c in redone:
                        cx = c.get("center_x", 0.5 * (c["left"] + c["right"]))
                        o = min(weak, key=lambda q: abs(
                            q.get("center_x", 0.5 * (q["left"] + q["right"])) - cx))
                        if float(c.get("edge_support", 0.0)) > float(
                                o.get("edge_support", 0.0)):
                            kept.append(c)
                    redone = kept
                if redone:
                    cols = sorted(others + redone, key=lambda c: c["center_x"])
    for i, col in enumerate(cols, 1):
        col["detected_label"] = i
    detected_count = len(cols)
    excluded = []
    if not roi_mode and cols:
        # 两端排除只针对"贴边裁切柱"（A66/A67-4：视野边缘被裁掉的
        # 残柱不可测量）；完整柱即使位于最左/最右也保留。规则：
        # 1) 两端候选中贴边的（left<=1 或 right>=w-2）排除；
        # 2) 若某端没有贴边候选，则回退排除该端最外侧柱（保持旧行为，
        #    避免单根完整柱无法清场）。v6.5 起回退排除若会清场全部
        #    柱（M-5：两根真暗柱都不贴边，双双被回退排除 → 0 柱），
        #    则放弃回退剔除，保留全部柱。
        left_idx = [0]
        right_idx = [len(cols) - 1]
        drop = set()
        # 大柱特写图（line_pair 合成）：全图仅 1-2 根宽柱是测量主体，
        # "回退排除最外侧柱"会把仅有的柱全灭（TBS 270/300）。此类图
        # 只排除真正贴边的残柱。
        big_pillar_img = any(c.get("merged_from") in ("line_pair", "flat_block")
                             for c in cols)
        for side, idx in (("left", left_idx), ("right", right_idx)):
            i = idx[0]
            c = cols[i]
            if int(c["left"]) <= 1 or int(c["right"]) >= w - 2:
                # P3.2: 贴边 flat_block（V1 左右块）是测量主体，不剔
                if c.get("merged_from") != "flat_block":
                    drop.add(i)
            elif len(cols) >= 2 and not big_pillar_img:
                # 该端无贴边残柱：回退排除最外侧柱。
                drop.add(i)
        # v6.5 清场保护：若回退剔除会导致 0 柱剩余，则撤销回退剔除
        # （只保留真正贴边残柱的强制排除），避免真柱被误清。
        if len(drop) >= len(cols):
            drop = {i for i in drop
                    if int(cols[i]["left"]) <= 1 or int(cols[i]["right"]) >= w - 2}
        excluded = [cols[i] for i in sorted(drop)]
        cols = [c for i, c in enumerate(cols) if i not in drop]
    for i, col in enumerate(cols, 1):
        col.update(label=i, color=PR_COLORS[(i - 1) % len(PR_COLORS)])
    warning = ""
    if not cols and excluded:
        warning = f"检测到 {detected_count} 根柱，按两端排除规则后无剩余柱；可用 ROI 单独测量。"
    elif not cols:
        warning = "未找到满足极性与边缘证据的柱子；请检查 ROI 或柱体明暗模式。"
    elif cols and float(np.median([c["y_top"] for c in cols])) > 0.7 * h:
        # S3 兜底：主体不在惯常上半区（构图下移/特写图），提示人工核对。
        warning = "柱体主体位于图像下半区（非常规构图），请人工核对结果。"
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
    # 显示用剖面平滑（x-structure 曲线降噪）；CD/H 仍由精修 ranges 计算。
    if profile.size:
        profile = smooth_1d(profile, 7)
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
# 管线 / 导出层
# =====================================================================

def _quality_warnings(columns, meta, limit=20):
    """v6.4 质量预警：把已有质量信号组装成可读列表（仅报告，不改变识别）。

    覆盖：整体置信度低、单柱边缘证据不足、底边二值回退、图像边界截断。
    """
    warns = []
    try:
        conf = meta.get("confidence") if isinstance(meta, dict) else None
        if conf is not None and float(conf) < 4.0:
            warns.append(f"整体置信度 {float(conf):.2f} < 4.0（暗图/低对比），建议复核")
    except Exception:
        pass
    for col in columns:
        try:
            label = col.get("label", "?")
            support = float(col.get("edge_support", 1.0))
            if support < 0.6:
                warns.append(f"PR{label} 边缘证据不足 (edge_support={support:.2f})")
            if col.get("bottom_source") == "binary_fallback":
                warns.append(f"PR{label} 底边为二值回退（梯度证据弱）")
            if col.get("clipped"):
                warns.append(f"PR{label} 触及图像边界（截断）")
        except Exception:
            continue
    return warns[:limit]


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
    v6.3: 先计算形态学分类，再渲染——结果图顶部带整图 profile 分类横幅。
    """
    stem, out_dir = _output_dir_for(image_path) if output_dir is None else (os.path.splitext(os.path.basename(image_path))[0], output_dir)
    os.makedirs(out_dir, exist_ok=True)
    scale = validate_scale(scale or read_image_scale(image_path))
    nm_per_px = scale["nm"] / scale["pixels"]

    meta = _detection_meta_from_aux(detection_meta, columns)
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
    # v6.4 质量预警：低置信度/边缘证据不足/底边回退/边界截断逐条列出。
    stats_data["quality_warnings"] = _quality_warnings(columns, meta)
    stats_data["scale"] = scale

    # 整图分类横幅：来自平均轮廓的互斥分类与各柱计数。
    morph = stats_data.get("morphology") or {}
    cls = morph.get("classification") or {}
    banner = None
    if cls.get("label_cn"):
        banner = dict(cls)
        banner["classification_counts"] = morph.get("classification_counts")
        banner["thresholds"] = morph.get("thresholds")

    px_main = render_main_image(img, columns, 1.0, banner=banner)
    nm_main = render_main_image(img, columns, nm_per_px, banner=banner)

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

    export_boundaries(px_txt, columns, 1.0, detection_meta=meta)
    export_boundaries(nm_txt, columns, nm_per_px, detection_meta=meta)

    write_stats_file(stats_path, stats_data, detection_meta=meta)
    morphology_path = os.path.join(out_dir, f"{prefix}_morphology.json")
    morph_json = dict(stats_data["morphology"])
    morph_json["quality_warnings"] = stats_data.get("quality_warnings", [])
    with open(morphology_path, "w", encoding="utf-8") as f:
        json.dump(morph_json, f, ensure_ascii=False, indent=2, allow_nan=False)
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
               "quality_warnings": stats.get("quality_warnings", []),
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


def analyze_image(image_path):
    """低层一站式识别：只做"预处理 + 柱检测 + 比例尺"，不写过程图、不画图。

    返回 {"columns": [...], "nm_per_px": float, "scale": dict}。
    供批量脚本、验证脚本与 GUI 复用同一条识别链路（结果可 pickle 缓存）。
    """
    img, binary, aux = preprocess_image(image_path, on_stage=None)
    columns, _profile = _detect_columns(img, binary, aux, roi_mode=False, on_stage=None)
    scale = validate_scale(read_image_scale(image_path))
    return {"columns": columns, "nm_per_px": scale["nm"] / scale["pixels"], "scale": scale}


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
    """批量处理文件夹内所有图（供脚本/GUI 调用的便捷入口）。

    v6.2.1：按 INPUT_GLOBS 扫描（含 PNG）；失败图片逐个记录到
    BATCH_FAILURE_LOG（文件名 + 异常摘要），便于批量后排查。
    """
    tif_files = sorted({
        p
        for ext in INPUT_GLOBS
        for p in glob.glob(os.path.join(INPUT_DIR, ext))
    })

    ok = 0
    fail = 0
    failures = []

    for path in tif_files:
        try:
            process_image(path, on_image=on_image)
            ok += 1
        except Exception as exc:
            fail += 1
            failures.append((path, "%s: %s" % (type(exc).__name__, exc)))

    if failures and BATCH_FAILURE_LOG:
        try:
            log_path = os.path.join(INPUT_DIR, BATCH_FAILURE_LOG)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("[%s]\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                for path, msg in failures:
                    fh.write("FAIL %s | %s\n" % (os.path.basename(path), msg))
        except Exception:
            pass

    return ok, fail
