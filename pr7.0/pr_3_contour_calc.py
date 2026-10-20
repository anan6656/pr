# -*- coding: utf-8 -*-
"""pr_3_contour_calc -- 轮廓计算层：绘图、边界导出、统计与形貌量化。

从 pr6.1.py 拆分而来。依赖方向（单向）：
pr_1_config -> pr_2_image_recognition -> 本模块。
本模块不包含 GUI、Baseline 报告绘制（pr_4_contour_compare）
与图像处理管线（pr_2_image_recognition）。
"""

import math
import os
from datetime import datetime

import cv2
import numpy as np

from pr_1_config import (
    ALPHA,
    AXIS_MARGIN_B,
    AXIS_MARGIN_L,
    ENABLE_LER_OUTLIER_FLAG,
    LINE_THICKNESS,
    MORPH_MODE_BIN_MIN_NM,
    MORPH_MODE_BIN_RATIO,
    MORPH_MODE_FLAT_FACTOR,
    MORPH_NOISE_K,
    MORPH_SHAPE_CAP_NM,
    MORPH_SHAPE_CD_RATIO,
    MORPH_SHAPE_CD_RATIO_BY_KEY,
    MORPH_SHAPE_FLOOR_NM,
    MORPH_TOP_ROUNDING_CD_RATIO,
    MORPH_WAVE_AMP_FLOOR_NM,
    MORPH_WAVE_CAP_NM,
    MORPH_WAVE_FLOOR_NM,
    MORPH_WAVE_K,
    MORPHOLOGY_DEFINITION,
    MORPHOMETRY,
    MORPH_SHAPES,
    PANEL_W,
    PILLAR_POLARITY,
    PROFILE_SAMPLES,
    RESULT_BANNER_H,
    ROUGHNESS_KEYS,
    STAT_ROWS,
)



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

    axis_color, tick_color, text_color = (70, 70, 70), (120, 120, 120), (85, 85, 85)
    cv2.line(out, (x_axis, y_axis), (x_axis + w - 1, y_axis), axis_color, 1)
    cv2.line(out, (x_axis, 0), (x_axis, h - 1), axis_color, 1)

    for x in range(0, w, max(1, int(tick_step))):
        px = AXIS_MARGIN_L + x
        cv2.line(out, (px, y_axis), (px, y_axis + 6), tick_color, 1)
        cv2.putText(
            out, fmt(x),
            (px + 2, y_axis + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42, text_color, 1, cv2.LINE_AA
        )

    for y in range(0, h, max(1, int(tick_step))):
        cv2.line(out, (x_axis, y), (x_axis - 6, y), tick_color, 1)
        cv2.putText(
            out, fmt(y),
            (max(1, x_axis - 40), y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42, text_color, 1, cv2.LINE_AA
        )

    cv2.putText(
        out, unit,
        (max(1, AXIS_MARGIN_L + w - 40), y_axis + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, text_color, 1, cv2.LINE_AA
    )
    cv2.putText(
        out, unit,
        (2, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42, text_color, 1, cv2.LINE_AA
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


# ---- v6.3 中文文本与分类横幅（v6.4 起横幅蓝色加粗）----
_CN_FONT_CANDIDATES = ("msyh.ttc", "simhei.ttf", "simsun.ttc")
_CN_FONT_BOLD_CANDIDATES = ("msyhbd.ttc", "msyh.ttc", "simhei.ttf", "simsun.ttc")
_CN_FONT_CACHE = {}


def _cn_font_path(bold=False):
    """按候选链查找可用中文字体；结果缓存（含失败 None）。bold 优先粗体字库。"""
    key = "path_bold" if bold else "path"
    if key in _CN_FONT_CACHE:
        return _CN_FONT_CACHE[key]
    for name in (_CN_FONT_BOLD_CANDIDATES if bold else _CN_FONT_CANDIDATES):
        path = os.path.join("C:/Windows/Fonts", name)
        if os.path.isfile(path):
            _CN_FONT_CACHE[key] = path
            return path
    _CN_FONT_CACHE[key] = None
    return None


def draw_text_cn(img, text, org, font_size=22, color=(0, 0, 0), bg=None, bold=False):
    """原位绘制支持中文的文本。返回绘制区域 (x, y, w, h)；失败返回 None。

    PIL/中文字体可用时用 TrueType 渲染（bold=True 优先 msyhbd.ttc 粗体字库，
    缺失时以 stroke_width=1 伪粗兜底）；否则剔除非 ASCII 字符后回退
    cv2.putText（英文降级，不乱码）。绝不抛异常。
    """
    if img is None or not text:
        return None
    x0, y0 = int(org[0]), int(org[1])
    b, g, r = (int(color[0]), int(color[1]), int(color[2]))
    try:
        from PIL import Image, ImageDraw, ImageFont  # 懒加载，缺失时走降级
        font_file = _cn_font_path(bold=bold)
        if font_file:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype(font_file, int(font_size))
            # stroke_width 伪粗：字库本身是粗体（文件名含 bd）时不再加描边
            stroke = 1 if (bold and "bd" not in os.path.basename(font_file).lower()) else 0
            if bg is not None:
                bbox = draw.textbbox((x0, y0), text, font=font, stroke_width=stroke)
                draw.rectangle([bbox[0], bbox[1], bbox[2] + 2, bbox[3] + 2],
                               fill=(int(bg[2]), int(bg[1]), int(bg[0])))
            draw.text((x0, y0), text, font=font, fill=(r, g, b),
                      stroke_width=stroke, stroke_fill=(r, g, b))
            out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            img[:, :, :] = out
            bx = draw.textbbox((x0, y0), text, font=font, stroke_width=stroke)
            return (int(bx[0]), int(bx[1]), int(bx[2] - bx[0]), int(bx[3] - bx[1]))
    except Exception:
        pass
    # 降级：仅保留 ASCII
    ascii_text = "".join(ch for ch in text if 32 <= ord(ch) < 127)
    if not ascii_text:
        return None
    (tw, th), _ = cv2.getTextSize(ascii_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    if bg is not None:
        cv2.rectangle(img, (max(0, x0 - 3), max(0, y0 - th - 3)),
                      (min(img.shape[1] - 1, x0 + tw + 3), min(img.shape[0] - 1, y0 + 3)),
                      (int(bg[0]), int(bg[1]), int(bg[2])), -1)
    cv2.putText(img, ascii_text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (b, g, r), 2, cv2.LINE_AA)
    return (x0, y0 - th, tw, th + 3)


def draw_classification_banner(img, label_cn, counts=None, thresholds=None, rating=None):
    """返回顶部扩展了分类横幅条的新图；label_cn 为空则原样返回副本。

    v6.5：改为扩展画布——顶部新增 RESULT_BANNER_H 深蓝底横幅条，
    原图（含坐标轴与柱体）完整下移保留，不再覆盖任何原图像素。
    文字为白色加粗（粗体字库/伪粗描边）；draw_text_cn 完全失败时
    退化为 ASCII putText（★ 降级为 *），再失败则恢复原图尺寸。
    rating 为 1~3 时在分类名后追加同数目的 ★（星级）。
    """
    out = img.copy()
    if not label_cn:
        return out
    stars = ""
    try:
        stars = "★" * int(rating) if rating else ""
    except (TypeError, ValueError):
        stars = ""
    text = f"Profile: {label_cn}" + (f" {stars}" if stars else "")
    if counts:
        try:
            items = sorted(counts.items(), key=lambda kv: -kv[1])
            text += " · " + "、".join(f"{k}×{v}" for k, v in items if v)
        except Exception:
            pass
    font_size = 20  # v6.5.1: 22px 字 bbox 底 y≈36 越过 34px 条带压到原图首行, 降到 20
    pad = 6
    strip_h = int(RESULT_BANNER_H)
    extended = np.full(
        (img.shape[0] + strip_h, img.shape[1], 3), (205, 90, 26), dtype=np.uint8)
    extended[strip_h:, :] = img

    def _paint(message, size=None):
        return draw_text_cn(extended, message, (pad, 5), font_size=size or font_size,
                            color=(255, 255, 255), bold=True)

    # v7.0.1 横幅自适应：文字超宽时先逐步缩小字号，仍超宽才去掉计数后缀，
    # 保证窄图上分类与星级完整可见（此前直接截断，出现"Norma2"这类残缺）。
    max_w = extended.shape[1] - 2 * pad
    box = _paint(text)
    if box is not None and box[2] > max_w:
        for size in (18, 16, 14, 12, 11):
            box = _paint(text, size)
            if box is None or box[2] <= max_w:
                break
    if box is not None and box[2] > max_w:
        # 仍超宽：去掉计数后缀重画，保底显示分类+星级。
        box = _paint(f"Profile: {label_cn}" + (f" {stars}" if stars else ""))
    cv2.line(extended, (0, strip_h - 1), (extended.shape[1] - 1, strip_h - 1),
             (180, 180, 180), 1)
    if box is None:
        # draw_text_cn 失败（无中文字库）→ ASCII 降级，仍扩展画布
        try:
            ascii_text = text.replace("★", "*").encode("ascii", "replace").decode("ascii")
            cv2.putText(extended, ascii_text, (pad, int(font_size) + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2,
                        cv2.LINE_AA)
        except Exception:
            return out
    return extended


def render_main_image(img, columns, nm_per_px=1.0, banner=None):
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
            0.62, color, 2, cv2.LINE_AA
        )

        last = col["ranges"][-1]
        for xx in range(int(round(last[1])) + xoff, int(round(last[2])) + xoff + 1, 6):
            cv2.line(
                main,
                (xx, bot),
                (int(min(xx + 3, round(last[2]) + xoff)), bot),
                color, 1
            )

    if banner:
        main = draw_classification_banner(
            main,
            banner.get("label_cn") if isinstance(banner, dict) else banner,
            banner.get("classification_counts") if isinstance(banner, dict) else None,
            banner.get("thresholds") if isinstance(banner, dict) else None,
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

    out = np.full((mh + profile_h, mw + pw, 3), 250, dtype=np.uint8)
    out[:mh, :mw] = main_img
    out[mh:mh + profile_h, AXIS_MARGIN_L:AXIS_MARGIN_L + w] = x_profile
    # v6.5：横幅扩展画布使主图内容整体下移 RESULT_BANNER_H，
    # 右侧逐柱面板与主图柱体对齐需同样下移。
    out[RESULT_BANNER_H:RESULT_BANNER_H + h, mw:mw + pw] = y_profile

    # v6.6 顶栏贯通：分类横幅只画在主图宽度上，右侧逐柱面板顶部会留白块，
    # 视觉上像"上方缺一块"。取主图顶栏颜色补齐右段，并在其上标注面板含义。
    strip_h = int(RESULT_BANNER_H)
    if strip_h > 0 and main_img.shape[0] > strip_h:
        top_color = main_img[strip_h // 2, min(mw - 1, main_img.shape[1] - 1)]
        if not bool(np.all(top_color > 240)):  # 非白 → 主图带分类横幅
            out[:strip_h, mw:mw + pw] = top_color
            draw_text_cn(out, "逐柱 y 剖面", (mw + 10, 6), font_size=17,
                         color=(255, 255, 255), bold=True)
            # 与主图横幅底边同高的分隔线，右侧面板段补齐
            cv2.line(out, (mw, strip_h - 1), (out.shape[1] - 1, strip_h - 1),
                     (180, 180, 180), 1)

    # 底部剖面区右侧留白：填浅底并标注单位，避免与剖面图之间出现断层。
    out[mh:mh + profile_h, AXIS_MARGIN_L + w:] = 250
    out[mh:mh + profile_h, :AXIS_MARGIN_L] = 250

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




def _finite_number(value):
    return isinstance(value, (int, float, np.number)) and math.isfinite(float(value))


def _detrended(values, coordinate, degree=1):
    values, coordinate = np.asarray(values, float), np.asarray(coordinate, float)
    centered = coordinate - np.mean(coordinate)
    return values - np.polyval(np.polyfit(centered, values, degree), centered)


def _mode_width_nm(widths):
    """逐柱 CD 众数：直方图最高 bin 中心；分布退化时回退中位数。

    返回 (cd_mode, method)；method ∈ {mode, single, median_few, median_flat}。
    """
    widths = np.asarray(widths, dtype=float)
    widths = widths[np.isfinite(widths) & (widths > 0)]
    if widths.size == 0:
        return None, "none"
    if widths.size < 5:
        return float(np.median(widths)), "median_few"
    if float(np.ptp(widths)) <= 1e-9:
        return float(widths[0]), "single"
    span = float(np.percentile(widths, 95) - np.percentile(widths, 10))
    bin_w = max(MORPH_MODE_BIN_MIN_NM, MORPH_MODE_BIN_RATIO * span)
    lo, hi = float(np.min(widths)), float(np.max(widths))
    n_bins = max(1, int(math.ceil((hi - lo) / bin_w - 1e-9)) + 1)
    counts, edges = np.histogram(widths, bins=n_bins, range=(lo, hi))
    peak = int(np.argmax(counts))
    median_count = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 0.0
    if counts[peak] < MORPH_MODE_FLAT_FACTOR * median_count:
        return float(np.median(widths)), "median_flat"
    return float(0.5 * (edges[peak] + edges[peak + 1])), "mode"


def _row_noise_sigma_nm(y, width):
    """单柱噪声底：宽度相邻行差分 std/√2，剔除断行缺口对（Δy 过大）。"""
    y = np.asarray(y, dtype=float)
    width = np.asarray(width, dtype=float)
    if y.size < 4:
        return None
    steps = np.diff(y)
    step_med = float(np.median(steps))
    if not math.isfinite(step_med) or step_med <= 0:
        return None
    keep = steps <= 1.5 * step_med
    if int(np.count_nonzero(keep)) < 3:
        return None
    dw = np.diff(width)[keep]
    dw = dw[np.isfinite(dw)]
    if dw.size < 3:
        return None
    sigma = float(np.std(dw)) / math.sqrt(2.0)
    return sigma if math.isfinite(sigma) and sigma > 0 else None


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




def contour_morphology(y, left, right, valid_left=None, valid_right=None, cd_mode=None):
    """从物理坐标计算数值；用于逐柱原生采样和平均轮廓，避免两套公式。

    六类几何形貌以众数 CD 为水平基准线：cd_mode 为 None 时内部自算（单柱路径），
    平均轮廓/方波路径可显式传入以统一口径。
    """
    out = {key: None for key, _, _ in MORPHOMETRY}
    y, left, right = (np.asarray(v, dtype=float) for v in (y, left, right))
    if len(y) < 8 or y[-1] <= y[0] or not np.all(np.isfinite(np.r_[y, left, right])):
        return out
    height = float(y[-1] - y[0])
    z = (y - y[0]) / height
    width = right - left
    out.update(height_nm=height, cd_mean_nm=float(np.mean(width)))

    cd_mode_value, cd_mode_method = _mode_width_nm(width)
    if _finite_number(cd_mode) and cd_mode > 0:
        cd_mode_value, cd_mode_method = float(cd_mode), "given"
    if cd_mode_value is not None and cd_mode_value > 0:
        out["cd_mode_nm"] = float(cd_mode_value)
        out["cd_mode_method"] = cd_mode_method

    def positive_q90(values):
        return max(0.0, float(np.percentile(values, 90))) if len(values) else None

    base = out["cd_mode_nm"]
    if base is not None and base > 0:
        # 物理范围守卫：宽度 <0.5×CD 或 >1.5×CD 的行为定位坏行（顶部首行/底部延展伪影），
        # 不参与六类量值统计；正常圆角收缩/展宽不会超过该范围。
        sane = (width > 0.5 * base) & (width < 1.5 * base)
        # 区间中位守卫：与区间内中位宽偏离 >0.35×CD 的行（如底部骤窄）同样剔除，
        # 防止少量坏行通过 Q90 虚增 undercut/footing/top_deficit。
        def zone_sane(mask):
            if not np.any(mask):
                return mask
            med = float(np.median(width[mask]))
            return mask & (np.abs(width - med) <= 0.35 * base)
        top = zone_sane((z <= .18) & sane)
        bottom = zone_sane((z >= .82) & sane)
        middle = zone_sane((z >= .35) & (z <= .65) & sane)
        cap = zone_sane((z <= .20) & sane)
        out["t_top_nm"] = positive_q90((width - base)[top])
        out["footing_nm"] = positive_q90((width - base)[bottom])
        out["undercut_nm"] = positive_q90((base - width)[bottom])
        out["middle_thick_nm"] = positive_q90((width - base)[middle])
        out["middle_thin_nm"] = positive_q90((base - width)[middle])
        out["top_deficit_nm"] = positive_q90((base - width)[cap])
        cap_deficit = np.maximum(0.0, base - width[cap])
        _trap = getattr(np, "trapezoid", None) or np.trapz
        area = float(_trap(cap_deficit, y[cap]))
        r_equiv = math.sqrt(max(0.0, area) / (2.0 * (1.0 - np.pi / 4.0)))
        # v6.4 锥度扣除：面积式仅对"垂直侧壁+圆顶角"精确，梯形侧壁会把
        # 整体锥度亏缺误计入圆角（虚增 r≈0.305·H·√k）。以中段 sane 行
        # 线性拟合锥度线 T(y) 外推到 cap 区，仅残差亏缺进换算。y 向下增大，
        # a>0（向下变宽=顶部收窄）与 a<0（向下变窄）都是物理锥形，残差
        # max(0, T−w) 双向有效；仅拟合失败时回退未扣锥度值。
        r_taper, use_taper = r_equiv, False
        cap_depth_side = None  # 顶部圆角亏缺深度（单侧，相对锥度线）
        if int(np.count_nonzero(middle)) >= 4 and int(np.count_nonzero(cap)) >= 2:
            ym, wm = y[middle], width[middle]
            med_w = float(np.median(wm))
            mad_w = float(np.median(np.abs(wm - med_w)))
            keep = np.abs(wm - med_w) <= 3.0 * (1.4826 * mad_w + 1.0)
            if int(np.count_nonzero(keep)) >= 4:
                try:
                    a, b = np.polyfit(ym[keep], wm[keep], 1)
                except Exception:
                    a, b = None, None
                if a is not None and np.isfinite(a) and np.isfinite(b):
                    # v6.5 锥度线封顶：中段远宽于帽区时（自然圆帽/窄帽柱），
                    # T(y) 外推会远超帽区观察宽度，把整个帽收窄误计为圆角。
                    # 以帽区最大观察宽 W_cap_max 为上限截断 T(y)，仅超出
                    # 帽区实际宽度的部分进残差。
                    t_line = a * y[cap] + b
                    w_cap_max = float(np.max(width[cap]))
                    t_line = np.minimum(t_line, w_cap_max)
                    resid = np.maximum(0.0, t_line - width[cap])
                    area_t = float(_trap(resid, y[cap]))
                    r_taper = math.sqrt(max(0.0, area_t) / (2.0 * (1.0 - np.pi / 4.0)))
                    use_taper = True
                    # v6.6：主量改为"顶部圆角亏缺深度（单侧）"。半径由面积
                    # 开方换算，会把 2nm 的顶部亏缺放大到 5nm 量级（√(A/0.43)），
                    # 与肉眼看到的"顶部圆掉多少"不符；深度口径直接对应
                    # 单侧壁相对锥度线收缩的 nm 数，量值与 t_top/footing
                    # 等其余五类同量级可比。
                    cap_depth_side = 0.5 * positive_q90(resid)
        # v6.5 圆帽上界：真圆角亏缺深度 max=2r 且底部归零，面积式 r 与其
        # 自洽；自然圆帽（平台+线性展宽）亏缺呈斜坡，面积式会虚增 r。
        # 以亏缺深度上界 deficit_max/2 封顶，对真圆角无影响，对斜坡亏缺
        # 消除虚增。
        deficit_max = float(np.max(cap_deficit)) if np.any(cap_deficit) else 0.0
        r_bound = 0.5 * deficit_max
        out["top_rounding_r_nm"] = min(r_taper if use_taper else r_equiv, r_bound)
        # 主量 = 顶部圆角亏缺深度（单侧）；锥度拟合不可用时回退众数 CD 基准。
        out["top_rounding_nm"] = (cap_depth_side if cap_depth_side is not None
                                  else 0.5 * float(out["top_deficit_nm"] or 0.0))
        # 相对口径：深度占众数 CD 的百分比（跨 CD 样本可比；正常圆顶工艺柱
        # 该比例稳定在低两位数百分比内）。
        if out["top_rounding_nm"] is not None and base:
            out["top_rounding_pct"] = 100.0 * float(out["top_rounding_nm"]) / float(base)
    out["noise_sigma_nm"] = _row_noise_sigma_nm(y, width)
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
    return {key: (0.0 if isinstance(value, float) and abs(value) < 1e-6 else value)
            for key, value in out.items()}


def morphology_labels(metrics):
    """兼容视图：六类几何形貌量值 ≥ 形状阈值即打标签，'+' 连接（可同时存在）。"""
    thr = MORPH_SHAPE_FLOOR_NM
    return "+".join(label for key, label in MORPH_SHAPES
                    if _finite_number(metrics.get(key)) and metrics[key] >= thr) or "低于阈值"


def _profile_rating(margin, eff_thr):
    """主标签星级：量值 ≥3×生效阈值 ★★★、≥2× ★★、≥1× ★，否则 0。"""
    if not (_finite_number(margin) and _finite_number(eff_thr) and eff_thr > 0):
        return 0
    if margin >= 3.0 * eff_thr:
        return 3
    if margin >= 2.0 * eff_thr:
        return 2
    if margin >= eff_thr:
        return 1
    return 0


def classify_profile_multi(metrics, shape_thr=None, wave_thr=None):
    """多标签 + 星级分类（v6.5）。

    标签规则：驻波最高优先（roughness ≥ wave_thr 且 FFT 周期幅值显著），
    驻波为主标签时六类最大者若也超生效阈值则作第二标签；无驻波时六类
    量值降序取 ≥ 生效阈值的前两个（最多同时两个 profile），0 个 → Normal。
    生效阈值 = max(shape_thr, MORPH_SHAPE_CD_RATIO × CD)，与单标签口径一致。
    星级只评主标签量值 margin：≥3×阈值 ★★★、≥2× ★★、≥1× ★，否则 0。
    返回 dict：label/label_cn/margin/wave/threshold（与单标签字段一致）+
    labels/labels_cn（多标签列表）、rating（0~3）、eff_thr、sorted_shapes
    （六类量值降序，报告与调参用）。无法评估时 label=None、labels=[]。
    """
    shape_thr = float(MORPH_SHAPE_FLOOR_NM if shape_thr is None else shape_thr)
    wave_thr = float(MORPH_WAVE_FLOOR_NM if wave_thr is None else wave_thr)
    ler = metrics.get("ler_nm")
    if not _finite_number(ler):
        sides = [metrics.get("ler_left_nm"), metrics.get("ler_right_nm")]
        sides = [v for v in sides if _finite_number(v)]
        ler = float(np.mean(sides)) if sides else None
    lwr = metrics.get("lwr_nm")
    wave_amp = metrics.get("wave_amplitude_nm")
    wave_significant = _finite_number(wave_amp) and wave_amp >= MORPH_WAVE_AMP_FLOOR_NM
    roughness = None
    if _finite_number(ler) and _finite_number(lwr):
        roughness = max(float(ler), float(lwr))
    elif _finite_number(ler):
        roughness = float(ler)
    elif _finite_number(lwr):
        roughness = float(lwr)
    candidates = [(float(metrics[key]), key, label) for key, label in MORPH_SHAPES
                  if _finite_number(metrics.get(key))]
    if not candidates:
        return {"label": None, "label_cn": "无法评估", "margin": None, "wave": False,
                "labels": [], "labels_cn": [], "rating": 0, "eff_thr": None,
                "sorted_shapes": []}
    cd = metrics.get("cd_mode_nm")
    if not _finite_number(cd) or cd <= 0:
        cd = metrics.get("cd_mean_nm")
    if _finite_number(cd) and cd > 0:
        eff_thr = max(shape_thr, MORPH_SHAPE_CD_RATIO * float(cd))
    else:
        eff_thr = shape_thr
    eff_thr = float(eff_thr)
    sorted_shapes = sorted(candidates, key=lambda item: -item[0])
    suffix = "_nm"

    def _shape_key(raw_key):
        return raw_key[:-len(suffix)] if raw_key.endswith(suffix) else raw_key

    is_wave = roughness is not None and roughness >= wave_thr and wave_significant
    if is_wave:
        labels = ["standing_wave"]
        labels_cn = ["驻波"]
        # 驻波主标签不参与星级；第二标签取六类最大者（超阈值才作标签）。
        second_rating = 0
        if sorted_shapes and sorted_shapes[0][0] >= eff_thr:
            labels.append(_shape_key(sorted_shapes[0][1]))
            labels_cn.append(sorted_shapes[0][2])
            second_rating = 1
        return {"label": "standing_wave", "label_cn": "驻波",
                "margin": roughness, "wave": True, "threshold": float(wave_thr),
                "labels": labels, "labels_cn": labels_cn,
                "rating": second_rating,
                "eff_thr": eff_thr, "sorted_shapes": sorted_shapes}
    labels = []
    for value, key, _ in sorted_shapes:
        k = _shape_key(key)
        # 分类别 CD 比例门槛：正常圆顶工艺柱的圆化、正常柱底部的渐变展宽
        # 都随 CD 等比缩放，属正常柱形而非缺陷，用高于通用比例的门过滤除。
        thr_k = eff_thr
        ratio_k = MORPH_SHAPE_CD_RATIO_BY_KEY.get(k)
        if ratio_k and _finite_number(cd) and cd > 0:
            thr_k = max(thr_k, float(ratio_k) * float(cd))
        if value >= thr_k:
            labels.append(k)
        if len(labels) >= 2:
            break
    cn_by_key = {_shape_key(key): label for _, key, label in sorted_shapes}
    labels_cn = [cn_by_key[key] for key in labels]
    if not labels:
        return {"label": "normal", "label_cn": "Normal", "margin": float(sorted_shapes[0][0]),
                "wave": False, "threshold": eff_thr,
                "labels": [], "labels_cn": [], "rating": 0,
                "eff_thr": eff_thr, "sorted_shapes": sorted_shapes}
    margin, key, label = sorted_shapes[0]
    return {"label": _shape_key(key), "label_cn": label, "margin": float(margin),
            "wave": False, "threshold": eff_thr,
            "labels": labels, "labels_cn": labels_cn,
            "rating": _profile_rating(margin, eff_thr),
            "eff_thr": eff_thr, "sorted_shapes": sorted_shapes}


def classify_profile(metrics, shape_thr=None, wave_thr=None):
    """互斥单标签分类（v6.5 起为 classify_profile_multi 的兼容包装）。

    驻波(LER/LWR 超阈值且 FFT 周期幅值显著)最高优先，其次六类量值取
    最大者，否则 Normal。驻波需周期证据（wave_amplitude ≥
    MORPH_WAVE_AMP_FLOOR_NM）：防止单包络收缩/鼓起等低频形貌因去线性
    趋势残差偏大被误判为驻波。形状阈值 CD 归一化——生效阈值 =
    max(σ自适应阈值, MORPH_SHAPE_CD_RATIO×CD)，CD 取 cd_mode_nm（回退
    cd_mean_nm）。返回 {"label", "label_cn", "margin", "wave",
    "threshold"}；无法评估时 label=None。多标签与星级见
    classify_profile_multi。
    """
    result = classify_profile_multi(metrics, shape_thr, wave_thr)
    return {"label": result["label"], "label_cn": result["label_cn"],
            "margin": result["margin"], "wave": result["wave"],
            "threshold": result.get("threshold")}


def build_morphology(columns, nm_per_px):
    if not _finite_number(nm_per_px) or nm_per_px <= 0:
        raise ValueError("比例尺 nm/px 必须为正数")
    per_column, aligned_left, aligned_right, heights = [], [], [], []
    eligible_widths, noise_sigmas = [], []
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
        if eligible and _finite_number(metrics.get("noise_sigma_nm")) and metrics["noise_sigma_nm"] > 0:
            noise_sigmas.append(float(metrics["noise_sigma_nm"]))
        record = {"label": f"PR{col['label']}", "metrics": metrics, "included": eligible,
                  "reason": "图像/ROI 边界截断" if clipped else ("边缘证据不足" if not eligible else ""),
                  "top_source": col.get("top_source"), "bottom_source": col.get("bottom_source")}
        per_column.append(record)
        if eligible:
            # 平均路径同样剔除定位坏行（与 contour_morphology 守卫同口径），
            # 防止个别柱的顶部首行/底部骤窄行污染平均轮廓。
            cd_mode_col = metrics.get("cd_mode_nm")
            if _finite_number(cd_mode_col) and cd_mode_col > 0:
                w_col = right - left
                keep = (w_col > 0.5 * cd_mode_col) & (w_col < 1.5 * cd_mode_col)
                if keep.sum() >= 8:
                    left, right, z = left[keep], right[keep], z[keep]
            aligned_left.append(np.interp(z_grid, z, left))
            aligned_right.append(np.interp(z_grid, z, right))
            heights.append(float(y[-1]))
            eligible_widths.extend((rr[:, 2] - rr[:, 1]) * nm_per_px)
    # 全图 CD 众数先算（v6.5：分类阈值含 CD 归一化项，需在此前可得）。
    global_cd_mode, global_cd_method = (None, "none")
    if eligible_widths:
        global_cd_mode, global_cd_method = _mode_width_nm(eligible_widths)
    # σ_noise 由图像识别结果自适应：纳入柱单柱噪声的中位数；无纳入柱时阈值取下限。
    # v6.4：σ 自适应阈值加上限（防高噪声图阈值过高漏报真实形貌）。
    noise_sigma = float(np.median(noise_sigmas)) if noise_sigmas else None
    if _finite_number(noise_sigma) and noise_sigma > 0:
        shape_thr = max(MORPH_SHAPE_FLOOR_NM,
                        min(MORPH_NOISE_K * noise_sigma, MORPH_SHAPE_CAP_NM))
        wave_thr = max(MORPH_WAVE_FLOOR_NM,
                       min(MORPH_WAVE_K * noise_sigma, MORPH_WAVE_CAP_NM))
    else:
        shape_thr, wave_thr = MORPH_SHAPE_FLOOR_NM, MORPH_WAVE_FLOOR_NM
    thresholds = {"shape": float(shape_thr), "wave": float(wave_thr),
                  "shape_cd_ratio": float(MORPH_SHAPE_CD_RATIO),
                  "cd_mode": float(global_cd_mode) if global_cd_mode else None}
    for record in per_column:
        record["classification"] = classify_profile(record["metrics"], shape_thr, wave_thr)
    summary = {}
    included = [c for c in per_column if c["included"]]
    shape_keys = {key for key, _ in MORPH_SHAPES}
    for key, _, _ in MORPHOMETRY:
        values = [c["metrics"][key] for c in included if _finite_number(c["metrics"].get(key))]
        threshold = shape_thr if key in shape_keys else None
        summary[key] = {"mean": float(np.mean(values)) if values else None,
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "min": min(values) if values else None, "max": max(values) if values else None,
                        "n": len(values), "threshold": threshold,
                        "above_1nm": sum(v >= threshold for v in values) if threshold is not None else None}
    profile, average_metrics = None, {key: None for key, _, _ in MORPHOMETRY}
    if heights:
        avg_y = z_grid * float(np.mean(heights))
        avg_l, avg_r = np.mean(aligned_left, axis=0), np.mean(aligned_right, axis=0)
        # 平均轮廓的形貌基准用其自身宽度的众数（401 点分布的众数），而非全图合并众数：
        # 柱间宽度差异会平移平均宽度，若以全图众数为基准会把柱间差异误计为形貌偏差。
        average_metrics = contour_morphology(avg_y, avg_l, avg_r)
        profile = {"z": z_grid.tolist(), "y_nm": avg_y.tolist(), "left_nm": avg_l.tolist(),
                   "right_nm": avg_r.tolist(), "width_nm": (avg_r - avg_l).tolist(),
                   "left_std_nm": np.std(aligned_left, axis=0).tolist(),
                   "right_std_nm": np.std(aligned_right, axis=0).tolist()}
    metrics = dict(average_metrics)
    for key in ROUGHNESS_KEYS:
        metrics[key] = summary[key]["mean"]
    if global_cd_mode is not None:
        metrics["cd_mode_nm"] = float(global_cd_mode)
        metrics["cd_mode_method"] = global_cd_method
    # v6.5：整图分类改多标签+星级；labels 取超阈值量值前二，rating 按主标签
    # 量值相对生效阈值的倍数评定。单标签五字段由 multi 结果保持兼容。
    classification = classify_profile_multi(metrics, shape_thr, wave_thr)
    counts = {}
    for record in included:
        label_cn = record["classification"]["label_cn"]
        counts[label_cn] = counts.get(label_cn, 0) + 1
    method_counts = {}
    for record in per_column:
        method = record["metrics"].get("cd_mode_method") or "none"
        method_counts[method] = method_counts.get(method, 0) + 1
    if ENABLE_LER_OUTLIER_FLAG and len(included) >= 10:
        ler_values = [c["metrics"]["ler_nm"] for c in included if _finite_number(c["metrics"].get("ler_nm"))]
        if len(ler_values) >= 5:
            med = float(np.median(ler_values))
            mad = float(np.median(np.abs(np.asarray(ler_values) - med))) or 1e-9
            for record in per_column:
                value = record["metrics"].get("ler_nm")
                if _finite_number(value) and value > med + 3.5 * mad:
                    record.setdefault("flags", []).append("疑似驻波离群")
    return {"metrics": metrics, "mean_profile_metrics": average_metrics, "summary": summary,
            "mean_profile": profile, "per_column": per_column, "included_count": len(included),
            "excluded_count": len(per_column) - len(included), "nm_per_px": float(nm_per_px),
            "cd_mode_nm": float(global_cd_mode) if global_cd_mode is not None else None,
            "noise_sigma_nm": noise_sigma, "thresholds": thresholds,
            "classification": classification, "classification_counts": counts,
            "cd_mode_method_counts": method_counts,
            "definition": MORPHOLOGY_DEFINITION}




def square_baseline(cd_nm, height_nm):
    if not all(_finite_number(v) and v > 0 for v in (cd_nm, height_nm)):
        raise ValueError("标准方波的 CD 和高度必须是正数（nm）")
    y = np.linspace(0.0, height_nm, PROFILE_SAMPLES)
    col = {"label": 1, "ranges": list(zip(y, np.full(len(y), -cd_nm / 2), np.full(len(y), cd_nm / 2)))}
    result = build_morphology([col], 1.0)
    result.update(name=f"方波 CD={cd_nm:g}nm H={height_nm:g}nm", kind="square", nm_per_px=None)
    return result


def _profile_entry(sample):
    """提取单个样本的互斥形貌分类与六类量化值（v6.3 baseline 量化对比用）。"""
    cls = sample.get("classification") or {}
    metrics = sample.get("metrics") or {}
    return {"sample_id": sample.get("sample_id", ""),
            "name": sample.get("name", ""),
            "kind": sample.get("kind", "image"),
            "classification": {"label": cls.get("label"),
                               "label_cn": cls.get("label_cn") or "无法评估",
                               "margin": cls.get("margin"),
                               "wave": cls.get("wave")},
            "classification_counts": dict(sample.get("classification_counts") or {}),
            "thresholds": dict(sample.get("thresholds") or {}),
            "noise_sigma_nm": sample.get("noise_sigma_nm"),
            "shape_metrics": {key: metrics.get(key) for key, _ in MORPH_SHAPES}}


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
            "profiles": {"baseline": _profile_entry(baseline),
                         "samples": [_profile_entry(s) for s in samples]},
            "definition": MORPHOLOGY_DEFINITION, "created_at": datetime.now().isoformat()}




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
    from pr_2_image_recognition import _column_width_px
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
    from pr_2_image_recognition import _observed_cd_range, _reported_cd_range
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
    classifications = [c["classification"]["label_cn"] for c in morph["per_column"]]
    rows.append(["形貌分类（互斥）", morph["classification"]["label_cn"], *classifications])
    # v6.5 整图多标签+星级：labels_cn 以 + 连接，主标签按量值/生效阈值倍数评星。
    cls_multi = morph.get("classification") or {}
    labels_cn = cls_multi.get("labels_cn") or []
    rating = cls_multi.get("rating") or 0
    multi_txt = "+".join(labels_cn) if labels_cn else (cls_multi.get("label_cn") or "无法评估")
    if rating:
        multi_txt += " " + "★" * rating
    rows.append(["形貌分类（多标签）", multi_txt])
    counts = morph.get("classification_counts") or {}
    if counts:
        rows.append(["分类计数（纳入柱）",
                     "；".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))])
    thresholds = morph.get("thresholds") or {}
    if thresholds:
        sigma = morph.get("noise_sigma_nm")
        sigma_txt = f"{sigma:.3f}" if _finite_number(sigma) else "无"
        cd_txt = thresholds.get("cd_mode")
        cd_part = f"；生效阈值=max(形状, {thresholds.get('shape_cd_ratio', 0.1):.0%}×CD={cd_txt:.2f})" if cd_txt else ""
        rows.append(["分类阈值 (nm)",
                     f"形状≥{thresholds.get('shape', 0):.2f} / 驻波LER·LWR≥{thresholds.get('wave', 0):.2f}（σ_noise={sigma_txt}）{cd_part}"])
    if data.get("scale"):
        sc = data["scale"]
        rows.append(["本图比例尺", f"{sc['pixels']:g} px = {sc['nm']:g} nm；1 px = {sc['nm']/sc['pixels']:.8g} nm"])
    warns = data.get("quality_warnings")
    if warns is None and isinstance(data.get("detection"), dict):
        warns = (data["detection"] or {}).get("quality_warnings")
    rows.append(["质量预警", "；".join(warns) if warns else "无"])
    rows.extend(_detection_meta_rows(detection_meta or data.get("detection")))
    write_aligned_table(out_txt_path, rows, left_first_text=True, footer=MORPHOLOGY_DEFINITION)
