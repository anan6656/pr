# -*- coding: utf-8 -*-
"""pr_4_contour_compare -- Baseline 对比层：形貌对比图绘制与报告导出。

从 pr6.1.py 拆分而来。依赖方向（单向）：
pr_1_config -> pr_3_contour_calc -> 本模块。
计算部分（square_baseline/compare_morphologies/build_morphology）
在 pr_3_contour_calc；本模块只负责可视化与导出。
"""

import csv
import json
import math
import os
from datetime import datetime

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from pr_1_config import (
    APP_VERSION,
    FORMULA_HELP,
    MATPLOTLIB_AVAILABLE,
    MORPHOMETRY,
    MORPH_SHAPES,
    PROFILE_FORMULA_HELP,
    application_dir,
)
from pr_3_contour_calc import _finite_number


def _sample_caption(sample, limit=25, with_class=False):
    name = sample["name"].replace("\n", " ")
    if len(name) > limit:
        name = name[:limit-1] + "…"
    text = f"{sample['sample_id']} · {name}"
    if with_class:
        cls = sample.get("classification") or {}
        if sample.get("kind") == "square":
            text += " [方波基准]"
        elif cls.get("label_cn"):
            text += f" [{cls['label_cn']}]"
    return text





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
        label = _sample_caption(sample, 23 if len(samples) <= 8 else 12, with_class=True)
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


def _profile_entry_from_sample(sample):
    """无 profiles 键的旧报告兜底：从样本 dict 直接提取。"""
    from pr_3_contour_calc import _profile_entry
    return _profile_entry(sample)


def _sample_caption_by_entry(entry, limit=25):
    name = (entry.get("name") or "").replace("\n", " ")
    if len(name) > limit:
        name = name[:limit-1] + "…"
    return f"{entry.get('sample_id', '')} · {name}"


def draw_profile_class_table(report, figure=None):
    """形貌量化对照表：行 = 六类形貌量值 + 互斥分类，列 = B0、S1…Sn。

    量值单元格显示样本值与 Δ=样本−基准（增大红 / 减小蓝）；
    最后一行给出每样本互斥分类与一致性结论。
    """
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("形貌量化对照表需要 Matplotlib")
    profiles = report.get("profiles") or {}
    base = profiles.get("baseline") or _profile_entry_from_sample(report["baseline"])
    entries = profiles.get("samples")
    if entries is None:
        entries = [_profile_entry_from_sample(s) for s in report["samples"]]
    fig = figure or Figure(figsize=(4.0 + 2.6 * (len(entries) + 1),
                                    1.2 + 0.62 * (len(MORPH_SHAPES) + 2)), dpi=150)
    fig.clear()
    fig.set_facecolor("#ffffff")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ffffff")
    ax.axis("off")

    shape_keys = [key for key, _ in MORPH_SHAPES]
    shape_titles = [title for _, title in MORPH_SHAPES]
    headers = ["指标 (nm)"] + ["B0 基准"] + [_sample_caption_by_entry(e, 22) for e in entries]
    row_titles = shape_titles + ["互斥分类"]
    n_rows = len(row_titles)
    # 行标题直接作为第一列单元格，避免 rowLabels 与网格错位。
    cell_text = [[title, ""] + [""] * len(entries) for title in row_titles]
    table = ax.table(cellText=cell_text, colLabels=headers,
                     cellLoc="center", rowLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    col_width = 0.9 / (len(entries) + 3.4)
    for (r, c), cell in table.get_celld().items():
        if c == 0:
            cell.set_width(col_width * 1.4)
            cell.set_text_props(fontweight="bold" if r == -1 else "normal",
                                color="#20334a")
        else:
            cell.set_width(col_width * 3.0)

    def fmt(v):
        return f"{v:.2f}" if _finite_number(v) else "—"

    base_vals = base.get("shape_metrics") or {}
    for j, key in enumerate(shape_keys):
        ref = base_vals.get(key)
        table[j + 1, 1].get_text().set_text(fmt(ref))
        table[j + 1, 1].set_text_props(color="#26364a")
        for i, entry in enumerate(entries):
            value = (entry.get("shape_metrics") or {}).get(key)
            delta = (value - ref) if (_finite_number(value) and _finite_number(ref)) else None
            txt = fmt(value)
            if delta is not None:
                txt += f"\nΔ {delta:+.2f}"
            cell = table[j + 1, i + 2]
            cell.get_text().set_text(txt)
            if delta is not None:
                if delta > 1e-6:
                    cell.get_text().set_color("#b91c1c")
                elif delta < -1e-6:
                    cell.get_text().set_color("#1d4ed8")
                else:
                    cell.get_text().set_color("#475569")
            else:
                cell.get_text().set_color("#64748b")
    base_label = "方波基准" if base.get("kind") == "square" else (base.get("classification") or {}).get("label_cn") or "—"
    table[n_rows, 1].get_text().set_text(base_label)
    table[n_rows, 1].set_text_props(fontweight="bold")
    for i, entry in enumerate(entries):
        label = (entry.get("classification") or {}).get("label_cn") or "—"
        if label == base_label:
            concl = f"{label}\n与基准一致"
        else:
            concl = f"{label}\n{base_label} → {label}"
        cell = table[n_rows, i + 2]
        cell.get_text().set_text(concl)
        cell.set_text_props(fontweight="bold",
                            color="#15803d" if label == base_label else "#b45309")
    ax.set_title("形貌量化对照 · 六类量值 (nm) 与互斥分类",
                 fontsize=11, fontweight="bold", color="#20334a", pad=12)
    fig.text(.5, .015, "Δ=样本−基准；红色增加 / 蓝色减少。互斥分类基于各图自适应阈值（见 profile_classes.csv）。",
             ha="center", fontsize=7.5, color="#64748b")
    fig.tight_layout()
    return fig


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

    def _class_label(s):
        if s.get("kind") == "square":
            return "方波基准"
        cls = s.get("classification") or {}
        labels_cn = cls.get("labels_cn")
        if isinstance(labels_cn, list) and labels_cn:
            return "+".join(str(v) for v in labels_cn)
        return cls.get("label_cn") or ""

    def _class_labels(s):
        if s.get("kind") == "square":
            return ""
        cls = s.get("classification") or {}
        labels = cls.get("labels")
        return "+".join(str(v) for v in labels) if isinstance(labels, list) and labels else ""

    def _class_rating(s):
        if s.get("kind") == "square":
            return ""
        rating = (s.get("classification") or {}).get("rating")
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            return ""
        return "★" * rating if rating > 0 else "0"

    def _class_counts(s):
        counts = s.get("classification_counts") or {}
        return "；".join(f"{k}×{v}" for k, v in counts.items())

    def _quality_join(s):
        warns = s.get("quality_warnings") or []
        return "；".join(str(w) for w in warns)

    write_rows("index", "sample_index.csv", ([s["sample_id"], s["name"], s.get("source", "标准方波"),
               s.get("mode", "理想轮廓"), s.get("selection", "平均轮廓"), s.get("nm_per_px"),
               s["included_count"], s["excluded_count"], _class_label(s), _class_labels(s),
               _class_rating(s), _class_counts(s),
               _quality_join(s)] for s in all_samples),
               ["sample_id", "sample", "source", "mode", "selection", "nm_per_px", "included_count", "excluded_count",
                "class", "labels", "rating", "class_counts", "quality_warnings"])
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

    # v6.3 形貌量化对照：每样本互斥分类 + 六类量值（基准行在前）。
    profiles = report.get("profiles") or {}
    profile_entries = []
    if profiles.get("baseline"):
        profile_entries.append(profiles["baseline"])
        profile_entries.extend(profiles.get("samples") or [])
    else:
        for i, s in enumerate(all_samples):
            entry = _profile_entry_from_sample(s)
            entry["sample_id"] = s.get("sample_id", "B0" if i == 0 else f"S{i}")
            profile_entries.append(entry)
    class_headers = ["sample_id", "sample", "role", "class_label", "class_label_en", "labels", "rating",
                     "margin_nm", "wave",
                     "thresholds_shape_nm", "thresholds_wave_nm", "noise_sigma_nm",
                     *[key for key, _ in MORPH_SHAPES]]
    class_rows = []
    for i, entry in enumerate(profile_entries):
        cls = entry.get("classification") or {}
        thr = entry.get("thresholds") or {}
        shape = entry.get("shape_metrics") or {}
        label_cn = "方波基准" if entry.get("kind") == "square" else (cls.get("label_cn") or "")
        labels = cls.get("labels")
        labels_txt = "+".join(str(v) for v in labels) if isinstance(labels, list) and labels else ""
        if entry.get("kind") == "square":
            rating_txt = ""
        else:
            try:
                stars_n = int(cls.get("rating") or 0)
            except (TypeError, ValueError):
                stars_n = 0
            rating_txt = "★" * stars_n if stars_n > 0 else "0"
        class_rows.append([entry.get("sample_id", ""), entry.get("name", ""),
                           "baseline" if i == 0 else "sample",
                           label_cn, cls.get("label") or "",
                           labels_txt, rating_txt,
                           cls.get("margin"), cls.get("wave"),
                           thr.get("shape"), thr.get("wave"), entry.get("noise_sigma_nm"),
                           *[shape.get(key) for key, _ in MORPH_SHAPES]])
    write_rows("classes", "profile_classes.csv", class_rows, class_headers)

    paths["readme"] = os.path.join(folder, "README.txt")
    with open(paths["readme"], "w", encoding="utf-8") as f:
        f.write("Baseline：" + _sample_caption(report["baseline"], 100) + "\n\n")
        f.write("先看 sample_index.csv 对应编号、文件路径与区域；同名样本不会合并。\n")
        f.write("comparison_values/deltas/percent.csv 为横向比较，每个样本一列。\n")
        f.write("baseline_comparison.csv 为全部指标长表；per_sample/ 为每个样本单独的指标、逐柱表和对比图。\n")
        f.write("baseline_overlay.png 为总览；baseline_delta_bars.png 为独立横向点柱图。\n")
        f.write("profile_classes.csv 为每款样本的互斥形貌类别与六类量值（含阈值与 σ_noise）；\n")
        f.write("profile_class_table.png 为类别量化对照表（样本值与相对基准 Δ）。\n")
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
        paths["class_table"] = os.path.join(folder, "profile_class_table.png")
        class_fig = draw_profile_class_table(report)
        class_fig.savefig(paths["class_table"])
        class_fig.clear()
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
