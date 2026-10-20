# -*- coding: utf-8 -*-
"""pr_8_profile_validation -- profile 判定验证与调参脚本（v6.5 新增）。

对文件夹内文件名末尾带人工标注（normal / footing / undercut / 中间细 等）
的图片做批量验证：识别一次并缓存，随后用缓存的柱轮廓反复调用
build_morphology 评估不同阈值参数下的多标签分类与星级，与人工标注
对比输出报告；--sweep 对参数网格扫描命中率，供调参定稿。

独立脚本，不修改 6 个主模块的行为；参数扫描只在进程内临时生效，
退出即还原。用法：
  python pr_8_profile_validation.py                # 现参数基线报告
  python pr_8_profile_validation.py --sweep        # 参数网格扫描
  python pr_8_profile_validation.py --dir <目录> --out <输出目录>
  python pr_8_profile_validation.py --no-process   # 只用已有缓存，不重新识别
仅使用当前 Python 环境的 NumPy / OpenCV，不联网、不下载。
"""

import argparse
import csv
import itertools
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# 文件名末尾的人工标注关键词 → 六类形貌 key（与 PROFILE_CLASSES 命名一致）
GT_KEYWORDS = [
    ("undercut", "undercut"),
    ("footing", "footing"),
    ("normal", "normal"),
    ("中间细", "middle_thin"),
    ("中间粗", "middle_thick"),
    ("t-top", "t_top"),
    ("ttop", "t_top"),
    ("top_rounding", "top_rounding"),
    ("toprounding", "top_rounding"),
    ("驻波", "standing_wave"),
    ("standing_wave", "standing_wave"),
]

# 已知样本的硬编码兜底映射（文件名小写包含 → 标注）
GT_FALLBACK = {
    "a45d609-14c-7": "undercut",
    "a4bs588-01-4": "footing",
    "a4by067-02m-5": "normal",
    "a4cd536-21m": "middle_thin",
    "a52q193-22": "middle_thin",
    "a575082-13": "undercut",
    "a63m149-2": "footing",
    "a63q162-24-2": "normal",
    "a63q162-24-6": "normal",
    "a68d024-07": "footing",
}

LABEL_CN = {
    "normal": "Normal", "t_top": "T-top", "top_rounding": "Top-rounding",
    "middle_thick": "中间粗", "middle_thin": "中间细",
    "footing": "Footing", "undercut": "Undercut", "standing_wave": "驻波",
}


def parse_ground_truth(filename):
    """从文件名解析人工标注：先关键词（取最后一个命中），再硬编码兜底。"""
    stem = os.path.splitext(os.path.basename(filename))[0]
    low = stem.lower()
    hit = None
    for keyword, label in GT_KEYWORDS:
        pos = low.rfind(keyword.lower())
        if pos >= 0 and (hit is None or pos > hit[0]):
            hit = (pos, label)
    if hit:
        return hit[1], stem[max(0, hit[0]):]
    for key, label in GT_FALLBACK.items():
        if key in low:
            return label, key
    return None, ""


def collect_images(root):
    """主目录下带图片扩展名的文件（不递归子目录，排除 备用/ 等）。"""
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(path) and name.lower().endswith(IMAGE_EXTS):
            out.append(path)
    return out


def identify_once(image_path, cache_dir):
    """低层管线识别（不写过程图）；结果 pickle 缓存，二次运行直接读缓存。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, pickle_name(image_path))
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "columns" in data:
            return data, True
    import pr_2_image_recognition as rec
    img, binary, aux = rec.preprocess_image(image_path, on_stage=None)
    columns, _profile = rec._detect_columns(img, binary, aux, roi_mode=False, on_stage=None)
    scale = rec.read_image_scale(image_path)
    data = {"columns": columns, "nm_per_px": scale["nm"] / scale["pixels"]}
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data, False


def pickle_name(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return safe + ".pkl"


# ---------------------------------------------------------------------
# 参数补丁：pr_3 以 from pr_1_config import 方式持有常量副本，
# 扫描时需同时替换两个模块中的值，finally 强制还原。
# ---------------------------------------------------------------------

def patch_params(overrides):
    import pr_1_config as cfg
    import pr_3_contour_calc as calc
    saved = {}
    for key, value in overrides.items():
        saved[key] = (getattr(cfg, key, None), getattr(calc, key, None))
        setattr(cfg, key, value)
        if hasattr(calc, key):
            setattr(calc, key, value)
    return saved


def restore_params(saved):
    import pr_1_config as cfg
    import pr_3_contour_calc as calc
    for key, (old_cfg, old_calc) in saved.items():
        setattr(cfg, key, old_cfg)
        setattr(calc, key, old_calc)


def evaluate(columns, nm_per_px, overrides=None):
    """对缓存的柱轮廓按给定参数组重算 build_morphology，返回评估摘要。"""
    import pr_3_contour_calc as calc
    saved = patch_params(overrides) if overrides else None
    try:
        morph = calc.build_morphology(columns, nm_per_px)
    finally:
        if saved:
            restore_params(saved)
    cls = morph.get("classification") or {}
    shapes = cls.get("sorted_shapes") or []
    return {
        "classification": cls,
        "labels": cls.get("labels") or [],
        "labels_cn": cls.get("labels_cn") or [],
        "rating": cls.get("rating") or 0,
        "thresholds": morph.get("thresholds") or {},
        "eff_thr": cls.get("eff_thr"),
        "noise_sigma_nm": morph.get("noise_sigma_nm"),
        "cd_mode_nm": morph.get("cd_mode_nm"),
        "included_count": morph.get("included_count"),
        "sorted_shapes": [(key, _finite(value)) for value, key, _label in shapes]
                         if shapes else
                         [(key, _finite(morph.get("metrics", {}).get(key)))
                          for key in ("t_top_nm", "top_rounding_nm", "middle_thick_nm",
                                      "middle_thin_nm", "footing_nm", "undercut_nm")],
    }


def _finite(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def match_verdict(gt, labels):
    """hit：标注 ∈ 多标签；primary_hit：标注 == 主标签；miss：不在标签内。"""
    if gt is None:
        return "n/a"
    if gt in labels:
        return "primary_hit" if labels and gt == labels[0] else "hit"
    return "miss"


def default_sweep_grid():
    return {
        "MORPH_SHAPE_CD_RATIO": [0.06, 0.08, 0.10, 0.12],
        "MORPH_NOISE_K": [3.0, 4.0, 5.0],
        "MORPH_SHAPE_CAP_NM": [4.0, 6.0, 8.0],
    }


def build_param_sets(grid):
    keys = sorted(grid)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        rows.append(dict(zip(keys, combo)))
    return rows


def fmt_shapes(sorted_shapes):
    parts = []
    for key, value in sorted_shapes:
        parts.append(f"{LABEL_CN.get(key, key)}={value:.2f}" if value is not None
                     else f"{LABEL_CN.get(key, key)}=-")
    return " ".join(parts)


def write_report(rows, out_dir, summary=None, sweep_rows=None, sweep_desc=""):
    os.makedirs(out_dir, exist_ok=True)
    headers = ["file", "ground_truth", "labels", "labels_cn", "primary", "rating",
               "verdict", "eff_thr_nm", "shape_thr_nm", "cd_mode_nm", "noise_sigma_nm",
               "included", "sorted_shapes"]
    csv_path = os.path.join(out_dir, "validation_report.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row["file"], row["gt"] or "", "+".join(row["labels"]),
                             "+".join(row["labels_cn"]), row["labels"][0] if row["labels"] else "",
                             "★" * row["rating"] if row["rating"] else "0",
                             row["verdict"],
                             _fmt(row["eff_thr"]), _fmt(row["thresholds"].get("shape")),
                             _fmt(row["cd_mode"]), _fmt(row["noise"]),
                             row["included"], fmt_shapes(row["sorted_shapes"])])
    md_path = os.path.join(out_dir, "validation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Profile 判定验证报告\n\n")
        if sweep_desc:
            f.write(f"参数组：{sweep_desc}\n\n")
        f.write("| 文件 | 人工标注 | 程序判定（多标签） | 星级 | 判定 | 主标签量值 | 生效阈值(nm) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for row in rows:
            stars = "★" * row["rating"] if row["rating"] else "-"
            margin = row["classification"].get("margin")
            f.write(f"| {row['file']} | {row['gt'] or '?'} | {'+'.join(row['labels_cn']) or 'Normal'} "
                    f"| {stars} | {row['verdict']} | {_fmt(margin)} | {_fmt(row['eff_thr'])} |\n")
        if summary:
            f.write(f"\n**汇总**：{summary}\n")
        if sweep_rows:
            f.write("\n## 参数扫描摘要（按命中数降序）\n\n")
            f.write("| 参数组 | hit | primary_hit | miss 明细 |\n|---|---|---|---|\n")
            for item in sweep_rows:
                f.write(f"| {item['desc']} | {item['hit']}/{item['total']} "
                        f"| {item['primary']}/{item['total']} | {item['miss_detail']} |\n")
            if sweep_rows:
                f.write(f"\n**建议**：采用命中率最高（并列时 primary_hit 最高）的参数组，"
                        f"写回 pr_1_config.py 后重跑基线确认。\n")
    return csv_path, md_path


def _fmt(value):
    value = _finite(value)
    return f"{value:.3f}" if value is not None else ""


def run_single(images, cache_dir, out_dir):
    """现参数基线验证：识别（或读缓存）→ build_morphology → 对比标注。"""
    rows = []
    for path in images:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        gt, gt_raw = parse_ground_truth(name)
        try:
            data, from_cache = identify_once(path, cache_dir)
            result = evaluate(data["columns"], data["nm_per_px"])
        except Exception as exc:
            rows.append({"file": name, "gt": gt, "labels": [], "labels_cn": [],
                         "rating": 0, "verdict": "error", "eff_thr": None,
                         "thresholds": {}, "cd_mode": None, "noise": None,
                         "included": 0, "sorted_shapes": [],
                         "classification": {}, "error": str(exc)})
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            continue
        labels = result["labels"]
        verdict = match_verdict(gt, labels)
        rows.append({"file": name, "gt": gt, "labels": labels,
                     "labels_cn": result["labels_cn"], "rating": result["rating"],
                     "verdict": verdict, "eff_thr": result["eff_thr"],
                     "thresholds": result["thresholds"], "cd_mode": result["cd_mode_nm"],
                     "noise": result["noise_sigma_nm"],
                     "included": result["included_count"],
                     "sorted_shapes": result["sorted_shapes"],
                     "classification": result["classification"]})
        stars = "★" * result["rating"]
        print(f"[{'缓存' if from_cache else '识别'}] {stem} -> "
              f"{'+'.join(result['labels_cn']) or 'Normal'} {stars} | GT={gt} | {verdict}")
    total = len([r for r in rows if r["verdict"] != "error"])
    hits = sum(1 for r in rows if r["verdict"] in ("hit", "primary_hit"))
    primary = sum(1 for r in rows if r["verdict"] == "primary_hit")
    summary = (f"共 {total} 图：hit {hits}/{total}，primary_hit {primary}/{total}")
    print(summary)
    csv_path, md_path = write_report(rows, out_dir, summary=summary)
    print("报告：", csv_path)
    print("报告：", md_path)
    return rows


def run_sweep(images, cache_dir, out_dir, grid):
    """参数网格扫描：每图识别一次，逐参数组评估并汇总命中率。"""
    param_sets = build_param_sets(grid)
    print(f"参数组共 {len(param_sets)} 组 × {len(images)} 图")
    cache = {}
    for path in images:
        data, from_cache = identify_once(path, cache_dir)
        cache[path] = data
    sweep_rows = []
    for idx, overrides in enumerate(param_sets):
        desc = ", ".join(f"{k.replace('MORPH_', '')}={v:g}" for k, v in sorted(overrides.items()))
        hit = primary = miss = 0
        miss_detail = []
        for path, data in cache.items():
            gt, _ = parse_ground_truth(os.path.basename(path))
            result = evaluate(data["columns"], data["nm_per_px"], overrides)
            verdict = match_verdict(gt, result["labels"])
            if verdict in ("hit", "primary_hit"):
                hit += 1
                if verdict == "primary_hit":
                    primary += 1
            elif verdict == "miss":
                miss += 1
                miss_detail.append(f"{os.path.basename(path)[:24]}→{'+'.join(result['labels_cn']) or 'Normal'}")
        sweep_rows.append({"desc": desc, "overrides": overrides, "hit": hit,
                           "primary": primary, "total": len(cache),
                           "miss_detail": "；".join(miss_detail) or "-"})
        print(f"[{idx + 1}/{len(param_sets)}] {desc} -> hit {hit}/{len(cache)}, primary {primary}/{len(cache)}")
    sweep_rows.sort(key=lambda item: (-item["hit"], -item["primary"]))
    rows = []
    for path in images:
        name = os.path.basename(path)
        gt, _ = parse_ground_truth(name)
        data = cache[path]
        result = evaluate(data["columns"], data["nm_per_px"])
        labels = result["labels"]
        rows.append({"file": name, "gt": gt, "labels": labels,
                     "labels_cn": result["labels_cn"], "rating": result["rating"],
                     "verdict": match_verdict(gt, labels), "eff_thr": result["eff_thr"],
                     "thresholds": result["thresholds"], "cd_mode": result["cd_mode_nm"],
                     "noise": result["noise_sigma_nm"],
                     "included": result["included_count"],
                     "sorted_shapes": result["sorted_shapes"],
                     "classification": result["classification"]})
    grid_desc = "；".join(f"{k}∈{[f'{v:g}' for v in vals]}" for k, vals in sorted(grid.items()))
    _, md_path = write_report(rows, out_dir, summary=None, sweep_rows=sweep_rows,
                              sweep_desc=grid_desc)
    sweep_csv = os.path.join(out_dir, "sweep_summary.csv")
    with open(sweep_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["params", "hit", "primary_hit", "total", "miss_detail"])
        for item in sweep_rows:
            writer.writerow([item["desc"], item["hit"], item["primary"], item["total"],
                             item["miss_detail"]])
    best = sweep_rows[0]
    print(f"\n最优参数组：{best['desc']} -> hit {best['hit']}/{best['total']}, "
          f"primary_hit {best['primary']}/{best['total']}")
    print("报告：", md_path)
    print("报告：", sweep_csv)
    return sweep_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="profile 判定验证与调参（人工标注对比）")
    parser.add_argument("--dir", default=HERE, help="图片目录（默认脚本所在目录）")
    parser.add_argument("--out", default=None, help="输出目录（默认 图片目录/validation_out）")
    parser.add_argument("--cache", default=None, help="识别缓存目录（默认 输出目录/cache）")
    parser.add_argument("--sweep", action="store_true", help="参数网格扫描")
    parser.add_argument("--no-process", action="store_true",
                        help="只用已有缓存，不重新识别（缓存缺失的图跳过）")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.dir)
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(root, "validation_out")
    cache_dir = os.path.abspath(args.cache) if args.cache else os.path.join(out_dir, "cache")
    images = collect_images(root)
    if not images:
        print(f"目录中没有图片：{root}")
        return 1
    print(f"待验证 {len(images)} 张图（人工标注来自文件名末尾）")
    if args.sweep:
        run_sweep(images, cache_dir, out_dir, default_sweep_grid())
    else:
        run_single(images, cache_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
