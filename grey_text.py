# -*- coding: utf-8 -*-
"""gray_text -- 整个文件夹的 TIF 图 ↔ 单个灰度数值文本 双向转换工具。

完全独立的小工具，不依赖本文件夹内任何其他脚本，可单独复制运行：
1. 图 → 文本：把文件夹内所有图片读为 8 位灰度，合并写入一个 .txt 文件；
   每张图一段，段头记录文件名与宽高，段内每行是逐像素灰度（空格分隔）。
2. 文本 → 图：读该 .txt，一次性还原全部图片（默认 PNG，可选 16 位 TIFF）。

文本格式（UTF-8）：
  # gray_text bundle v1
  # count=3
  # file=图片名.tif width=640 height=480
  <height 行，每行 width 个灰度值>
  # file=下一张.tif ...

用法：
  python gray_text.py                启动图形界面
  python gray_text.py --to-text      当前目录所有图 → gray_images.txt
  python gray_text.py --to-image     当前目录 gray_images.txt → 图片
  python gray_text.py --to-text --dir "D:\\imgs" --out "D:\\out\\all.txt"
  python gray_text.py --to-image --txt "D:\\out\\all.txt" --out "D:\\imgs" --format tif
仅使用当前 Python 环境的 NumPy / OpenCV / Tkinter，不联网、不下载。
"""

import argparse
import os
import re
import sys

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

APP_TITLE = "灰度图 ↔ 文本转换工具"
BUNDLE_MAGIC = "# gray_text bundle v1"
DEFAULT_TXT_NAME = "gray_images.txt"

FILE_LINE_RE = re.compile(r"^(.*)\s+width=(\d+)\s+height=(\d+)\s*$")


# =====================================================================
# Unicode 安全读写（兼容中文路径 / UNC 网络路径）
# =====================================================================

def imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """兼容 Windows 中文路径与 UNC 路径的图片读取。"""
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
    """兼容中文路径的图片写入。"""
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, encoded = cv2.imencode(ext, img)
        if ok:
            encoded.tofile(path)
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img)


# =====================================================================
# 灰度化
# =====================================================================

def to_gray8(img_raw):
    """任意输入图转 8 位灰度：
    BGR/BGRA 用 G = 0.114B + 0.587G + 0.299R；
    16 位等高位深按 Q1~Q99 拉伸到 0~255，避免离群亮暗点压垮动态范围。
    """
    if img_raw is None:
        raise ValueError("空图像")
    if img_raw.ndim == 2:
        arr = img_raw.astype(np.float64)
    elif img_raw.ndim == 3 and img_raw.shape[2] in (3, 4):
        arr = (0.114 * img_raw[:, :, 0].astype(np.float64)
               + 0.587 * img_raw[:, :, 1].astype(np.float64)
               + 0.299 * img_raw[:, :, 2].astype(np.float64))
    else:
        raise ValueError("不支持的图像维度：%r" % (img_raw.shape,))

    if arr.max() > 255.5:  # 高位深 → 8 位
        lo = float(np.percentile(arr, 1))
        hi = float(np.percentile(arr, 99))
        if not np.isfinite(hi) or hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros(arr.shape, dtype=np.uint8)
        arr = (arr - lo) * 255.0 / (hi - lo)
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


# =====================================================================
# 打包文本：一个 txt 包含文件夹内全部图片
# =====================================================================

def write_gray_bundle(txt_path, items, overwrite=False):
    """把 [(文件名, 灰度矩阵), ...] 合并写入一个 txt。返回写入的图片数。"""
    if not overwrite:
        txt_path = _unique_path(txt_path)
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(BUNDLE_MAGIC + "\n")
        f.write("# count=%d\n" % len(items))
        for name, gray in items:
            h, w = gray.shape
            f.write("# file=%s width=%d height=%d\n" % (name, w, h))
            for row in gray.tolist():
                f.write(" ".join(map(str, row)))
                f.write("\n")
    return len(items), txt_path


def parse_gray_bundle(txt_path):
    """解析打包文本 → [(文件名或None, uint8灰度矩阵, 元信息dict), ...]。
    兼容旧版单图格式（无 # file= 行、仅有 # width=/# height= 头）。
    """
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    entries = []
    cur_name, cur_meta, cur_rows = None, {}, []

    def finish():
        if not cur_rows:
            return
        widths = {len(r) for r in cur_rows}
        if len(widths) != 1:
            raise ValueError("图像 %s 各行数值个数不一致（%s）"
                             % (cur_name or "?", sorted(widths)))
        w = widths.pop()
        h = len(cur_rows)
        try:
            hw = int(cur_meta.get("width", w))
            hh = int(cur_meta.get("height", h))
        except ValueError:
            hw, hh = w, h
        if hw != w or hh != h:
            raise ValueError("%s：头信息 %dx%d 与实际 %dx%d 不符"
                             % (cur_name or txt_path, hw, hh, w, h))
        gray = np.clip(np.asarray(cur_rows, dtype=np.int32), 0, 255).astype(np.uint8)
        entries.append((cur_name, gray, cur_meta))
        del cur_rows[:]

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            body = s.lstrip("#").strip()
            if body.startswith("file="):
                finish()
                cur_name, cur_meta, cur_rows = None, {}, []
                rest = body[len("file="):].strip()
                m = FILE_LINE_RE.match(rest)
                if m:
                    cur_name = m.group(1).strip()
                    cur_meta["width"] = m.group(2)
                    cur_meta["height"] = m.group(3)
                else:
                    cur_name = rest
                continue
            for tok in body.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    cur_meta[k.strip().lower()] = v.strip()
            continue
        try:
            cur_rows.append([int(t) for t in s.replace(",", " ").split()])
        except ValueError:
            raise ValueError("无法解析的数值行：%s" % s[:60])
    finish()

    if not entries:
        raise ValueError("文本中没有图像数据：%s" % txt_path)
    return entries


# =====================================================================
# 批量转换
# =====================================================================

IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
TEXT_EXT = ".txt"


def _unique_path(path):
    """目标已存在时追加 _1/_2… 避免覆盖。"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 1
    while True:
        cand = "%s_%d%s" % (base, n, ext)
        if not os.path.exists(cand):
            return cand
        n += 1


def batch_images_to_text(in_dir, out_txt=None, overwrite=False, exts=IMAGE_EXTS):
    """文件夹内所有图片 → 合并为一个灰度文本。返回 (成功数, 输出txt, 失败详情 list)。"""
    in_dir = os.path.abspath(in_dir)
    if not os.path.isdir(in_dir):
        raise ValueError("输入文件夹不存在：%s" % in_dir)
    if not out_txt:
        out_txt = os.path.join(in_dir, DEFAULT_TXT_NAME)
    elif os.path.isdir(out_txt):
        out_txt = os.path.join(out_txt, DEFAULT_TXT_NAME)
    exts = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in exts)
    names = sorted(n for n in os.listdir(in_dir)
                   if n.lower().endswith(exts) and os.path.isfile(os.path.join(in_dir, n)))
    if not names:
        raise ValueError("文件夹内没有可转换的图片：%s" % in_dir)

    items, fails = [], []
    for name in names:
        try:
            raw = imread_unicode(os.path.join(in_dir, name), cv2.IMREAD_UNCHANGED)
            gray = to_gray8(raw)
            items.append((name, gray))
            print("[OK] %s (%dx%d)" % (name, gray.shape[1], gray.shape[0]))
        except Exception as exc:
            msg = "%s: %s" % (name, exc)
            print("[失败] " + msg)
            fails.append(msg)
    if not items:
        return 0, None, fails

    count, final_txt = write_gray_bundle(out_txt, items, overwrite=overwrite)
    size_mb = os.path.getsize(final_txt) / (1024.0 * 1024.0)
    print("图→文本 完成：%d 张图片合并为 %s（%.1f MB），%d 张失败"
          % (count, final_txt, size_mb, len(fails)))
    return count, final_txt, fails


def batch_text_to_images(txt_path, out_dir=None, fmt="png", overwrite=False):
    """一个灰度文本 → 还原全部图片。返回 (成功数, 失败详情 list)。"""
    txt_path = os.path.abspath(txt_path)
    if not os.path.isfile(txt_path):
        raise ValueError("文本文件不存在：%s" % txt_path)
    out_dir = os.path.abspath(out_dir) if out_dir else os.path.dirname(txt_path)
    os.makedirs(out_dir, exist_ok=True)
    entries = parse_gray_bundle(txt_path)

    stem0 = os.path.splitext(os.path.basename(txt_path))[0]
    ext = "." + fmt.lower().lstrip(".")
    ok_count, fails = 0, []
    for i, (name, gray, _meta) in enumerate(entries):
        if name:
            stem = os.path.splitext(name)[0]
        else:
            stem = stem0 if len(entries) == 1 else "%s_%d" % (stem0, i + 1)
        dst = os.path.join(out_dir, stem + ext)
        if not overwrite:
            dst = _unique_path(dst)
        try:
            if ext in (".jpg", ".jpeg"):
                img = gray  # jpg 不支持 16 位
            else:
                img = gray.astype(np.uint16) * 257  # 还原为 16 位，避免有损
            if not imwrite_unicode(dst, img):
                raise ValueError("图片写入失败")
            print("[OK] %s -> %s (%dx%d)" % (name or stem0, os.path.basename(dst),
                                             gray.shape[1], gray.shape[0]))
            ok_count += 1
        except Exception as exc:
            msg = "%s: %s" % (name or stem, exc)
            print("[失败] " + msg)
            fails.append(msg)
    print("文本→图 完成：%d 成功 / %d 失败，输出目录：%s" % (ok_count, len(fails), out_dir))
    return ok_count, fails


# =====================================================================
# 命令行入口
# =====================================================================

def run_cli(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(prog="gray_text.py",
                                     description="整个文件夹的 TIF 图与单个灰度文本双向转换")
    parser.add_argument("--to-text", action="store_true", help="文件夹内所有图 → 一个灰度文本")
    parser.add_argument("--to-image", action="store_true", help="一个灰度文本 → 还原全部图片")
    parser.add_argument("--dir", default=here, help="图片文件夹（--to-text 用，默认本脚本所在目录）")
    parser.add_argument("--txt", default=None, help="灰度文本文件（--to-image 用；未给则在 --dir 内查找）")
    parser.add_argument("--out", default=None,
                        help="输出：--to-text 为 txt 路径（默认 图片目录/gray_images.txt）；"
                             "--to-image 为图片输出目录（默认 txt 所在目录）")
    parser.add_argument("--format", default="png", choices=["png", "tif", "jpg", "bmp"],
                        help="文本→图 的输出格式（默认 png）")
    parser.add_argument("--ext", default="tif,tiff,png,jpg,jpeg,bmp",
                        help="图→文本 时处理的扩展名（逗号分隔，默认全部图片类型）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    args = parser.parse_args(argv)

    if args.to_text and args.to_image:
        parser.error("--to-text 与 --to-image 只能选一个")
    if args.to_text:
        batch_images_to_text(args.dir, args.out, overwrite=args.overwrite,
                             exts=[e.strip() for e in args.ext.split(",") if e.strip()])
    elif args.to_image:
        txt = args.txt
        if not txt:
            found = sorted(n for n in os.listdir(args.dir)
                           if n.lower().endswith(TEXT_EXT) and os.path.isfile(os.path.join(args.dir, n)))
            if not found:
                parser.error("在 %s 中未找到灰度文本，请用 --txt 指定" % args.dir)
            txt = os.path.join(args.dir, found[0])
            print("使用文本文件：" + txt)
        batch_text_to_images(txt, args.out, fmt=args.format, overwrite=args.overwrite)
    else:
        print("未指定转换方向。图→文本加 --to-text，文本→图加 --to-image；直接运行本脚本可打开图形界面。")
        parser.print_help()
    return 0


# =====================================================================
# 图形界面
# =====================================================================

def gui_main():
    if cv2 is None:
        print("未安装 OpenCV（cv2），无法运行图形界面。请先安装后重试。")
        return 1
    try:
        from tkinter import filedialog, messagebox, ttk
        import tkinter as tk
    except Exception:
        print("未找到 Tkinter，无法运行图形界面。请改用命令行参数（--to-text / --to-image）。")
        return 1

    here = os.path.dirname(os.path.abspath(__file__))
    root = tk.Tk()
    root.title(APP_TITLE)
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=12)
    frm.grid(sticky="nsew")

    var_mode = tk.StringVar(value="img2txt")
    var_in_dir = tk.StringVar(value=here)
    var_out_txt = tk.StringVar()
    var_in_txt = tk.StringVar()
    var_out_dir = tk.StringVar()
    var_fmt = tk.StringVar(value="png")
    var_overwrite = tk.BooleanVar(value=False)

    # ---- 模式选择 ----
    row = 0
    ttk.Radiobutton(frm, text="图 → 文本（文件夹内所有图片合并为一个 txt）",
                    variable=var_mode, value="img2txt",
                    command=lambda: switch_mode()).grid(row=row, column=0, columnspan=3, sticky="w")
    row += 1
    ttk.Radiobutton(frm, text="文本 → 图（一个 txt 还原全部图片）",
                    variable=var_mode, value="txt2img",
                    command=lambda: switch_mode()).grid(row=row, column=0, columnspan=3, sticky="w")

    row += 1
    ttk.Separator(frm, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=6)

    # ---- 图→文本 区域 ----
    frm_i2t = ttk.Frame(frm)
    ttk.Label(frm_i2t, text="图片文件夹").grid(row=0, column=0, sticky="w")
    ent_in_dir = ttk.Entry(frm_i2t, textvariable=var_in_dir, width=52)
    ent_in_dir.grid(row=0, column=1, padx=6)

    def pick_in_dir():
        d = filedialog.askdirectory(parent=root, title="选择图片文件夹", initialdir=var_in_dir.get() or here)
        if d:
            var_in_dir.set(d)
            if not var_out_txt.get():
                var_out_txt.set(os.path.join(d, DEFAULT_TXT_NAME))

    ttk.Button(frm_i2t, text="浏览…", command=pick_in_dir).grid(row=0, column=2)
    ttk.Label(frm_i2t, text="输出文本文件").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ent_out_txt = ttk.Entry(frm_i2t, textvariable=var_out_txt, width=52)
    ent_out_txt.grid(row=1, column=1, padx=6, pady=(6, 0), sticky="ew")

    def pick_out_txt():
        d = filedialog.asksaveasfilename(
            parent=root, title="保存灰度文本", defaultextension=".txt",
            initialfile=DEFAULT_TXT_NAME,
            initialdir=os.path.dirname(var_out_txt.get()) or var_in_dir.get() or here,
            filetypes=[("文本文件", "*.txt"), ("全部文件", "*.*")])
        if d:
            var_out_txt.set(d)

    ttk.Button(frm_i2t, text="浏览…", command=pick_out_txt).grid(row=1, column=2, pady=(6, 0))

    # ---- 文本→图 区域 ----
    frm_t2i = ttk.Frame(frm)
    ttk.Label(frm_t2i, text="灰度文本").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm_t2i, textvariable=var_in_txt, width=52).grid(row=0, column=1, padx=6)

    def pick_in_txt():
        f = filedialog.askopenfilename(parent=root, title="选择灰度文本",
                                       initialdir=os.path.dirname(var_in_txt.get()) or here,
                                       filetypes=[("文本文件", "*.txt"), ("全部文件", "*.*")])
        if f:
            var_in_txt.set(f)
            if not var_out_dir.get():
                var_out_dir.set(os.path.dirname(f))

    ttk.Button(frm_t2i, text="浏览…", command=pick_in_txt).grid(row=0, column=2)
    ttk.Label(frm_t2i, text="输出图片文件夹").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ttk.Entry(frm_t2i, textvariable=var_out_dir, width=52).grid(row=1, column=1, padx=6, pady=(6, 0))

    def pick_out_dir():
        d = filedialog.askdirectory(parent=root, title="选择图片输出文件夹",
                                    initialdir=var_out_dir.get() or os.path.dirname(var_in_txt.get()) or here)
        if d:
            var_out_dir.set(d)

    ttk.Button(frm_t2i, text="浏览…", command=pick_out_dir).grid(row=1, column=2, pady=(6, 0))
    ttk.Label(frm_t2i, text="输出格式").grid(row=2, column=0, sticky="w", pady=(6, 0))
    ttk.Combobox(frm_t2i, textvariable=var_fmt, values=["png", "tif", "jpg", "bmp"],
                 width=8, state="readonly").grid(row=2, column=1, sticky="w", padx=6, pady=(6, 0))

    # ---- 公共区域 ----
    row += 1
    ttk.Checkbutton(frm, text="覆盖已存在的输出文件（默认自动改名 _1/_2 不覆盖）",
                    variable=var_overwrite).grid(row=row, column=0, columnspan=4, sticky="w", pady=(6, 0))

    row += 1
    out_log = tk.Text(frm, width=78, height=14, state="disabled")
    out_log.grid(row=row, column=0, columnspan=4, pady=(8, 4))
    sb = ttk.Scrollbar(frm, command=out_log.yview)
    sb.grid(row=row, column=4, sticky="ns")
    out_log.configure(yscrollcommand=sb.set)

    def log(msg):
        out_log.configure(state="normal")
        out_log.insert("end", str(msg) + "\n")
        out_log.see("end")
        out_log.configure(state="disabled")
        root.update_idletasks()

    def switch_mode():
        if var_mode.get() == "img2txt":
            frm_t2i.grid_remove()
            frm_i2t.grid(row=2, column=0, columnspan=4, sticky="ew")
        else:
            frm_i2t.grid_remove()
            frm_t2i.grid(row=2, column=0, columnspan=4, sticky="ew")

    def run_convert():
        try:
            if var_mode.get() == "img2txt":
                in_dir = var_in_dir.get().strip()
                if not in_dir or not os.path.isdir(in_dir):
                    messagebox.showerror(APP_TITLE, "请选择有效的图片文件夹", parent=root)
                    return
                out_txt = var_out_txt.get().strip() or None
                count, final_txt, fails = batch_images_to_text(in_dir, out_txt,
                                                               overwrite=var_overwrite.get())
                if final_txt:
                    log("输出文本：" + final_txt)
            else:
                txt = var_in_txt.get().strip()
                if not txt or not os.path.isfile(txt):
                    messagebox.showerror(APP_TITLE, "请选择有效的灰度文本文件", parent=root)
                    return
                out_dir = var_out_dir.get().strip() or os.path.dirname(txt)
                count, fails = batch_text_to_images(txt, out_dir, fmt=var_fmt.get(),
                                                    overwrite=var_overwrite.get())
            log("— 完成 —")
            if fails:
                log("有 %d 个失败，详见上方 [失败] 行。" % len(fails))
                messagebox.showwarning(APP_TITLE, "完成：%d 成功，%d 失败。\n详情见日志区。"
                                       % (count, len(fails)), parent=root)
            else:
                messagebox.showinfo(APP_TITLE, "完成：%d 个文件转换成功。" % count, parent=root)
        except Exception as exc:
            log("错误：%s" % exc)
            messagebox.showerror(APP_TITLE, "转换出错：%s" % exc, parent=root)

    row += 1
    ttk.Button(frm, text="开始转换", command=run_convert).grid(row=row, column=1, pady=(8, 2))

    switch_mode()
    if not var_out_txt.get():
        var_out_txt.set(os.path.join(here, DEFAULT_TXT_NAME))
    root.mainloop()
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        return gui_main()
    return run_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
