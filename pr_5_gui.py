# -*- coding: utf-8 -*-
"""pr_5_gui -- 前端 UI 层：Tkinter 主界面、帮助窗口与 Baseline 对比窗口。

从 pr6.1.py 拆分而来。依赖方向（单向）：
pr_1_config -> pr_2_image_recognition -> pr_3_contour_calc -> pr_4_contour_compare -> 本模块。
拆分前单文件里 ``core = sys.modules[__name__]`` 指向自身；
拆分后改为 ``import pr_1_config as core``，运行时可变参数通过
``_sync_param`` 同步到所有已导入该参数的模块，保持原语义。
"""

import contextlib
import copy
import glob
import json
import math
import os
import queue
import shutil
import sys
import threading
import traceback

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
import numpy as np

import pr_1_config as core
from pr_1_config import (
    ABOUT_HELP,
    APP_TITLE,
    BASELINE_HELP,
    FILE_TAG_HELP,
    FORMULA_HELP,
    MATPLOTLIB_AVAILABLE,
    MORPHOMETRY,
    MORPHOLOGY_DEFINITION,
    PIPELINE_HELP,
    PROFILE_FORMULA_HELP,
    PROFILE_HELP,
    QUICKSTART_HELP,
    RESULT_BANNER_H,
    ROUGHNESS_KEYS,
    SCALE_HELP,
    SHAPE_HELP,
    STEP_HELP,
    AXIS_MARGIN_L,
    PANEL_W,
    image_key,
    read_image_scale,
    save_image_scale,
    validate_scale,
)
from pr_2_image_recognition import (
    export_detection_outputs,
    imwrite_unicode,
    imread_unicode,
    process_image,
    process_image_roi,
)
from pr_3_contour_calc import (
    _finite_number,
    _num,
    build_morphology,
    compare_morphologies,
    morphology_labels,
    square_baseline,
)
from pr_4_contour_compare import (
    _sample_caption,
    draw_baseline_figure,
    export_baseline_for_app,
    export_baseline_report,
)

if MATPLOTLIB_AVAILABLE:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

_MUTABLE_PARAM_MODULES = None


def _sync_param(attr, value):
    """把运行时可变参数同步到所有已导入它的模块，保持拆分前单模块语义。"""
    global _MUTABLE_PARAM_MODULES
    if _MUTABLE_PARAM_MODULES is None:
        import pr_3_contour_calc as _calc
        import pr_2_image_recognition as _recog
        _MUTABLE_PARAM_MODULES = [_calc, _recog]
    setattr(core, attr, value)
    for mod in _MUTABLE_PARAM_MODULES:
        if attr in vars(mod):
            setattr(mod, attr, value)




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
        for title, var, choices, width in (("查看方式", self.table_view, ("横向比较", "形貌对比", "样本明细", "逐柱数值"), 12),
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
            base_line = (f"{sample['sample_id']} · {sample['name']}  |  {sample.get('source', '标准方波')}  |  "
                         f"{sample.get('mode', '理想轮廓')} · {sample.get('selection', '平均轮廓')}  |  "
                         f"纳入 {sample['included_count']} / 排除 {sample['excluded_count']} 柱")
            if sample.get("kind") == "square":
                base_line += "  |  方波基准（理想轮廓）"
            else:
                cls = sample.get("classification") or {}
                if cls.get("label_cn"):
                    counts = sample.get("classification_counts") or {}
                    counts_txt = "、".join(f"{k}×{v}" for k, v in counts.items()) if counts else ""
                    thr = sample.get("thresholds") or {}
                    thr_txt = f"，阈值 形状≥{thr.get('shape', '-'):.2f}/驻波≥{thr.get('wave', '-'):.2f}nm"
                    if thr:
                        cd_m = thr.get('cd_mode')
                        if cd_m:
                            thr_txt += f"；生效≥max(形状, {thr.get('shape_cd_ratio', 0.1):.0%}×CD={cd_m:.2f})"
                    base_line += f"  |  形貌分类：{cls['label_cn']}" + (f"（{counts_txt}）" if counts_txt else "") + thr_txt
            lines.append(base_line)
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
        elif view == "形貌对比":
            columns = [("cls", "互斥分类", 120)] + [(k, title + " Δ(nm)", 130) for k, title, _ in MORPH_SHAPES] + [("dominant", "较基准主导项", 160)]
        elif view == "样本明细":
            columns = [("ref", "B0 (nm)", 85), ("value", "样本值 (nm)", 95), ("delta", "增减 Δ (nm)", 100),
                       ("pct", "变化 (%)", 85), ("mean", "逐柱均值 (nm)", 110), ("sd", "逐柱标准差", 100),
                       ("n", "有效柱数", 75), ("hits", "≥阈值柱数", 85), ("basis", "计算依据", 150)]
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
        elif view == "形貌对比":
            shape_defs = [(k, t) for k, t, _ in MORPH_SHAPES]
            for i, sample in enumerate([baseline] + samples):
                sid = sample["sample_id"]
                cls_txt = "方波基准" if sample.get("kind") == "square" else \
                    ((sample.get("classification") or {}).get("label_cn") or "—")
                deltas = []
                for key, _ in shape_defs:
                    ref, value = baseline["metrics"].get(key), sample["metrics"].get(key)
                    ok = _finite_number(value) and _finite_number(ref)
                    deltas.append(float(value) - float(ref) if ok else None)
                dominant = "—"
                if sid != "B0":
                    finite = [d for d in deltas if d is not None]
                    if finite:
                        max_i = max(range(len(deltas)),
                                    key=lambda j: deltas[j] if deltas[j] is not None else float("-inf"))
                        d = deltas[max_i]
                        dominant = f"{shape_defs[max_i][1]} {d:+.2f}" if abs(d) > 1e-9 else "与基准持平"
                    else:
                        dominant = "—"
                row_vals = [cls_txt] + [_num(d, True) if d is not None else "—" for d in deltas] + [dominant]
                tags = ("group",) if sid == "B0" else (("stripe",) if i % 2 else ())
                tree.insert("", "end", iid=sid, text=_sample_caption(sample, 60), values=row_vals, tags=tags)
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
        ttk.Button(top, text="选择图片", command=self._browse_file).pack(side="left", padx=4)

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

    def _list_images(self):
        folder = self._folder()
        if not os.path.isdir(folder):
            return []
        return sorted({
            p
            for ext in core.INPUT_GLOBS
            for p in glob.glob(os.path.join(folder, ext))
        })

    def _refresh_files(self):
        names = [os.path.basename(p) for p in self._list_images()]
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

        for p in self._list_images():
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
            title="选择包含图片的文件夹"
        )
        if d:
            self.var_dir.set(os.path.normpath(d))
            self._refresh_files()

    def _browse_file(self):
        if self.running:
            return
        f = filedialog.askopenfilename(
            initialdir=self._folder(),
            title="选择 PR 图像",
            filetypes=[
                ("TIF 图像", "*.tif *.tiff *.TIF *.TIFF"),
                ("PNG 图像", "*.png *.PNG"),
                ("JPG 图像", "*.jpg *.jpeg *.JPG *.JPEG"),
                ("BMP 图像", "*.bmp *.BMP"),
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

        _sync_param("INPUT_DIR", folder)
        _sync_param("OUTPUT_SUBDIR", bool(self.var_subdir.get()))
        _sync_param("DEBUG_OUTPUT", bool(self.var_debug.get()))
        _sync_param("PILLAR_POLARITY", {"亮柱（推荐）": "bright", "暗柱": "dark", "自动（需核对）": "auto"}[self.var_polarity.get()])

        for attr, name, kind, *_ in MAIN_PARAMS:
            raw = self.spins[attr].get().strip().replace("，", ".").replace(",", ".")
            value = int(float(raw)) if kind == "int" else float(raw)

            if not math.isfinite(float(value)):
                raise ValueError(f"{name} 必须是有限数值")
            if attr == "AUTO_CD_ESTIMATE_PX" and value < 0:
                raise ValueError("估计 Line CD 只能填 0 或正数")
            if attr == "AUTO_CD_TOLERANCE_PX" and value <= 0:
                raise ValueError("允许误差必须大于 0 px")

            _sync_param(attr, value)

        self._collect_scale_only()

    def _collect_scale_only(self):
        self._flush_scale_edits(refresh=False)
        scale = self._scale_from_fields()
        path = self._selected_image_path()
        if path:
            previous = read_image_scale(path)
            if previous != scale:
                self._apply_image_scale(path, scale, refresh=False)
        _sync_param("INPUT_DIR", self._folder())
        _sync_param("OUTPUT_SUBDIR", bool(self.var_subdir.get()))

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
        paths = self._list_images()
        if not paths:
            messagebox.showwarning("提示", "没有可识别的图片")
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
                        process_image_roi(
                            job["path"],
                            job["roi"],
                            on_image=self._make_cb("_user"), scale=job["scale"]
                        )
                    else:
                        process_image(
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
        self.combo.configure(values=[os.path.basename(p) for p in self._list_images()])
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
        # v6.5：主图顶部含 RESULT_BANNER_H 横幅条（render_combined 主图整体加高）
        return main_h + 120 + RESULT_BANNER_H, main_w + PANEL_W * n

    @staticmethod
    def _label_client_size(label, frame):
        """取预览 Label 的实际可用客户区尺寸（像素）。

        Label 以 fill=both+expand 塞进 pack_propagate(False) 的
        LabelFrame，其 winfo_width/height 即图片可显示区域；窗口尚未
        布局（返回 1）或异常时回退到 "frame-8/-42" 旧估算。
        """
        try:
            lw = int(label.winfo_width())
            lh = int(label.winfo_height())
        except Exception:
            lw = lh = 0
        if lw > 8 and lh > 8:
            return max(lw - 2, 80), max(lh - 2, 60)
        fw = max(frame.winfo_width() - 8, 80)
        fh = max(frame.winfo_height() - 42, 60)
        return fw, fh

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

        # v6.5：改用 Label 实际客户区（winfo_*）计算统一缩放比，
        # 旧的 "frame-8/-42" 估算值偏大，预览图底部/右侧会被裁切。
        lw_t, lh_t = self._label_client_size(self.lbl_tool, self.frm_tool)
        common = max(0.02, min(2.0, lw_t / tool_w, lh_t / tool_h))

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
            lw, lh = self._label_client_size(label, frame)
            s = min(s, max(0.02, min(2.0, lw / w, lh / h)))

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
        square_size = max(100, min(square_size, 760, max(100, parent.winfo_height() - 10)))  # v6.5 上限 600→760，大屏预览更大

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
        classifications = [c["classification"]["label_cn"] for c in morph["per_column"]]
        tree.insert("", "end", text="形貌分类（互斥）",
                    values=[morph["classification"]["label_cn"], *classifications], tags=("profile",))
        thresholds = morph.get("thresholds") or {}
        if thresholds:
            sigma = morph.get("noise_sigma_nm")
            sigma_txt = f"{sigma:.3f}" if isinstance(sigma, (int, float)) and math.isfinite(sigma) else "无"
            cd_m = thresholds.get('cd_mode')
            cd_part = (f"；生效阈值=max(形状, {thresholds.get('shape_cd_ratio', 0.1):.0%}×CD={cd_m:.2f})"
                       if cd_m else "")
            tree.insert("", "end", text="分类阈值 (nm)",
                        values=[f"形状≥{thresholds.get('shape', 0):.2f} / 驻波≥{thresholds.get('wave', 0):.2f}（σ_noise={sigma_txt}）{cd_part}"],
                        tags=("profile",))

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
            morph = payload["stats"]["morphology"]
            thresholds = morph.get("thresholds") or {}
            lines.append(
                "classification ok: "
                f"avg={morph['classification']['label_cn']}, "
                f"counts={morph.get('classification_counts')}, "
                f"thresholds=shape {thresholds.get('shape'):.2f}/wave {thresholds.get('wave'):.2f}, "
                f"sigma_noise={morph.get('noise_sigma_nm')}, "
                f"cd_mode={morph.get('cd_mode_nm')}"
            )
            if not MATPLOTLIB_AVAILABLE:
                raise AssertionError("缺少实时灰度曲线 / baseline 绘图依赖 Matplotlib")
            if payload["stats"]["morphology"]["included_count"]:
                sample = sample_morphology(comparison_sample(payload))
                baseline = square_baseline(sample["metrics"]["cd_mean_nm"], sample["metrics"]["height_nm"])
                report = compare_morphologies(baseline, [sample])
                paths = export_baseline_report(report, os.path.join(out_dir, "selftest_baseline"))
                if not os.path.isfile(paths["plot"]):
                    raise AssertionError("未生成 baseline 对比图")
                profiles = report.get("profiles") or {}
                if not profiles.get("baseline") or not profiles.get("samples"):
                    raise AssertionError("compare_morphologies 未返回 profiles（形貌量化对比）")
                if not os.path.isfile(paths.get("classes", "")):
                    raise AssertionError("未生成 profile_classes.csv")
                s_cls = profiles["samples"][0]["classification"]["label_cn"]
                lines.append(
                    f"profiles ok: baseline={profiles['baseline']['classification']['label_cn']}, "
                    f"sample={s_cls}, shape_keys={len(profiles['samples'][0]['shape_metrics'])}"
                )
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
