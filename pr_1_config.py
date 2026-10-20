# -*- coding: utf-8 -*-
"""PR profile analyzer v6.4 — 公共配置模块。

集中管理全局常量、形貌指标定义、帮助文本与每图比例尺读写。
所有其他模块（识别、计算、对比、GUI、主入口）统一从本模块读取配置；
GUI 通过 `import pr_1_config as core` 修改运行期参数（与原单文件行为一致）。
"""

import os
import sys

# ---------------------------------------------------------------------
# 限制科学计算库线程数：避免 GUI / 打包环境中过度抢占 CPU。
# 注意：本模块必须先于 numpy / cv2 被导入，环境变量才能生效。
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

APP_VERSION = "6.5"
APP_TITLE = f"PR profile analyzer v{APP_VERSION}"
EXE_NAME = f"PR_profile_analyzer_v{APP_VERSION}"


def application_dir():
    return os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))


# matplotlib 仅用于直线量测灰度剖面与 Baseline 图
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: F401
    from matplotlib.figure import Figure  # noqa: F401

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
INPUT_GLOBS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF",
              "*.png", "*.PNG",
              "*.jpg", "*.jpeg", "*.JPG", "*.JPEG",
              "*.bmp", "*.BMP")  # v6.2.2 多类型图片支持
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
AUTO_CD_ESTIMATE_PX = 0.0
AUTO_CD_TOLERANCE_PX = 12.0

# ---- 柱子有效性 ----
MIN_COLUMN_HEIGHT = 20
MIN_AVG_WIDTH = 15
MIN_ASPECT_RATIO = 1.2
MAX_TRACE_GAP = 8

# ---- v6.2.2 识别迭代回退开关（置 False 逐位回退对应修复）----
ENABLE_TOP_V_NOTCH_FIX = True        # 顶部 V 形凹陷插值回填（头部缺口修复）
ENABLE_WIDE_TOP_SHRINK = True        # 宽顶渐降收缩（横线帽残留，PEB110C PR5）
ENABLE_ASYMMETRIC_HEAD_TRIM = True   # head trim 按边缘差分单侧收缩（防对称误切）
TOP_EXTEND_GAIN_GATE = 1.05          # 顶部 DP 扩窗收益门（v6.5 由 1.08 放宽，减少头部缺失）
HEAD_TRIM_SCAN_ROWS = 8              # head trim 扫描行数（原 5）
ENABLE_FOOT_EXTEND_EVIDENCE = True
ENABLE_GRADIENT_FOOT_CLIP = True
GRADIENT_FOOT_CLIP_MAX = 4   # 底部延展加灰度证据校验（防拖尾）
FOOT_DROP_GRAY_GATE = 10.0           # foot 修复灰度差门（原 12.0）
ENABLE_BIG_PILLAR_MODE = True        # 大柱特写模式（30kx 单柱/双柱 PNG）
ENABLE_FLAT_BLOCK_EXEMPT = True      # 宽扁亮块豁免（沟槽亮块 PNG）
ENABLE_TRAPEZOID_GUARD = True        # 梯形台面守卫（跳过 cap/trim 误收缩）
ENABLE_OVERLAY_DOWNWEIGHT = True     # overlay 白色标注在 DP 中降权
ENABLE_BOTTOM_2ND_DEPTH_LIMIT = True # 底部二次下探深度限制（防高度虚高）
BATCH_FAILURE_LOG = "batch_failures.log"  # 批量失败日志文件名；None 关闭

# ---- 通用化：两阶段参数扫描（S2）----
ENABLE_EXTENDED_SWEEP = True
EXTENDED_SWEEP_GATE = 3.5
EXTENDED_SWEEP_MARGIN = 0.15

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
# 形貌指标定义
# =====================================================================

MORPHOMETRY = [
    ("height_nm", "高度", "nm"),
    ("cd_mean_nm", "平均 CD", "nm"),
    ("cd_mode_nm", "众数 CD", "nm"),
    ("t_top_nm", "T-top 顶部展宽量", "nm"),
    ("top_rounding_nm", "Top-rounding 等效圆角半径", "nm"),
    ("middle_thin_nm", "中间细：收缩量", "nm"),
    ("middle_thick_nm", "中间粗：鼓出量", "nm"),
    ("footing_nm", "Footing 底部展宽量", "nm"),
    ("undercut_nm", "Undercut 底部收缩量", "nm"),
    ("ler_left_nm", "左侧 LER 3σ", "nm"),
    ("ler_right_nm", "右侧 LER 3σ", "nm"),
    ("ler_nm", "两侧平均 LER 3σ", "nm"),
    ("lwr_nm", "LWR 3σ", "nm"),
    ("wave_amplitude_nm", "周期起伏：单边平均幅值", "nm"),
    ("wave_period_nm", "周期起伏：主周期", "nm"),
]
# 六类几何形貌（互斥分类用，基准线为众数 CD）；LER/LWR 不参与互斥分类，单独统计。
MORPH_SHAPES = [
    ("t_top_nm", "T-top"), ("top_rounding_nm", "Top-rounding"),
    ("middle_thick_nm", "中间粗"), ("middle_thin_nm", "中间细"),
    ("footing_nm", "Footing"), ("undercut_nm", "Undercut"),
]
ROUGHNESS_KEYS = {"ler_left_nm", "ler_right_nm", "ler_nm", "lwr_nm",
                  "wave_amplitude_nm", "wave_period_nm"}
PROFILE_SAMPLES = 401

# ---- v6.3 互斥分类：CD 众数基准与自适应阈值（v6.4 起阈值带上限）----
MORPH_MODE_BIN_MIN_NM = 0.5     # 众数分箱最小 bin 宽 (nm)
MORPH_MODE_BIN_RATIO = 0.05     # bin 宽 = max(下限, 该比例 × (P95-P10))
MORPH_MODE_FLAT_FACTOR = 1.5    # 众数峰计数 < 该倍数 × 中位 bin 计数 → 视为平坦，回退中位数
MORPH_NOISE_K = 4.0             # 形貌阈值系数：shape_thr = max(下限, min(K × σ_noise, 上限))
MORPH_WAVE_K = 6.0              # 驻波阈值系数：wave_thr = max(下限, min(K × σ_noise, 上限))
MORPH_SHAPE_FLOOR_NM = 1.0      # 形貌阈值下限 (nm)，与旧 1nm 统计口径一致
MORPH_SHAPE_CAP_NM = 6.0        # 形貌阈值上限 (nm)：σ 自适应封顶，防高噪声图漏报形貌
MORPH_WAVE_FLOOR_NM = 5.0       # 驻波 LER/LWR 阈值下限 (nm)
MORPH_WAVE_CAP_NM = 10.0        # 驻波阈值上限 (nm)：σ 自适应封顶
MORPH_WAVE_AMP_FLOOR_NM = 1.0   # 驻波周期幅值下限 (nm)：LER 偏大且 FFT 周期幅值显著才判驻波
MORPH_SHAPE_CD_RATIO = 0.10     # 形状阈值 CD 归一化：生效阈值 = max(σ自适应阈值, 比例×CD)
BOTTOM_PULL_WEAK = 0.3          # 底边弱证据二次搜索的向下粘滞先验（防基底条带吸附）
RESULT_BANNER_H = 34            # 结果图顶部分类横幅条高度 (px)：扩展画布，不覆盖原图
TOP_RESCUE_PULL = 0.35          # 顶边弱证据二次搜索的向上粘滞先验（仅 binary_fallback 触发）
TOP_RESCUE_GATE = 0.85          # 顶边二次搜索采纳门：新路径中位强度 ≥ 该倍数 × 原路径
ENABLE_LER_OUTLIER_FLAG = True  # 柱数≥10 时对 LER 显著离群柱加"疑似驻波离群"旗标
PROFILE_CLASSES = [
    ("normal", "Normal"), ("top_rounding", "Top-rounding"), ("t_top", "T-top"),
    ("middle_thick", "中间粗"), ("middle_thin", "中间细"),
    ("footing", "Footing"), ("undercut", "Undercut"), ("standing_wave", "驻波"),
]
MORPHOLOGY_DEFINITION = """每根柱以中段中心为 x=0，顶部为 y=0；按相对高度 0~1 对齐，等权平均左右边界坐标。
不归一化宽度；y 换算为各柱平均高度，所有几何值保留 nm 尺度。
众数 CD：每根柱取自身逐行宽度 W 的直方图众数（bin 宽 = max(0.5nm, 5%×(P95−P10))），
样本过少（<5）、分布恒定或众数峰不突出时回退中位数；全图众数 CD 由纳入柱行宽合并计算。
六类形貌量值以该柱众数 CD 为水平基准线（不再拟合柱身参考线），偏差先取 Q90 再与 0 取大：
T-top：顶部 z≤18% 的 W−CD众数 正偏差 Q90；Top-rounding：顶部 z≤20% 的 CD众数−W 正偏差 Q90，
并以同区域缺失截面积 A 换算等效半径 sqrt(A/(2(1−π/4)))。
Top-rounding 先扣除中段线性锥度（侧壁斜率）再换算等效半径：纯梯形/锥形轮廓约为 0，
外推锥度线以帽区最大观察宽封顶（防把整帽收窄误计为圆角）；等效半径再以亏缺深度上界
deficit_max/2 封顶——真圆角亏缺深度恰为 2r 不受影响，自然圆帽（平台+线性展宽）的
斜坡亏缺不再虚增半径；无法拟合（中段行不足或斜率非物理）时退回未扣锥度值；
圆角半径超过柱高 20% 时为下界估计。
中间粗/细：中段 35~65% 的 W−CD众数 正/负偏差 Q90。
Footing：底部 z≥82% 的 W−CD众数 正偏差 Q90；Undercut：底部 z≥82% 的 CD众数−W 正偏差 Q90。
互斥分类（每柱单标签）：先判驻波——LER 或 LWR ≥ 驻波阈值即"驻波"；否则六量值取最大者，
≥ 形状阈值归该类；全部低于阈值则为 Normal。分类是主导形貌（同区间互斥只报最大者），所有量值仍全量导出。
形状阈值 = max(1nm, 4×σ_noise)；驻波阈值 = max(5nm, 6×σ_noise)。
v6.5 整图多标签+星级：整图（平均轮廓）分类在互斥主标签之外，另取超阈值的量值第二大者作
第二标签（最多同时两个 profile，如 Footing+Undercut）；0 个超阈值 → Normal。
星级只评主标签量值：≥3×生效阈值 ★★★、≥2× ★★、≥1× ★；生效阈值 = max(形状阈值, 10%×众数 CD)。
横幅显示"Profile: 标签+标签 ★★ · 各类柱计数"，JSON classification 增加 labels/labels_cn/rating
字段，profile_classes.csv 与 sample_index.csv 相应增加 labels/rating 列。逐柱仍为互斥单标签。
σ_noise 由图像识别结果自适应：各纳入柱宽度相邻行差分 std/√2 的中位数（剔除断行缺口对），
柱间差异与低频形貌不进入该估计；无纳入柱时阈值取下限。
LER：中间 15~85% 侧壁 x(y) 去线性趋势后的 3σ，左右分别计算；LWR 对同一高度段的 R(y)-L(y) 去线性趋势后计算 3σ。
粗糙度主比较取逐柱原始采样测量值的均值，另存平均坐标轮廓的粗糙度，避免平均时互相抵消。
周期起伏：中间侧壁去二次趋势、Hann 窗频谱找至少 2 周期且每周期至少 4 个采样点的主峰；主峰邻域能量占比≥45% 后拟合正弦幅值。
周期起伏是图像中的周期成分，不等同于已证实的光学驻波；无显著周期时幅值为 0，周期留空。
形貌出现数量以量值≥形状阈值计数（JSON 字段沿用 above_1nm 旧名），仅作统一统计阈值；所有原始数值（含 0）均导出。
检测触及图像/ROI 边界的残柱仍显示逐柱值和截断提示，但不参与平均形貌 baseline；无完整柱时禁止比较。
比例尺来自用户输入，跨图比较可分别设置 nm/px；滤波、像素尺寸和测量长度会影响粗糙度，结果未作噪声去偏。
Δ=样本-baseline；百分比=Δ/abs(baseline)*100；baseline 为 0 时百分比留空，不显示无穷大。
"""

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


# =====================================================================
# GUI 可见参数
# =====================================================================

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


# =====================================================================
# 比例尺校验与每图独立标定
# =====================================================================

import json  # noqa: E402
import math  # noqa: E402


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


# =====================================================================
# 帮助文本（导出报告与 GUI 共用部分）
# =====================================================================

FORMULA_HELP = """## 1  符号、范围与单位
本文描述 v6.4 实际实现。几何分区为程序定义的工程指标，各项可以同时出现。
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
定义 W(z)=R(z)−L(z)。基准线为该柱（或平均轮廓）的众数 CD：
bin 宽 b=max(0.5nm, 5%×(P95−P10))，取宽度直方图最高 bin 中心；
样本数<5、IQR≈0（恒定值）或众数峰计数<1.5×中位 bin 计数（平坦分布）时回退中位数。
Q90(v)：排序 v_(0)…v_(n−1)，h=0.9(n−1)，k=floor(h)，t=h−k；
Q90=(1−t)v_(k)+t·v_(min(k+1,n−1))。先求百分位，再与0取较大值；不是先逐点删掉负值。

T-top 顶部展宽量：max(0,Q90({W(z)−CD众数 : z≤0.18}))。
Footing 底部展宽量：max(0,Q90({W(z)−CD众数 : z≥0.82}))。
Undercut 底部收缩量：max(0,Q90({CD众数−W(z) : z≥0.82}))。
中间粗：max(0,Q90({W(z)−CD众数 : 0.35≤z≤0.65}))；中间细：max(0,Q90({CD众数−W(z) : 0.35≤z≤0.65}))。
Top-rounding 顶部收窄量 D：max(0,Q90({CD众数−W(z) : z≤0.20}))；
等效圆角半径取 z≤0.20 的 d_i=max(0,CD众数−W_i)，以实际 Y 作梯形积分：
A=Σ_i (d_i+d_(i+1))(Y_(i+1)−Y_i)/2，只积分选中顶部采样点之间的区间。
r_eq=sqrt(max(0,A)/(2(1−π/4)))。两侧相等理想圆角的缺失面积各为 r²(1−π/4)，据此换算。
这是面积等效值，非圆弧拟合结果；严重 T-top、倾斜或遮挡时需配合轮廓判断。
六类形貌做互斥单标签分类：六量值取最大者≥max(形状阈值, 10%×众数 CD) 归该类，全部低于阈值则 Normal；
Undercut 与 Footing 同区间互斥，两者均超阈值时只报量值大者（主导形貌）。
各项量值是左右边界合计的宽度差，不是单侧突出距离。

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
  xx_tool_result.png            综合结果图（主图 + 下方列投影 + 右侧逐柱投影）；
                                顶部含整图像貌分类横幅（Profile: 类名 · 各类柱计数），
                                中文依赖系统字体，缺失时降级为英文
  xx_tool_result_nm.png         同上，但坐标轴按 nm 标注
  xx_tool_boundaries.txt        逐行边界（px；末尾附识别模式/CD范围注释）
  xx_tool_boundaries_nm.txt     逐行边界（nm；末尾附识别模式/CD范围注释）
  xx_tool_stats_nm.txt          尺寸、形貌、比例尺与识别参数统计表（nm）
  xx_tool_morphology.json       平均坐标、各柱形貌、柱间离散与统计定义（阈值含 σ 自适应上限；含质量预警）

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

Baseline 文件还包含：sample_index.csv（B0/S1…编号、完整路径、模式、比例尺与互斥分类）、
comparison_values.csv / comparison_deltas.csv / comparison_percent.csv（横向指标矩阵）、
profile_classes.csv（每款样本互斥分类与六类形貌量值）、profile_class_table.png（量化对照表）、
per_pillar_morphology.csv（所有逐柱数据及排除原因）、per_sample/Sx_*（每款 PR 单独的指标、
逐柱 CSV 与增减图）。同名图片只要路径或 ROI 不同，编号也不同，不会合并。

每次识别的文本输出都会附带：识别模式、候选版本（含亮/暗极性）、
置信度（候选排序分数，非百分比）、候选尝试次数、估计 Line CD/允许误差、
用户先验范围、候选保护范围、完整主体柱/全部柱/边缘残柱观测范围。
Line CD 与统计 CD 均按左右边界距离输出。
"""


ABOUT_HELP = """PR profile analyzer v6.4
原作者：E924744 Kang An

binary 粗轮廓 + 有符号 Scharr + 动态规划 + 亚像素定位。
支持整图和 ROI、亮/暗柱模式、逐行/逐列灰度曲线、独立比例尺及 Baseline 对比。
过程图含最终选中候选、识别点坐标、证据标记与全部灰度曲线。
采用可解释的传统视觉算法，无需模型下载。自动明暗不等于语义识别。
算法、指标公式和输出说明见本程序帮助菜单。
"""


QUICKSTART_HELP = """## 1  选图与标定
点击“选择图片”或“浏览文件夹”。输入这张图的标尺像素长度与实际 nm 长度，例如 100 px = 50 nm。
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
“全部指标与逐柱统计”默认每个样本一列、每项指标一行，共15项数值。
“横向数值”可切换实测值(nm)、增减Δ(nm)、变化率(%)；B0列始终展示原基准值(nm)。
“形貌对比”视图给出每款样本的互斥分类、六类形貌量值Δ及较基准主导项，
与基准分类一致显示“与基准一致”，否则显示“基准类 → 样本类”。
点击样本列可在下方查看其完整名称、路径、区域和形貌分类；列多时用表底横向滚动条。
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
sample_index.csv：编号、名称、原图完整路径、区域、比例尺、纳入/排除数与互斥分类。
comparison_values/deltas/percent.csv：三张横向表，每个样本一列。
baseline_comparison.csv/json：全指标长表与完整报告；average_coordinates.csv：平均坐标。
per_pillar_morphology.csv：全部逐柱数值；per_sample/Sx_*：各样本独立的指标表、逐柱表和对比图。
profile_classes.csv：每款样本互斥分类、阈值与六类形貌量值；profile_class_table.png：量化对照表。
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
