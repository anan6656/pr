# pr7.0 — PR profile analyzer 代码目录

`pr7.0/` 根目录始终保存**当前最新**的代码，本工作空间每完成一次改动即整体覆盖上传；
每次更新同时在 `pr7.0/` 下新增一个小版本文件夹（v7.0.1、v7.1.0 …）保存该次快照，
便于回退与对比。

| 文件 | 职责 |
|---|---|
| `pr_1_config.py` | 全局配置：常量、形貌指标定义、阈值、比例尺读写；`parse_ground_truth` / `param_overrides` |
| `pr_2_image_recognition.py` | 图像识别管线：预处理、柱检测、边界精修；`process_image` / `analyze_image` |
| `pr_3_contour_calc.py` | 轮廓量化：六类形貌量值、`build_morphology`、分类与星级、结果图渲染 |
| `pr_4_contour_compare.py` | Baseline 对比图与报告导出 |
| `pr_5_gui.py` | GUI 主界面（含直线量测实时灰度曲线、Baseline 三联图） |
| `pr_6_main.py` | 主入口：启动 GUI / `--build-exe` 离线打包 |
| `pr_8_profile_validation.py` | 测试工具：标注集验证、参数扫描、报告（生产逻辑在 pr_1~pr_3） |
| `grey_text.py` | 灰度文本 ↔ TIF 互转（`--batch` 多文件转子文件夹、`--depth`、`--verify`） |
| `PR_profile_判定量化标准.md` | 判定量化标准说明（标准 CD、六类量值、Top-rounding 口径、阈值） |

## v7.1.0（2026-10-21）

**版本号 7.1.0**，打包产物 `PR_profile_analyzer_v7.1.0.exe`。

- **删除周期起伏/驻波**：移除 `_periodic_edge_metrics`（侧壁频谱拟合）、
  wave_amplitude_nm/wave_period_nm 量值、驻波分类、MORPH_WAVE_* 阈值；
  输出指标 = 六类形貌 + LER/LWR 3σ。
- **识别算法迭代**（数值化评估驱动：边界梯度对齐度 align、顶/底边界面证据
  topR/botR、柱宽/柱高一致性、断裂率 gap）：
  - 粘连宽柱按**纵向暗缝拆分**（A671318-01M-2 柱宽 CV 0.462→0.249）；
  - 被切碎的窄条碎片柱合并回邻柱（断行填补）；
  - 无效结构剔除：柱高 > 40% 图高的"柱"视为衬底/通孔长结构，
    全剔后保留最合理的一根——**A575082 via 图由 9 根收敛为 1 个孔**；
  - 柱底共识延伸：把高于同伴的柱底延到共识底边（自带灰度证据校验）。
- 指标变化：柱宽 CV 均值 0.080→0.055，align 0.767→0.761（基本持平），
  botR 1.62→1.67，总柱数 161→155；**grey_images 标注集命中率保持 8/10、主标签 7/10**。
- 星级与"最多同时两种异常形貌"横幅：`Profile: Footing+Undercut ★★ · 各柱计数`。

## v7.0.1（2026-10-21）

- 版本号 7.0.1；补齐 Matplotlib/Pillow 依赖后重新打包，恢复**直线量测实时灰度曲线**
  与 **Baseline 三联图**（左轮廓 / 中 CD-高度 / 右六类+LER·LWR 增减柱状图，可切表格）。
- 横幅文字自适应缩字号（20→11），修复 "Normal×2" 截断显示为 "Norma2"。

## v6.6（2026-10-21 凌晨）

Top-rounding 深度口径（单侧亏缺深度）、分类别 CD 门槛（top_rounding 0.30×CD、
footing 0.12×CD）、结果图横幅贯通修复、pr_8 拆分重构。
