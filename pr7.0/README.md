# pr7.0 — PR profile analyzer 代码目录

`pr7.0/` 根目录始终保存**当前最新**的代码，本工作空间每完成一次改动即整体覆盖上传；
每次更新同时在 `pr7.0/` 下新增一个小版本文件夹（v7.0.1、v7.0.2 …）保存该次快照，
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

## v7.0.1（2026-10-21）

**软件版本号由 6.5 升为 7.0.1**（`APP_VERSION`），打包产物
`PR_profile_analyzer_v7.0.1.exe`（含 Matplotlib + Pillow 的完整版）。

- **恢复直线量测实时灰度曲线**：此前打包环境缺 Matplotlib，GUI 的量测曲线面板、
  Baseline 对比图整体未创建；现补齐依赖并重新打包，`dist/` 内 v7.0.1 EXE 为完整版。
- **恢复 Baseline 对比三联图**：左=平均轮廓叠加，中=CD 随高度变化，右=六类形貌
  量值 + LER/LWR 增减横向柱状图（可切换表格视图），量化与单图判定同源。
- **横幅文字自适应**：分类横幅超宽时自动逐级缩小字号（20→11），仍超宽才省略计数，
  修复 "Normal×2" 显示成 "Norma2" 的截断（根因之一是无 Pillow 时 cv2 降级丢弃
  "×" 与中文）。
- 判定量化（v6.6 起）：Top-rounding 主量 = 顶部圆角亏缺深度（单侧 nm），
  新增 `top_rounding_pct`；分类别门槛 top_rounding 0.30×CD、footing 0.12×CD；
  标准 CD = 柱宽众数。grey_images 标注集 hit 8/10、主标签 7/10。
- 识别层：柱状垂直延续性主段选择、柱顶向上扩展、StopIteration 崩溃修复。
- pr_8 拆分：标注解析 / `analyze_image` / `param_overrides` 并入 pr_1~pr_3，pr_8 只留测试。

## v6.6（2026-10-21 凌晨）

首版 pr7.0 上传：Top-rounding 深度口径、分类别阈值标定、结果图横幅贯通修复、
pr_8 拆分重构、matplotlib 降级打包（当时缺依赖，功能不完整，已被 v7.0.1 取代）。
