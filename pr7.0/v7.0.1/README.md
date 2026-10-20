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
| `pr_5_gui.py` | GUI 主界面 |
| `pr_6_main.py` | 主入口：启动 GUI / `--build-exe` 离线打包 |
| `pr_8_profile_validation.py` | 测试工具：标注集验证、参数扫描、报告（生产逻辑在 pr_1~pr_3） |
| `grey_text.py` | 灰度文本 ↔ TIF 互转（`--batch` 多文件转子文件夹、`--depth`、`--verify`） |
| `PR_profile_判定量化标准.md` | 判定量化标准说明（标准 CD、六类量值、Top-rounding 口径、阈值） |

## v7.0.1（2026-10-21）

- Top-rounding 主量改为**顶部圆角亏缺深度（单侧）**，新增 `top_rounding_pct`（占 CD %）与
  辅助的等效半径 `top_rounding_r_nm`（旧半径口径由面积开方换算，会把 2nm 亏缺放大到 5nm 量级）。
- 分类别 CD 比例门槛 `MORPH_SHAPE_CD_RATIO_BY_KEY`：top_rounding 0.30×CD、footing 0.12×CD；
  标准 CD 仍为柱宽众数。grey_images 标注集 hit 8/10、主标签 7/10。
- 识别层：主段选择加入柱状垂直延续性证据（IoU），修复粗糙颗粒基底被当成柱的问题；
  主段顶部向上扩展；修复超大柱重跑精修的 StopIteration 崩溃。
- 结果图：分类横幅贯通整幅（原右侧面板顶部留白造成"上方缺一块"）、坐标轴与配色美化。
- pr_8 通用能力并入主干（标注解析 / `analyze_image` / `param_overrides`），pr_8 只留测试逻辑。
- 打包：matplotlib 降级为可选依赖，离线打包出 `PR_profile_analyzer_v6.5.exe`。
