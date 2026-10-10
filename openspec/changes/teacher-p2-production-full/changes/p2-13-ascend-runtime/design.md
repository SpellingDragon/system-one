# p2-13 · design（910B 移植主线）

## 技术判断（C1 证据链）

**2026-10-07 P0-1 回写（原判"卡点=头而非通路"过乐观，据 349 符号实勘改判）**：① 移植主体是 **codegen 向量侧压在 950 SIMT/SIMD 方言上**（C 类 250/72%）；本仓七件 kernel 全部以 T.SimtVF 为唯一向量载体（8/8 文件实证），910B 无 SIMT 硬件模型 → 算子**改道**（SIMD 化/标量化）而非加头。② AIC(cube) 通路 API 族（asc_init/asc_mmad/l12l0 系 14 符号，**asc_init 由 codegen 无条件发射每 kernel 首行**）静态不可判 = P0-2 vecadd 生死线探针（probe910b.sh P1-P10 已备好）。③ B 类 20（类型代理）已离线闭环（host_selfcheck 三配置 PASS、20 万样本 0 mismatch）。
- 代系卡点=头而非通路：bisheng 对 910B（dav-2201）生成的 `asc_copy_gm2l1_nd2nz/ascend_gemm_l1<...>` 等调用在 C1 崩溃源码中完整出现且属 910B 兼容面；仅 `c_api/asc_simd.h`+`simt_api/*`（950 SIMD/SIMT 指令）缺失。故**头桩 MVP（原地制，不建分支）**：纯新增 `src/c_api/asc_simd.h`+`src/simt_api/*.h` 五桩（bisheng 的 -I 扫 src/ 即命中，零改上游文件），AIC-only 用例先通、缺失符号随验随补。
- 七算子分层（P0-1 修正）：**无一纯 AIC**（全含 SimtVF 向量面）→ 首批=P0-2 判定 AIC 通路可用后，以 readout/gemm/dW/add_ln（向量面最薄、可标量化）先行改道；rope/attn_sw/GDN 向量面厚，依赖 SIMD 改道立项。
- 路线A 参照：950 实例一旦可得，同批用例双跑，差异即"代系税"实证（研究报告素材）。

## 文件清单
`ascend/kernels/`（接口不动）+ 主仓工作区原地新增（五桩已落 `src/c_api/`+`src/simt_api/`）+ `ascend/port910b/`（移植件与远端 pip 包同步脚本）

## 风险与回退
- 头桩实验失败（AIC 通路也隐式依赖新头）→ 界定最小改造集，如实入档，路线A 权重上调
- 950 实例不可得 → 参照系改用 TileKernels 官方基准数字（卡面）+ 本地 CPU 对拍
- vehicle 永远可跑：torch_npu 底座已验，研究不阻塞训练实验

## 主战场本地化（2026-10-08，三轮上卡摩擦税所促）
- **关键事实**：bisheng 将 .asc 编译为 .o 属交叉编译，只需 CANN 工具链不需 NPU 在线；CANN 8.5.2 有 aarch64-Linux 公开版，Mac Docker 原生 aarch64 可跑 → **编译判决（方言可用性、头/类型/intrinsic 缺失、vecadd 生死）全部本地零边际成本迭代**；卡窗口仅保留运行时验证（数值/tflops，≤10min）。
- **制品化纪律**：上卡动作一律单文件自包含 bundle（幂等：装依赖→铺树→跑→输出 verdict JSON），禁止临场拼命令链。


## 真机数值首战判读（2026-10-10，run p2-13-oncard-wave1）
- **D-num1**：向量面（标量化+表加载+软件数学件）真机数值全绿（1e-7~1e-8 级）→ P1-1 向量侧改道方案**数值层闭合**。
- **D-num2**：cube 面 gemm_l1/dW 同型 aicore 507015（非法访问型崩溃）→ V1/V2（load_cbuf_to_ca 装填位段）从"数值待证"升格为"致命缺陷"；修复路径=官方 mm 链构造点逐参对照（P1-1d），本地修完单窗复验——禁止卡上试错（R20）。
- **D-num3**：gemm direct（UB 直读 mad）rel=nan 维持 backlog（l1 路修好后无生产必要性）。
