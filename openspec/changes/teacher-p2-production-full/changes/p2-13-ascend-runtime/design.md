# p2-13 · design（910B 移植主线）

## 技术判断（C1 证据链）
- 代系卡点=头而非通路：bisheng 对 910B（dav-2201）生成的 `asc_copy_gm2l1_nd2nz/ascend_gemm_l1<...>` 等调用在 C1 崩溃源码中完整出现且属 910B 兼容面；仅 `c_api/asc_simd.h`+`simt_api/*`（950 SIMD/SIMT 指令）缺失。故**头桩 MVP**：fork `src/tl_templates/ascend/common.h` 等六处 include，做 910B 分支（空桩或 910B vec API 等价），AIC-only 用例先通。
- 七算子分层：gemm/dW/letter_readout/add_ln 纯 AIC 或 elementwise-AIC → 首批移植；rope/attn_sw/GDN 含 softmax/非线性 → 依赖 vector 通路（910B 旧 vector API），第二批或临时 torch 混合（同栈混合先例已在册）。
- 路线A 参照：950 实例一旦可得，同批用例双跑，差异即"代系税"实证（研究报告素材）。

## 文件清单
`ascend/kernels/`（不动）+ fork 分支 `tilelang@sys1-910b`（模板层条件编译）+ `ascend/port910b/`（头桩与移植件）

## 风险与回退
- 头桩实验失败（AIC 通路也隐式依赖新头）→ 界定最小改造集，如实入档，路线A 权重上调
- 950 实例不可得 → 参照系改用 TileKernels 官方基准数字（卡面）+ 本地 CPU 对拍
- vehicle 永远可跑：torch_npu 底座已验，研究不阻塞训练实验
