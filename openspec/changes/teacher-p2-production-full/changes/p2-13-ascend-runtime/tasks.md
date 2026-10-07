# p2-13 · tasks（910B 移植主线，2026-10-07 重排）

- [x] R3 【C1 已决】910B4 实勘：方言 950-only（三重证据）、torch_npu 978 tok/s、s3fs 运维风险 —— 验证：run de15
- [ ] B1 **头桩 MVP 实验**（首孙任务·风险前置）：fork 主仓 ascend 模板层建 `sys1-910b` 分支，以空桩/910B 等价物替换 c_api+simt_api 六处 include，AIC-only vecadd→gemm 两级用例在 910B 实编实跑 —— 验证：远端 run notes 含"910B 头桩编译通过/失败"结论行与 bisheng 命令原文
- [ ] B2 gemm/dW/letter_readout/add_ln 四件 AIC 移植 + fp32 对拍（远端 NPU 实跑） —— 验证：gradcheck NPU 档 exit 0
- [ ] B3 rope/attn_sw/GDN vector 通路移植（或同栈混合如实标注） —— 验证：同上
- [ ] A1 950/A3 实例可得性调查（ModelArts 镜像面/专属池申请路径）+ 可得则同批双跑 —— 验证：调查结论行入 run notes
- [ ] R6 cost_ledger 接入 B/A 各段（沿用 D12 熔断） —— 验证：ledger 行齐
- [ ] R7 C4 tiny 冒烟（torch_npu 底座）+ gate 精确报价 —— 验证：报价行 + 熔断判定
