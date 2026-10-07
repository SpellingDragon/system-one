# p2-13 · tasks（910B 移植主线，2026-10-07 重排）

- [x] R3 【C1 已决】910B4 实勘：方言 950-only（三重证据）、torch_npu 978 tok/s、s3fs 运维风险 —— 验证：run de15
- [ ] B1 **头桩 MVP 实验**（首孙任务；本地已落子：主仓纯新增五桩零改上游，见 `src/c_api/`+`src/simt_api/`）：同步桩至远端 pip 包后 AIC-only vecadd→gemm 两级用例在 910B（snt9b 镜像）实编实跑 —— 验证：远端 run notes 含"头桩编译通过/失败"结论行与 bisheng 命令原文；失败则列出缺失符号清单（桩内容随验随补）
- [ ] B2 gemm/dW/letter_readout/add_ln 四件 AIC 移植 + fp32 对拍（远端 NPU 实跑） —— 验证：gradcheck NPU 档 exit 0
- [ ] B3 rope/attn_sw/GDN vector 通路移植（或同栈混合如实标注） —— 验证：同上
- [x] A1 【已裁定】950 不可得（用户 2026-10-07）；参照系=TileKernels 卡面 + CPU 对拍；环境锚定 snt9b 镜像 —— 验证：本行 + proposal 路线A条
- [ ] R6 cost_ledger 接入 B/A 各段（沿用 D12 熔断） —— 验证：ledger 行齐
- [ ] R7 C4 tiny 冒烟（torch_npu 底座）+ gate 精确报价 —— 验证：报价行 + 熔断判定
