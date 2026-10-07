# p2-13 · tasks（甲路·2026-10-07 用户裁定：C5 暂停，GDN/方言移植为唯一主线）

- [x] R3 C1 实勘（run de15）· A1 路线A废（950 不可得）· B1 首轮五层剥四（run 7793）
- [ ] **P0-1 编译链打通（离线主战场）**：第 5 层 bisheng 类型系统兼容——离线盘点 tilelang 模板对 950 类型代理（bfloat16_t 转换算子/vec_t/内置 builtin 变量）的全部依赖点，产出 `port910b_compat.h` 兼容层（910B `__bf16` builtin 等价映射）；判据=本地静态穷举清单完备 —— 验证：清单入 run notes，覆盖模板中全部 950-only 符号（grep 举证）
- [ ] **P0-2 链通里程碑（上卡）**：vecadd 级最小 kernel 在 910B 实编实跑数值对 —— 验证：远端 run 结论行
- [ ] P1-1 gemm/dW/readout/add_ln 四件 AIC 移植+对拍 —— 验证：gradcheck NPU 档
- [ ] P1-2 rope/attn_sw 移植 —— 验证：同上
- [ ] **P1-3 GDN 件移植（本项目旗舰）**：短卷积+delta rule 的 910B 表达（conv 用 AIC 滑窗累加替代或 910B vector API）+前后向 —— 验证：fp32 对拍 + 对 torch 参考数值（partial 分步验证可入账）
- [ ] P2 三栈联调（torch_npu+自研件混合栈，GDN 件替换Conv2D路径）—— 验证：0.8B 训练步跑通
- [ ] R6/R7 成本记账与 gate 报价（恢复 C5 的前置）—— 验证：报价行+熔断判定

## 挂起（甲路裁定）
- p2-05/06/11 的 C5 云端正式训练：**全部挂起**，解锁条件=P2 行（三栈联调）达成
