# p2-13 · 昇腾运行时与自研算子栈（ascend-runtime）

## Why
用户令（2026-10-05）：训练与推理优先完全使用昇腾卡（云端 910B ¥20/h），算子充分参考 TileKernels；训练栈主路线=全 TileLang 自研（D6/D11–D13）。本域承载：环境、方言探针、七类算子移植、梯度对拍、性能基准、成本护栏——**G9 验收面**与三栈（5/6/11）的公共底座。

## What Changes
- 新增 `release/ascend/`：环境脚本（CANN/torch_npu/tilelang 版本 pin）+ 910B 开箱自检
- 新增七类算子昇腾方言件（linear/rope/attn/GDN/LN/lm_head 读出/LoRA 注入），参考 TileKernels（`modeling` autograd 封装范式、`transform` RoPE、量化件备查）与 P1 `sys1/kernels/` 结构
- 新增 `tests/test_ascend_gradcheck.py`：fp32 参考互锁梯度对拍（cpu target 本地 + NPU 云端两跑）
- 新增基准与成本护栏：自研栈 vs torch_npu 底座 A/B 表；`ascend/cost_ledger.py` 分段记账（D12）
- **回退决策点**：C1 探针不可用→torch_npu 底座（决策+原因入 run notes，三栈实验不受阻）

## 边界与依赖
- 依赖：tilelang（主仓 ascend dialect）、TileKernels（参考实现，950 标注需 910B 实证）、C1 云端实例
- 被依赖方：p2-05/06/11（训练栈底座）、p2-10（NPU 推理后端）、p2-03（harness NPU 口）
- 接口面：`ascend/kernels.py` 暴露与 P1 `sys1/kernels/` 同名接口（前向+backward），训练侧零改动切换
- 禁止事项：不碰 `decision/`；不自研已在 TileKernels 验证过的非热点算子（先复用后自研）
