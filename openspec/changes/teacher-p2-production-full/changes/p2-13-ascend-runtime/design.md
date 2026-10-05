# p2-13 · design

## 技术要点
- **探针先行（C1，风险前置）**：① tilelang ascend dialect 在 910B 编译跑通（TileKernels 标注 950，910B 未验——最大单点）；② TileKernels 昂腾后端自动切换是否在 910B 生效；③ torch_npu+transformers 前向作参照系与回退底座；④ bf16/fp16 数值行为。任一失败即按 D6 回退决策记录。
- **七算子清单**（与 P1 Metal 方言件同构，接口同名）：gemm（含 dW 反向，参考 P1 `gemm_bwd_dw`）/rope（负角反向）/attn_sw（滑窗，1M 三件套复用）/GDN（Gated DeltaRule 前向+反向——TileKernels `engram` gating 范式参考，最重件）/add_ln（闭式反向）/读出 letter_rows（index_select）/LoRA 注入（旁路 gemm）。每件：cpu target 语义对拍（本地）→ NPU 编译对拍（云端）→ 基准（vs torch_npu）。
- **梯度对拍互锁**：fp32 torch 参考为真值（P1 `torch_ref/` 复用），数值容差按 dtype 定；对拍不过不合入。
- **成本护栏**：`cost_ledger.py` 记每段起止/时长/费用 → run notes；C4 报价 = 实测吞吐 × 三栈 token 预算；¥600 熔断。
- 数据上云：rsync/oss 皆可，数据包在本地 W0/W1 装配完成后一次性上传。

## 文件清单
`release/ascend/{env_setup.sh,selfcheck.py,kernels/*.py,cost_ledger.py}` + `release/tests/test_ascend_gradcheck.py`

## 风险与回退
- 910B dialect 不可用 → torch_npu 底座（三栈照跑，算子故事降级为"基准表+回退记录"，G9 验收改为如实记录）
- GDN 反向移植受阻 → 该层临时走 torch 实现（同栈混合），其余算子自研——渐进式，不搞全有全无
- 云端环境搭建超时（>6h）→ 上报用户换镜像/换实例
