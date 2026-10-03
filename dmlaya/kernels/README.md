# dmlaya/kernels/ — TileLang 跨后端算子（一阶段 M1 主战场）

> 目的：决策模型的**计算主体**用 TileLang 手写，并以三件套范式跑通 **CUDA(参考) / MPS / 昇腾**。数值对拍是验收生命线。

## 三件套约定（源自 TileKernels，扩展 `_mps`）

```
<op>_kernel.py   # torch 面向外入口：校验/分配输出/按 backends.py 分发。不写 TileLang。
<op>_cuda.py     # 参考实现（CUDA 方言 tilelang.language）
<op>_mps.py      # Metal 方言（tilelang.metal.language：simdgroup/cooperative tensor）
<op>_asc.py      # 昇腾方言（tilelang.ascend.language：AIC/AIV、T.Stage、SIMD S.*）
```
参考：`refs/TileKernels/tile_kernels/moe/topk_gate_{kernel,cuda,asc}.py`（同算法双范式的样板）。

## 算子清单与要点

| 算子 | 说明 | 直接参考（refs/） |
|---|---|---|
| `gemm_*` | `C=act(A@Wᵀ+b)`，fp32 累加、`transpose_B`、`M` 动态 | **`laya/laya/tl_kernels.py::gemm_kernel`**（改 dtype/参数即可）；`tilelang/examples/gemm/` |
| `layernorm_*` | LN+残差融合，**残差流保 fp32**（laya 实测 bf16 残差会漂移） | `tl_kernels.py::add_ln_kernel`；注意 laya 用 LayerNorm 而非 RMSNorm |
| `rope_*` | packed-qkv 就地旋转（rotate-half），cos/sin 分表 | `tl_kernels.py::rope_kernel` |
| `attn_sw_*` | **因果滑窗** flash-attention（窗口 W：仅回看 W 个 key） | ⚠️ `tl_kernels.py::attn_kernel` 是**双向**滑窗——掩码改因果即为其一半工作量，方向别抄反 |
| `readout_gemm_*`（可选） | 末位 hidden × 26 字母行的小矩阵乘，可先行 torch 回退 | decision/README |

`../backends.py`：探测顺序 `cuda → mps → npu`（昇腾判据示例：`os.path.exists('/dev/davinci_manager')`，参考 `refs/TileKernels/tile_kernels/config.py`），并收敛设备能力（核数/显存/dtype 白名单：MPS bf16 受限→fp16 回退）。

## 对拍纪律（进 `tests/`，按设备 skip）

- 先写 `testing/torch_ref/<op>_ref.py`，再写 kernel；逐算子对拍。
- bf16/fp16 下 `max|err| ≤ 2e-2`，且**决策 argmax 100% 一致**（M1 验收）。
- `@tilelang.jit` 一次编译多 batch 复用：`M` 为动态符号；序列长 pad 到固定 ladder（线性核按长度编译，参考 StartLux `PAD_LENGTHS` 思路）。
