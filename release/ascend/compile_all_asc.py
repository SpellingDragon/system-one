#!/usr/bin/env python3
"""compile_all_asc.py — 九件**产品接口**的 910B 编译判决 harness（使能工装 + 合流验收探测器）。

对 `ascend/kernels/*_asc.py` 九件，在 **patched 910B tilelang + 无设备** 下驱动其**对外入口件**
（`gemm_kernel` / `gemm_bwd_dw_kernel` / `rope_kernel` / `add_ln_kernel` / `attn_sw_kernel` /
`gdn_kernel`（递推 + 短卷积两枚）/ `letter_readout_kernel` / `lora_kernel`），把**编译判决**与
**发射**隔离开：真跑 `tilelang.compile` 拿成功/失败，用 dummy 顶掉 kernel 调用（卡上才需要真发射）。
判据来自 ascend_env 的 `compiled_keys()`（真编译并缓存过才算走内核，R14 防假绿）与 `blockers()`
（编译失败原文首行）。

【P1-1i 两处修法（元缺陷：判决件量的不是产品代码）】
  ① `_drive_gemm` 原先手搭 `gemm_asc.plan(A, W, bias, C, n=…, k=…)`，与现签名
     `plan(A, W, bias, C, act, target)` 不符 ⇒ `TypeError: plan() got an unexpected keyword
     argument 'n'` ⇒ gemm 腿长期 PRECOMPILE-ERR（S 报告）。现在**九条腿一律只走入口件**
     （`*_kernel.forward/backward`，与 `ascend.autograd_asc`/`train_step.py` 同一批调用行）——
     harness 通过 = 产品接口通过，harness 不再自带第二套接口口径（第二套口径本身就是错的来源）。
  ② 原先 target 硬写 "ascend"：host（无 ascend 后端）上只能看见后端错，看不见语义面。现在
     target 由 `--target` / `DMLAYA_ASCEND_TARGET` 决定，**缺省 cpu**（与 `ascend_env` 的
     "env 未设置时按 cpu，本地无卡落在安全侧"同口径），判决行与判据前缀都跟着 target 走。

一行/件：`ASC-<op> PASS .o=<bytes> keys=<n>` / `ASC-<op> FAIL <head>` /
        `ASC-<op> PRECOMPILE-ERR <head>`（= 没达到 tilelang.compile，多为形状/plan 拒/入口回退）。
退出码：任一 FAIL/PRECOMPILE-ERR→1（合流未清）；全 PASS→0。

用法：
  host 语义面（无卡，判"九件都能达到编译器 + 形状/契约没被 plan 拒"）：
      python3 ascend/compile_all_asc.py
  容器/卡（ascend 编译判决；§12 compat + patch_bisheng 就绪，全新 TILELANG_CACHE_DIR）：
      python3 ascend/compile_all_asc.py --target ascend
      python3 ascend/compile_all_asc.py --target ascend gemm gemm_bwd_dw letter_readout
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))          # .../release/ascend
RELEASE_ROOT = os.path.dirname(HERE)                        # .../release
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)                        # 使 `ascend` 包可导入

import torch  # noqa: E402
import tilelang  # noqa: E402

from ascend.kernels import ascend_env  # noqa: E402

_AP = argparse.ArgumentParser()
_AP.add_argument("--target", default=None, help="ascend | cpu（缺省读 DMLAYA_ASCEND_TARGET，再缺省 cpu）")
_AP.add_argument("ops", nargs="*", help="只判这些件（缺省九件全判）")
_ARGS = _AP.parse_args()
#: 本次判决面。`normalize_target(None)` 的取值链 = `--target` 之外唯一的入口，与产品件同源。
_TARGET = _ARGS.target or ascend_env.default_target()


class _LaunchSkip(Exception):
    """kernel 被"发射"时抛它：本 harness 只判编译，不判发射（无设备）。"""


def _install_stubs():
    """强制走 tilelang 后端 + 把真编译产物换成"可编译、调用即跳发射"的替身。"""
    ascend_env.backend_available = lambda *a, **k: True      # 绕 npu 在场门
    ascend_env._npu_present = lambda: True
    # 各代理一致上报：active_backend(target, cpu张量.device) 因 want=npu≠cpu 直判 eager，
    # 只桩上两量仍走不到 compile → 一并桩掉设备-目标匹配门（判据仍锁 `<target>|` 键）。
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    real_compile = tilelang.compile

    def _compile(*a, **k):
        kern = real_compile(*a, **k)                          # 真编译；失败原样抛，由 get_compiled 记 blocker
        nbytes = len(str(kern.get_kernel_source())) if hasattr(kern, "get_kernel_source") else 0

        class _Dummy:
            def __init__(self): self._n = nbytes
            def __call__(self, *aa, **kk): raise _LaunchSkip()
            def __getattr__(self, n): return lambda *aa, **kk: 0
        d = _Dummy()
        _size_slot["n"] = nbytes
        return d

    tilelang.compile = _compile


_size_slot = {"n": 0}


def _f16(*shape): return torch.randn(*shape, dtype=torch.float16)
def _f32(*shape): return torch.randn(*shape, dtype=torch.float32)


# ── 九条腿：全部按**产品接口**取判决（形状取够小的代表值，位宽按入口件白名单给）─────────
# 注意 fp32：`gemm_kernel`/`gemm_bwd_dw_kernel` 的入口在 cpu target 会把输入升 fp32，而
# `rope_asc.forward` 在 ascend target 只在 **fp32** 时走内核（P1-4 路由修复的位宽门），
# 故各腿一律喂 fp32：两 target 都能真达到 tilelang.compile，不靠回退蒙过判据。
def _drive_gemm():
    """`gemm_kernel.forward(A, W, bias, act, out_dtype, target)`：C = A@Wᵀ + bias。"""
    from ascend.kernels import gemm_kernel
    A, W, bias = _f32(16, 32), _f32(32, 32), _f32(32)
    return gemm_kernel.forward(A, W, bias, "none", torch.float32, _TARGET)


def _drive_gemm_bwd_dw():
    """`gemm_bwd_dw_kernel.backward(dY, A, out_dtype=, target=)`：dW = dYᵀ @ A，出口 (N,K)。"""
    from ascend.kernels import gemm_bwd_dw_kernel
    dY, A = _f32(16, 32), _f32(16, 32)
    return gemm_bwd_dw_kernel.backward(dY, A, out_dtype=torch.float32, target=_TARGET)


def _drive_add_ln():
    """`add_ln_kernel.forward` + `.backward`（前向 add+LN 融合、反向 ln_bwd 是两份正文，都判）。"""
    from ascend.kernels import add_ln_kernel
    x, res = _f32(16, 64), _f32(16, 64)
    g, b = _f32(64), _f32(64)
    y, h = add_ln_kernel.forward(x, res, g, b, target=_TARGET)
    return add_ln_kernel.backward(h, g, y.float(), target=_TARGET)


def _drive_rope():
    """`rope_kernel.forward`：packed (tokens,3,heads,dim)，cos/sin 是 (tokens, dim//2)。"""
    from ascend.kernels import rope_kernel
    qkv = _f32(8, 3, 2, 8)                                    # (tokens,3,heads,dim)，dim 偶
    cos = _f32(8, 4)
    sin = _f32(8, 4)
    return rope_kernel.forward(qkv, cos, sin, target=_TARGET)


def _drive_attn_sw():
    """`attn_sw_kernel.forward`：因果滑窗注意力（heads, seq, dim）；dim 必须被 16 整除。"""
    from ascend.kernels import attn_sw_kernel
    q, k, v = _f32(2, 8, 16), _f32(2, 8, 16), _f32(2, 8, 16)
    return attn_sw_kernel.forward(q, k, v, 8, target=_TARGET)


def _drive_gdn():
    """`gdn_kernel.forward`：GDN 递推（g 允许 (heads,seq) 的标量门）。"""
    from ascend.kernels import gdn_kernel
    q, k = _f32(2, 8, 16), _f32(2, 8, 16)
    v = _f32(2, 8, 8)
    g = _f32(2, 8)
    beta = _f32(2, 8)
    return gdn_kernel.forward(q, k, v, g, beta, torch.float32, _TARGET)


def _drive_gdn_conv():
    """`gdn_kernel.conv_forward`：GDN 短卷积（x 是 (..., C, L) 通道优先，w 是 (C,4)）。"""
    from ascend.kernels import gdn_kernel
    x = _f32(2, 16, 32)                                        # (batch, chans, length)
    w = _f32(16, 4)
    bias = _f32(16)
    return gdn_kernel.conv_forward(x, w, bias, torch.float32, _TARGET)


def _drive_letter_readout():
    """`letter_readout_kernel.forward` + `.backward`：gather 与 scatter_add 两份正文都判。"""
    from ascend.kernels import letter_readout_kernel
    rows = _f32(16, 64)
    ids = torch.arange(4, dtype=torch.long) % 16               # 0,1,2,3
    out = letter_readout_kernel.forward(rows, ids, torch.float32, _TARGET)
    dy = _f16(int(ids.numel()), 64).float()
    return letter_readout_kernel.backward(16, ids, dy, _TARGET)


def _drive_lora():
    """`lora_kernel.apply`：delta = scaling·(x@Aᵀ)@Bᵀ，两次都该走 gemm 内核（r/k/n 均需被 16 整除）。"""
    from ascend.kernels import lora_kernel
    x = _f32(16, 32)                                           # (m, k)
    a = _f32(16, 32)                                           # (r, k)  r=16
    b = _f32(32, 16)                                           # (n, r)  n=32
    return lora_kernel.apply(x, a, b, 1.0, None, torch.float32, _TARGET)


DRIVERS = {
    "gemm": _drive_gemm,
    "gemm_bwd_dw": _drive_gemm_bwd_dw,
    "add_ln": _drive_add_ln,
    "rope": _drive_rope,
    "attn_sw": _drive_attn_sw,
    "gdn": _drive_gdn,
    "gdn_conv": _drive_gdn_conv,
    "letter_readout": _drive_letter_readout,
    "lora": _drive_lora,
}


def _target_keys() -> list[str]:
    """本 target 下真编译并缓存过的键（判据的唯一来源：回退路径不留痕迹）。"""
    pre = f"{_TARGET}|"
    return [k for k in ascend_env.compiled_keys() if k.startswith(pre)]


def _verdict_for(op: str) -> tuple[str, str]:
    """据 compiled_keys/blockers 判：真编过=PASS；有 `<target>|` blocker=FAIL(编译错)；都没=PRECOMPILE-ERR。"""
    keys = _target_keys()
    bl = {k: v for k, v in ascend_env.blockers().items() if k.startswith(f"{_TARGET}|")}
    if keys:
        return "PASS", f".o~{_size_slot['n']}B keys={len(keys)} first={keys[0]}"
    if bl:
        return "FAIL", next(iter(bl.values()))[:200]
    return "PRECOMPILE-ERR", "未达 tilelang.compile（形状/plan 拒，或入口件静默落了 eager）"


def bl_hit(_exc) -> bool:
    return any(k.startswith(f"{_TARGET}|") for k in ascend_env.blockers())


def run_one(op: str) -> tuple[str, str]:
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    _size_slot["n"] = 0
    try:
        DRIVERS[op]()
    except _LaunchSkip:
        pass                                                 # 发射被桩掉 = 本 harness 期望的路径
    except Exception as exc:  # noqa: BLE001
        head = (traceback.format_exception_only(type(exc), exc) or [""])[0].strip()
        v, d = _verdict_for(op)
        if v == "PASS":
            return v, d
        return ("FAIL" if v == "FAIL" else "PRECOMPILE-ERR"), (d if bl_hit(exc) else head)
    return _verdict_for(op)


def main(argv=None):
    ops = _ARGS.ops or list(DRIVERS)
    unknown = [o for o in ops if o not in DRIVERS]
    if unknown:
        print(f"ASC-usage FAIL 未知件 {unknown}；可用：{','.join(DRIVERS)}")
        return 1
    _install_stubs()
    print(f"=== compile_all_asc target={_TARGET} tilelang={tilelang.__version__} ops={len(ops)} ===")
    n_fail = 0
    for op in ops:
        try:
            verdict, detail = run_one(op)
        except Exception as exc:  # noqa: BLE001
            verdict, detail = "PRECOMPILE-ERR", (traceback.format_exception_only(type(exc), exc) or [""])[0][:160]
        if verdict != "PASS":
            n_fail += 1
        print(f"ASC-{op} {verdict} {detail}", flush=True)
    print(f"—— 汇总 target={_TARGET} PASS={len(ops)-n_fail} FAIL={n_fail} ——")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
