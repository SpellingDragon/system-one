#!/usr/bin/env python3
"""compile_all_asc.py — P1-4 接口 home 的 910B 编译判决 harness（使能工装 + 合流验收探测器）。

对 `ascend/kernels/*_asc.py` 每件，在 **patched 910B tilelang + 无设备** 下驱动其
`forward/run(target="ascend")`，把**编译判决**与**发射**隔离开：真跑 tilelang.compile 拿
成功/失败，用 dummy 顶掉 kernel 调用（卡上才需要真发射）。判据来自 ascend_env 的
`compiled_keys()`（真编译并缓存过才算走内核，R14 防假绿）与 `blockers()`（编译失败原文首行）。

一行/件：`ASC-<op> PASS .o=<bytes>` / `ASC-<op> FAIL <head>` / `ASC-<op> PRECOMPILE-ERR <head>`。
退出码：任一 FAIL→1（合流未清）；全 PASS→0。
基线预期：SimtVF 件在 910B 编不出 → 全 FAIL（= 接口 home 未就绪，见 D-int1）。

用法（容器内，release 挂到 /work）：
  python3 /work/ascend/compile_all_asc.py [op ...]
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
os.environ.setdefault("DMLAYA_ASCEND_TARGET", "ascend")
HERE = os.path.dirname(os.path.abspath(__file__))          # .../release/ascend
RELEASE_ROOT = os.path.dirname(HERE)                        # .../release
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)                        # 使 `ascend` 包可导入

import torch  # noqa: E402
import tilelang  # noqa: E402

from ascend.kernels import ascend_env  # noqa: E402


class _LaunchSkip(Exception):
    """kernel 被"发射"时抛它：本 harness 只判编译，不判发射（无设备）。"""


def _install_stubs():
    """强制走 tilelang 后端 + 把真编译产物换成"可编译、调用即跳发射"的替身。"""
    ascend_env.backend_available = lambda *a, **k: True      # 绕 npu 在场门
    ascend_env._npu_present = lambda: True
    # 各代理一致上报：active_backend(target, cpu张量.device) 因 want=npu≠cpu 直判 eager，
    # 只桩上两量仍走不到 compile → 一并桩掉设备-目标匹配门（判据仍锁 compiled_keys 含 ascend|）。
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


# 每件的最小可编译驱动（形状取够小的代表值；个别件若 plan 需更贴语义的入参，合流期由 per-op agent 精修）
def _drive_gemm():
    from ascend.kernels import gemm_asc
    A, W, bias, C = _f16(16, 32), _f16(32, 32), _f32(32), _f32(16, 32)
    spec = gemm_asc.plan(A, W, bias, C, n=32, k=32, bm=16, bn=16, bk=32, act_mode=0, target="ascend")
    if spec is None:
        raise RuntimeError("plan=None")
    return gemm_asc.run(A, W, bias, C, spec)


def _drive_add_ln():
    from ascend.kernels import add_ln_asc
    return add_ln_asc.forward(_f16(16, 64), _f16(16, 64), _f32(64), _f32(64), target="ascend")


def _drive_rope():
    from ascend.kernels import rope_asc
    qkv = _f16(8, 3, 2, 8)                                    # (tokens,3,heads,dim) packed, dim 偶
    cos = _f32(8, 4); sin = _f32(8, 4)                         # (tokens, dim//2)
    return rope_asc.forward(qkv, cos, sin, target="ascend")


def _drive_attn_sw():
    from ascend.kernels import attn_sw_asc
    q, k, v = _f16(2, 8, 16), _f16(2, 8, 16), _f16(2, 8, 16)   # (heads?, seq, dim) 单头代表
    return attn_sw_asc.forward(q, k, v, window=8, target="ascend")


def _drive_gdn():
    from ascend.kernels import gdn_asc
    q, k = _f16(2, 8, 16), _f16(2, 8, 16)
    v = _f16(2, 8, 8)
    g = _f32(2, 8); beta = _f32(2, 8)
    return gdn_asc.forward(q, k, v, g, beta, target="ascend")


def _drive_letter_readout():
    from ascend.kernels import letter_readout_asc
    ids = torch.arange(4, dtype=torch.long) % 16
    return letter_readout_asc.forward(_f32(16, 64), ids, target="ascend")


def _drive_gemm_bwd_dw():
    from ascend.kernels import gemm_bwd_dw_asc
    dY, A, GW = _f16(16, 32), _f16(16, 32), _f32(32, 32)
    spec = gemm_bwd_dw_asc.plan(dY, A, GW, target="ascend")
    if spec is None:
        raise RuntimeError("plan=None")
    return gemm_bwd_dw_asc.run(dY, A, GW, spec)


DRIVERS = {
    "gemm": _drive_gemm,
    "add_ln": _drive_add_ln,
    "rope": _drive_rope,
    "attn_sw": _drive_attn_sw,
    "gdn": _drive_gdn,
    "letter_readout": _drive_letter_readout,
    "gemm_bwd_dw": _drive_gemm_bwd_dw,
}


def _verdict_for(op: str) -> tuple[str, str]:
    """据 compiled_keys/blockers 判：真编过=PASS；有 ascend| blocker=FAIL(编译错)；都没=PRECOMPILE-ERR。"""
    keys = [k for k in ascend_env.compiled_keys() if k.startswith("ascend|")]
    bl = {k: v for k, v in ascend_env.blockers().items() if k.startswith("ascend|")}
    if keys:
        return "PASS", f".o~{_size_slot['n']}B keys={keys[0]}"
    if bl:
        return "FAIL", next(iter(bl.values()))[:200]
    return "PRECOMPILE-ERR", "未达 tilelang.compile（形状/plan 拒）"


def run_one(op: str) -> tuple[str, str]:
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda *a, **k: ascend_env.TILELANG
    _size_slot["n"] = 0
    try:
        DRIVERS[op]()
    except _LaunchSkip:
        pass
    except Exception as exc:  # noqa: BLE001
        head = (traceback.format_exception_only(type(exc), exc) or [""])[0].strip()
        v, d = _verdict_for(op)
        if v == "PASS":
            return v, d
        return ("FAIL" if v == "FAIL" else "PRECOMPILE-ERR"), (d if bl_hit(exc) else head)
    return _verdict_for(op)


def bl_hit(_exc) -> bool:
    return any(k.startswith("ascend|") for k in ascend_env.blockers())


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ops = argv or list(DRIVERS)
    _install_stubs()
    n_fail = 0
    for op in ops:
        try:
            verdict, detail = run_one(op)
        except Exception as exc:  # noqa: BLE001
            verdict, detail = "PRECOMPILE-ERR", (traceback.format_exception_only(type(exc), exc) or [""])[0][:160]
        if verdict != "PASS":
            n_fail += 1
        print(f"ASC-{op} {verdict} {detail}")
    print(f"—— 汇总 PASS={len(ops)-n_fail} FAIL={n_fail} ——")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
