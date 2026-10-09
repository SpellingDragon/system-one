#!/usr/bin/env python3
"""run_numerics.py — 910B 上卡收数工装（P0-2 收官件，六 target）。

每 target 打印一行判决（R19：无输出=未完成）：
  NUM-<name>-PASS rel=<v> [ms=.. tflops=..]     数值正确（阈值见各 target）
  NUM-<name>-FAIL <reason>                      编译/执行/超差
用法（卡上，bundle 已铺好 compat+注入）：
  python3 run_numerics.py                 # 全部
  python3 run_numerics.py --only dw       # 单件
  python3 run_numerics.py --no-run        # 仅编译判（本地 dry-run 用）
"""
import os
import sys
import argparse
import importlib

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
ATT = os.path.join(HERE, "attempts")
for d in ("C", "D", "E"):
    sys.path.insert(0, os.path.join(ATT, d))

import numpy as np  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
from tilelang.ascend.language import Cube, alloc_l1, alloc_l0c  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--only", default="all")
AP.add_argument("--no-run", action="store_true")
ARGS = AP.parse_args()

DT = {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16"}


def _t(shape, dtype, scale=1.0):
    import torch
    t = getattr(torch, DT[dtype])(torch.randn(*shape, device="npu", dtype=torch.float32) * scale)
    return t


def _np(x):
    return x.detach().float().cpu().numpy()


def compile_and_time(kern, args, iters=20):
    import torch, time
    torch.npu.synchronize()
    out = kern(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        kern(*args)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return out, ms


def relerr(a, b):
    a = a.astype(np.float32).ravel(); b = b.astype(np.float32).ravel()
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))


# ── 1. gemm direct（bf16 2048³，e2e_cube 形态 + 数值与 tflops）────────────────
def num_gemm_direct():
    def gemm(M, N, K, blk):
        @T.prim_func
        def main(A: T.Tensor((M, K), "bfloat16"), B: T.Tensor((K, N), "bfloat16"),
                 C: T.Tensor((M, N), "bfloat16")):
            with T.Kernel(T.ceildiv(N, blk), T.ceildiv(M, blk), threads=1) as (bx, by):
                A_s = T.alloc_shared((blk, K), "bfloat16")
                B_s = T.alloc_shared((K, blk), "bfloat16")
                C_l = T.alloc_shared((blk, blk), "float")
                with Cube():
                    T.copy(A[by * blk, 0], A_s)
                    T.copy(B[0, bx * blk], B_s)
                    T.gemm(A_s, B_s, C_l, clear_accum=True)
                    for i in T.serial(blk):
                        for j in T.serial(blk):
                            C[by * blk + i, bx * blk + j] = T.cast(C_l[i, j], "bfloat16")
        return main
    M = N = K = 2048
    k = tilelang.compile(gemm(M, N, K, 128), target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-gemm_direct-PASS rel=n/a (compile-only)"
    import torch
    A = torch.randn(M, K, dtype=torch.bfloat16, device="npu")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="npu")
    out, ms = compile_and_time(k, (A, B))
    ref = (A.float() @ B.float()).to(torch.bfloat16)
    r = relerr(_np(out), _np(ref))
    tf = 2 * M * N * K / (ms * 1e-3) / 1e12
    assert r < 0.02, f"rel={r}"
    return f"NUM-gemm_direct-PASS rel={r:.5f} ms={ms:.3f} tflops={tf:.2f}"


# ── 2. gemm_l1（shared.l1 模板路 64³；V1/V2/V3 单位证真载体）──────────────────
def num_gemm_l1():
    R = C64 = 64

    def kern():
        @T.prim_func
        def main(A: T.Tensor((R, C64), "bfloat16"), B: T.Tensor((C64, C64), "bfloat16"),
                 O: T.Tensor((R, C64), "float32")):
            with T.Kernel(1):
                a_l1 = alloc_l1((R, C64), "bfloat16")
                b_l1 = alloc_l1((C64, C64), "bfloat16")
                c_l0 = alloc_l0c((R, C64), "float32")
                with Cube():
                    T.copy(A, a_l1)
                    T.copy(B, b_l1)
                    T.gemm(a_l1, b_l1, c_l0, transpose_B=True, clear_accum=True)
                    T.copy(c_l0, O)
        return main
    k = tilelang.compile(kern(), target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-gemm_l1-PASS rel=n/a (compile-only)"
    A = _t((R, C64), "bfloat16")
    Bm = _t((C64, C64), "bfloat16")
    out, _ = compile_and_time(k, (A, Bm), iters=5)
    ref = A.float() @ Bm.float()
    r = relerr(_np(out), ref.cpu().numpy())
    assert r < 0.02, f"rel={r} —— V1/V2/V3 任一单位错都体现在此"
    return f"NUM-gemm_l1-PASS rel={r:.5f}"


# ── 3. gdn short conv（import C 波 kernel+golden）─────────────────────────────
def num_gdn():
    mod = importlib.import_module("c_gdn_conv_910b")
    gd = importlib.import_module("c_gdn_golden")
    C_, L_, Kk = 512, 512, 4          # 缩容快跑（全 2048×512 可 --only gdn 复跑）
    fn = mod.gdn_short_conv_fwd(C=C_, L=L_, K=Kk, silu="exp")
    k = tilelang.compile(fn, target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-gdn-PASS rel=n/a (compile-only)"
    rng = np.random.default_rng(7)
    X = rng.standard_normal((C_, L_)).astype(np.float32)
    W = rng.standard_normal((C_, Kk)).astype(np.float32) * 0.5
    BI = rng.standard_normal((C_,)).astype(np.float32) * 0.1
    import torch
    args = tuple(torch.from_numpy(x).cuda() if False else torch.from_numpy(x).npu() for x in (X, W, BI))
    out, _ = compile_and_time(k, args, iters=5)
    ref = gd.short_conv_compat(X, W, BI, silu="exp")
    r = relerr(_np(out), ref)
    assert r < 2e-5, f"rel={r} (期望 ≤ C 波实测 3.3e-7 级)"
    return f"NUM-gdn-PASS rel={r:.2e}"


# ── 4. dW l0tr（npz 判据：P0 方向判别 + P1 阈值）─────────────────────────────
def num_dw():
    z = np.load(os.path.join(ATT, "D", "dw_pairs.npz"))
    dY, X, exact, exact_T = z["dY"], z["X"], z["exact"], z["exact_T"]
    Tv, Nv, Kv, TT, BN, BK = int(z["T"]), int(z["N"]), int(z["K"]), int(z["TT"]), int(z["BN"]), int(z["BK"])
    mod = importlib.import_module("d_dw_910b")
    fn = mod.dw_l0tr(Tv, Nv, Kv, BN, BK, TT, min(Nv // BN * (Kv // BK), 64), "float16", "float32")
    k = tilelang.compile(fn, target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-dw-PASS rel=n/a (compile-only)"
    import torch
    a1 = torch.from_numpy(dY).npu(); a2 = torch.from_numpy(X).npu()
    out, _ = compile_and_time(k, (a1, a2), iters=5)
    got = _np(out)
    r_ok, r_swap = relerr(got, exact), relerr(got, exact_T)
    assert r_ok < 0.02 and r_ok < r_swap - 0.5, f"rel={r_ok:.4f} rel_swap={r_swap:.4f}（P0 方向判别失败）"
    return f"NUM-dw-PASS rel={r_ok:.5f} dir_disc={r_swap:.3f}"


# ── 5. rope（表加载 fwd，f64 精确式对拍）─────────────────────────────────────
def num_rope():
    mod = importlib.import_module("e_rope_910b")
    eg = importlib.import_module("e_golden")
    tk, hd, dm = 64, 8, 64
    fn = mod.rope_fwd(tokens=tk, heads=hd, dim=dm, mode="table", sign=1)
    k = tilelang.compile(fn, target="ascend", out_idx=1)
    if ARGS.no_run:
        return "NUM-rope-PASS rel=n/a (compile-only)"
    rng = np.random.default_rng(7)
    qkv = rng.standard_normal((tk, 3, hd, dm)).astype(np.float32)
    ang = rng.uniform(0, 3.14, (tk, dm // 2)).astype(np.float32)
    cos, sin, freq = np.cos(ang), np.sin(ang), (ang * 0 + 1).astype(np.float32)
    import torch
    ts = [torch.from_numpy(x).npu() for x in (qkv, cos, sin, freq)]
    out = k(*ts)
    ref = eg.rope_ref_f64(qkv, cos, sin, sign=1)
    r = relerr(np.asarray(out).astype(np.float32), ref.astype(np.float32))
    assert r < 5e-6, f"rel={r}"
    return f"NUM-rope-PASS rel={r:.2e}"


# ── 6. attn_sw（滑窗 softmax，numpy 参考）─────────────────────────────────────
def num_attnsw():
    mod = importlib.import_module("e_attn_sw_910b")
    h, s, d, w = 4, 64, 32, 8
    fn = mod.attn_sw_fwd(heads=h, seq=s, dim=d, window=w, variant="stage")
    k = tilelang.compile(fn, target="ascend", out_idx=3)
    if ARGS.no_run:
        return "NUM-attnsw-PASS rel=n/a (compile-only)"
    rng = np.random.default_rng(7)
    Q = rng.standard_normal((h, s, d)).astype(np.float32)
    K = rng.standard_normal((h, s, d)).astype(np.float32)
    V = rng.standard_normal((h, s, d)).astype(np.float32)
    import torch
    out = k(*[torch.from_numpy(x).npu() for x in (Q, K, V)])
    scale = d ** -0.5
    ref = np.zeros_like(Q)
    for hh in range(h):
        for t in range(s):
            lo = max(0, t - w + 1)
            sc = (Q[hh, t] @ K[hh, lo:t + 1].T) * scale
            sc -= sc.max()
            e = np.exp(sc); e /= e.sum()
            ref[hh, t] = e @ V[hh, lo:t + 1]
    r = relerr(np.asarray(out).astype(np.float32), ref)
    assert r < 1e-4, f"rel={r}"
    return f"NUM-attnsw-PASS rel={r:.2e}"


TARGETS = {"gemm_direct": num_gemm_direct, "gemm_l1": num_gemm_l1, "gdn": num_gdn,
           "dw": num_dw, "rope": num_rope, "attnsw": num_attnsw}

if __name__ == "__main__":
    names = [ARGS.only] if ARGS.only != "all" else list(TARGETS)
    for nm in names:
        try:
            print(TARGETS[nm](), flush=True)
        except Exception as e:
            print(f"NUM-{nm}-FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
