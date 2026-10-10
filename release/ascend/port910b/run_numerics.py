#!/usr/bin/env python3
"""run_numerics.py — 910B 上卡收数工装（P0-2 收官件，八 target）。

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
# B 域不列入 sys.path：b_addln_910b.py / b_readout_910b.py 是 module 级脚本
# （末尾 sys.exit(main_())），importlib 整脚本执行会直接终止本进程；两件已按
# 其 build() 逐字内联为 num_addln / num_readout（见 §7 §8）。
for d in ("C", "D", "E"):
    sys.path.insert(0, os.path.join(ATT, d))

import numpy as np  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402
import tilelang.ascend.language as TA  # noqa: E402  # Ascend 方言（T.Kernel 无 threads=，刻意 shadow CUDA 门面）
from tilelang.ascend.language import Cube, alloc_l1, alloc_l0c  # noqa: E402

AP = argparse.ArgumentParser()
AP.add_argument("--only", default="all")
AP.add_argument("--no-run", action="store_true")
ARGS = AP.parse_args()

DT = {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16"}


def _t(shape, dtype, scale=1.0):
    import torch
    TORCH_DT = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    return (torch.randn(*shape, device="npu", dtype=torch.float32) * scale).to(TORCH_DT[DT[dtype]])


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
    cos, sin = np.cos(ang), np.sin(ang)
    freq = (1.0 / (10000 ** (2 * np.arange(dm // 2) / dm))).astype(np.float32)
    import torch
    ts = [torch.from_numpy(x).npu() for x in (qkv, cos, sin, freq)]
    out = k(*ts)
    ref = eg.rope_ref_f64(qkv, cos, sin, sign=1)
    r = relerr(_np(out).astype(np.float32), ref.astype(np.float32))
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
    r = relerr(_np(out).astype(np.float32), ref)
    assert r < 1e-4, f"rel={r}"
    return f"NUM-attnsw-PASS rel={r:.2e}"


# ── 7. add_ln（残差相加 + 层内整形）───────────────────────────────────────────
# DSL 逐字内联自 attempts/B/b_addln_910b.py::build()（静态 rows 分支，即 --dyn 关闭），
# 位宽/形状取该文件的环境变量缺省：B_ROWS=64 B_DIM=512 B_BM=8 B_RES_FP32=1 B_OUT_FP16=1
# B_EPS=1e-5 ⇒ RDT=float32、ODT=float16、UB~28KB。out_idx=-1 与该文件 main_() 的调用一致。
# 数值口径照其 docstring：三趟扫描、**有偏方差**（/DIM 非 /(DIM-1)）、Hout 全程 fp32 不降位宽、Y 落 fp16。
# 阈值来自宿主机仿真（照 DSL 逐趟 fp32 标量算术重放，/tmp/g_probe/emu.py）：
#   r_h：正确=0.0（单次 fp32 加，与参考逐位同）/ 残差路降 fp16=2.1e-4 ⇒ 1e-5 精确判别"不许降位宽"
#   r_y：正确=1.5e-5 / 方差错用无偏=7.4e-4 ⇒ 3e-4 判别"有偏"口径（原拟 5e-3 判别不了，已收紧）
def num_addln():
    ROWS, DIM, BM = 64, 512, 8
    F16, F32 = "float16", "float32"
    RDT, ODT = F32, F16                 # RES_FP32=1 / OUT_FP16=1
    EPS = 1e-5

    def build():
        @TA.prim_func
        def main(X: TA.Tensor((ROWS, DIM), F16), Res: TA.Tensor((ROWS, DIM), RDT),
                 G: TA.Tensor((DIM,), F32), Bt: TA.Tensor((DIM,), F32),
                 Y: TA.Tensor((ROWS, DIM), ODT), Hout: TA.Tensor((ROWS, DIM), F32)):
            with TA.Kernel(TA.ceildiv(ROWS, BM)) as bx:
                x_ub = TA.alloc_shared((BM, DIM), F16)
                r_ub = TA.alloc_shared((BM, DIM), RDT)
                g_ub = TA.alloc_shared((DIM,), F32)
                b_ub = TA.alloc_shared((DIM,), F32)
                TA.copy(X[bx * BM, 0], x_ub)
                TA.copy(Res[bx * BM, 0], r_ub)
                TA.copy(G[0:DIM], g_ub)
                TA.copy(Bt[0:DIM], b_ub)
                for i in TA.serial(BM):
                    row = bx * BM + i
                    val = TA.alloc_var(F32)
                    acc = TA.alloc_var(F32)
                    mu = TA.alloc_var(F32)
                    dv = TA.alloc_var(F32)
                    rs = TA.alloc_var(F32)
                    # 第一趟：残差相加，fp32 原样交出口，同时攒行和
                    acc = 0.0
                    for j in TA.serial(DIM):
                        val = TA.cast(x_ub[i, j], F32) + TA.cast(r_ub[i, j], F32)
                        Hout[row, j] = val
                        acc = acc + val
                    mu = acc / DIM
                    # 第二趟：偏差平方和（有偏方差）
                    acc = 0.0
                    for j in TA.serial(DIM):
                        dv = TA.cast(x_ub[i, j], F32) + TA.cast(r_ub[i, j], F32) - mu
                        acc = acc + dv * dv
                    rs = TA.rsqrt(acc / DIM + EPS)
                    # 第三趟：缩放 + 仿射，按 ODT 交货
                    for j in TA.serial(DIM):
                        dv = TA.cast(x_ub[i, j], F32) + TA.cast(r_ub[i, j], F32) - mu
                        Y[row, j] = TA.cast(dv * rs * g_ub[j] + b_ub[j], ODT)

        return main

    k = tilelang.compile(build(), target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-addln-PASS rel=n/a (compile-only)"
    import torch
    X = _t((ROWS, DIM), F16)
    Res = _t((ROWS, DIM), RDT) + 4.0    # 大共模/小波动：贴生产残差流量级，也是三趟式的动机
    G = _t((DIM,), F32)
    Bt = _t((DIM,), F32)
    Y = torch.zeros((ROWS, DIM), dtype=torch.float16, device="npu")   # 非 out 槽 ⇒ 就地写回
    out, _ = compile_and_time(k, (X, Res, G, Bt, Y), iters=5)         # out == Hout
    # 参考实现：入参先按 kernel 实际位宽量化（X 是 fp16），再全程 fp32；算式顺序照 DSL
    xv, rv = _np(X).astype(np.float32), _np(Res).astype(np.float32)
    gv, bv = _np(G).astype(np.float32), _np(Bt).astype(np.float32)
    v = (xv + rv).astype(np.float32)                        # Hout：fp32 残差流
    mu = v.mean(axis=1, keepdims=True)
    dev = (v - mu).astype(np.float32)
    var_b = (dev * dev).mean(axis=1, keepdims=True).astype(np.float32)      # 有偏（kernel 口径）
    var_u = (dev * dev).sum(axis=1, keepdims=True) / (DIM - 1)              # 无偏（错误口径，仅作判别）
    ref_h = v
    ref_y = ((dev * (1.0 / np.sqrt(var_b + EPS)).astype(np.float32)) * gv + bv).astype(np.float16)
    alt_y = ((dev * (1.0 / np.sqrt(var_u + EPS)).astype(np.float32)) * gv + bv).astype(np.float16)
    got_y, got_h = _np(Y).astype(np.float32), _np(out).astype(np.float32)
    r_h = relerr(got_h, ref_h.astype(np.float32))
    r_y = relerr(got_y, ref_y.astype(np.float32))
    r_unb = relerr(got_y, alt_y.astype(np.float32))          # 与"无偏"错口径的距离（须明显更远）
    assert r_h < 1e-5, f"Hout rel={r_h:.3e}（fp32 残差路应逐位同；~2e-4 即降了位宽）"
    assert r_y < 3e-4, (f"Y rel={r_y:.3e}（仿真底噪 1.5e-5；若 Y 全零则 rel≈1 ⇒ 就地写回未生效）"
                        f" r_unb={r_unb:.3e}")
    assert r_y < r_unb - 2e-4, f"有偏/无偏口径判别失败 r_y={r_y:.3e} r_unb={r_unb:.3e}"
    return f"NUM-addln-PASS rel_y={r_y:.2e} rel_h={r_h:.2e} unb_disc={r_unb:.2e}"


# ── 8. readout（末位读出 × 字母行投影 = scores）───────────────────────────────
# DSL 逐字内联自 attempts/B/b_readout_910b.py::build()，取其环境变量缺省：
# R_BATCH=8 R_SEQ=32 R_DIM=512 R_K=26 R_BM=4 R_HDT=float16（UB~60KB），out_idx=-1 同该文件。
# 范围声明照其 docstring：只出 scores，不含 softmax/to_probs（dav-2201 vector 面无 expf/log）。
# 阈值来自宿主机仿真：512 项 fp32 顺序累加 vs f64 参考 rel=4.0e-7 ⇒ 1e-5 留 25x 余量。
def num_readout():
    BATCH, SEQ, DIM, KK, BM = 8, 32, 512, 26, 4
    F32, I32 = "float32", "int32"
    HDT = "float16"
    VOCAB = 4096

    def build():
        @TA.prim_func
        def main(HID: TA.Tensor((BATCH, SEQ, DIM), HDT), POS: TA.Tensor((BATCH,), I32),
                 W: TA.Tensor((VOCAB, DIM), HDT), IDS: TA.Tensor((KK,), I32),
                 S: TA.Tensor((BATCH, KK), F32)):
            with TA.Kernel(TA.ceildiv(BATCH, BM)) as bx:
                h_ub = TA.alloc_shared((BM, DIM), F32)   # 本块每行的"末位数字"，已升 fp32
                w_ub = TA.alloc_shared((KK, DIM), F32)    # k 个字母行，已升 fp32
                # 1) 逐行按 pos 取末位（动态下标散取；界守卫由 codegen 生成）
                for i in TA.serial(BM):
                    for j in TA.serial(DIM):
                        h_ub[i, j] = TA.cast(HID[bx * BM + i, POS[bx * BM + i], j], F32)
                # 2) 按 letter_ids 取字母行
                for t in TA.serial(KK):
                    for j in TA.serial(DIM):
                        w_ub[t, j] = TA.cast(W[IDS[t], j], F32)
                # 3) 点积出 (BM, K) 分数，标量落 GM
                for i in TA.serial(BM):
                    acc = TA.alloc_var(F32)
                    for t in TA.serial(KK):
                        acc = 0.0
                        for j in TA.serial(DIM):
                            acc = acc + h_ub[i, j] * w_ub[t, j]
                        S[bx * BM + i, t] = acc

        return main

    k = tilelang.compile(build(), target="ascend", out_idx=-1)
    if ARGS.no_run:
        return "NUM-readout-PASS rel=n/a (compile-only)"
    import torch
    rng = np.random.default_rng(7)
    HID = _t((BATCH, SEQ, DIM), HDT)
    W = _t((VOCAB, DIM), HDT)
    pos = rng.integers(0, SEQ, size=BATCH).astype(np.int32)   # 动态下标必须落在 [0,SEQ)
    ids = np.arange(KK, dtype=np.int32)                       # A..Z 取词表前 26 行
    # 前置守卫：散取下标必须"非平凡"，否则本 case 会退化成没测动态寻址的空测
    assert pos.min() >= 0 and pos.max() < SEQ, f"POS 越界 [{pos.min()},{pos.max()}] vs SEQ={SEQ}"
    assert np.unique(pos).size > 1, "POS 全同 ⇒ 动态散取未被激励"
    assert np.unique(ids).size == KK and ids.max() < VOCAB, "IDS 退化/越界"
    POS = torch.from_numpy(pos).npu()
    IDS = torch.from_numpy(ids).npu()
    out, _ = compile_and_time(k, (HID, POS, W, IDS), iters=5)   # out == S (BATCH,KK) fp32
    hid_np, w_np = _np(HID).astype(np.float32), _np(W).astype(np.float32)
    rows = hid_np[np.arange(BATCH), pos, :].astype(np.float64)   # 逐行按 pos 散取
    wr = w_np[ids, :].astype(np.float64)
    ref = (rows @ wr.T).astype(np.float32)                       # .float() 语义 = fp32 累加
    bad = (hid_np[:, 0, :].astype(np.float64) @ wr.T).astype(np.float32)  # 错取首位（不随 pos）
    r = relerr(_np(out).astype(np.float32), ref)
    r_bad = relerr(_np(out).astype(np.float32), bad)
    assert r < 1e-5, f"rel={r}（仿真底噪 4.0e-7；若 S 全零则 rel≈1）"
    assert r < r_bad - 0.5, f"末位散取方向判别失败 rel={r:.3e} rel_firstpos={r_bad:.3e}"
    return f"NUM-readout-PASS rel={r:.2e} gather_disc={r_bad:.2e}"


TARGETS = {"gemm_direct": num_gemm_direct, "gemm_l1": num_gemm_l1, "gdn": num_gdn,
           "dw": num_dw, "rope": num_rope, "attnsw": num_attnsw,
           "addln": num_addln, "readout": num_readout}

if __name__ == "__main__":
    names = [ARGS.only] if ARGS.only != "all" else list(TARGETS)
    for nm in names:
        try:
            print(TARGETS[nm](), flush=True)
        except Exception as e:
            print(f"NUM-{nm}-FAIL {type(e).__name__}: {str(e)[:300]}", flush=True)
