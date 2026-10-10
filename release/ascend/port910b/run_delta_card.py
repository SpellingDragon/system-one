#!/usr/bin/env python3
"""run_delta_card.py — P1-3 GDN delta rule 前向的**真机执行**收数件（910B）。

F 波已在 CPU 证真（fp32 递推 golden rel=5.76e-07 / adjoint-vs-FD 7.28e-10），但容器无
davinci 设备从未**执行**过 kernel——本件在卡上编译+跑+对拍 fwd_exact(float64 精确递推)。

纪律：不用 out_idx 魔法（本地无法 run-verify，猜错即烧卡窗）。显式分配全部 9 张量，
kernel 就地写 HIST/SOUT/O，逐一比对。镜像 run_numerics 里已验证的 num_gdn 喂参形态。

判据行：NUM-delta_fwd PASS rel_o=<x> rel_s=<y>   /   NUM-delta_fwd FAIL <reason>
"""
import os, sys, argparse
os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "attempts", "F"))
import numpy as np
import tilelang
import tilelang.language as T  # noqa: F401
# torch / torch_npu 延迟到 run 半程（device-less 容器可验编译，与 run_numerics --no-run 同纪律）

AP = argparse.ArgumentParser()
AP.add_argument("--compile-only", action="store_true", help="只编译不跑（device-less 判决）")
ARGS = AP.parse_args()

H, TT, DK, DV, CORES = 2, 8, 8, 8, 1
DT = "float32"


def relerr(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = np.linalg.norm(b) + 1e-30
    return np.linalg.norm(a - b) / d


def main():
    try:
        import f_gdn_delta_fwd_910b as mod
        import f_gdn_golden as gd
        fn = mod.gdn_delta_rule_fwd(H=H, T_LEN=TT, DK=DK, DV=DV, cores=CORES,
                                    dtype=DT, variant="ub", emit_hist=True)
        k = tilelang.compile(fn, target="ascend")  # 显式传全部张量，不用 out_idx
        if ARGS.compile_only:
            print("NUM-delta_fwd PASS rel=n/a (compile-only)")
            return 0
    except Exception as e:  # compile 失败
        print(f"NUM-delta_fwd FAIL compile: {type(e).__name__}: {str(e)[:200]}")
        return 1

    try:
        import torch
        import torch_npu  # noqa: F401  # device backend（卡上才有）
        rng = np.random.default_rng(11)
        Q = rng.standard_normal((H, TT, DK)).astype(DT)
        Kt = (rng.standard_normal((H, TT, DK)) * 0.3).astype(DT)
        V = rng.standard_normal((H, TT, DV)).astype(DT)
        BETA = (rng.random((H, TT)) * 0.5 + 0.25).astype(DT)
        G = (-rng.random((H, TT))).astype(DT)          # decay log 门 g<=0
        H0 = (rng.standard_normal((H, DV, DK)) * 0.3).astype(DT)
        HIST = np.zeros((H, TT, DV, DK), DT)
        SOUT = np.zeros((H, DV, DK), DT)
        O = np.zeros((H, TT, DV), DT)

        dev = lambda x: torch.from_numpy(np.ascontiguousarray(x)).npu()
        inps = [dev(x) for x in (Q, Kt, V, BETA, G, H0)]
        outs = [dev(x) for x in (HIST, SOUT, O)]
        # kernel 签名序 = Q,K,V,BETA,G,H0,HIST,SOUT,O
        k(*inps, *outs)
        torch.npu.synchronize()

        got_o = outs[2].cpu().numpy().astype(np.float64)
        got_s = outs[1].cpu().numpy().astype(np.float64)
        ref_o, _ref_h, ref_sf = gd.fwd_exact(H, TT, DK, DV,
                                             Q, Kt, V, BETA, G, H0)
        r_o = relerr(got_o, ref_o)
        r_s = relerr(got_s, ref_sf)
        # F CPU 口径 (a) 仅 expf，期望 rel_o 1e-6 级；阈值给 bf16/漂移余量
        if r_o < 2e-4 and r_s < 2e-4:
            print(f"NUM-delta_fwd PASS rel_o={r_o:.2e} rel_s={r_s:.2e}")
            return 0
        print(f"NUM-delta_fwd FAIL rel_o={r_o:.3e} rel_s={r_s:.3e} (阈值 2e-4)")
        return 1
    except Exception as e:
        print(f"NUM-delta_fwd FAIL run: {type(e).__name__}: {str(e)[:200]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
