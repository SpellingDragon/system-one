#!/usr/bin/env python3
"""reconcile/verify_dw.py — K 代理：gemm_bwd_dw_asc.py（l0tr 主案）的 910B target=ascend 编译判决。

判决主体（接口真实路径，**如实报红**）：绕设备门 + monkeypatch compile，用真实形状
（dY(m,n) A(m,k) GW(n,k)，m 动态、GW fp32）驱动 gemm_bwd_dw_kernel.backward(target="ascend")。
预期：因动态 m 末块触发 §12 缺件 asc_fill_l1 → compile 失败 → blockers 含 ascend| → 打
K-GEMM-BWD-DW ASC-COMPILE-FAIL（**不假绿**），并把首行错误（应含 'asc_fill_l1'）原样打印。

结构自证（旁证，非判决主体）：另编一份"静态 m 且整除 tc"的 l0tr 同型件——若无 asc_fill_l1
即编过，则证明 DSL 形态正确、唯一挡编译点是动态尾填充件（对应 reconcile/compat_patch_K.h）。

用法（容器内）：
  source /usr/local/Ascend/cann-8.5.0/set_env.sh
  export BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler ASCEND_NPU_ARCH=dav-2201
  cd /work && TILELANG_CACHE_DIR=$(mktemp -d) timeout 300 python3 ascend/kernels/reconcile/verify_dw.py
"""
import os
import sys

os.environ.setdefault("ASCEND_NPU_ARCH", "dav-2201")
HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

import torch           # noqa: E402
import tilelang        # noqa: E402
from ascend.kernels import ascend_env, gemm_bwd_dw_kernel  # noqa: E402


class LaunchSkip(Exception):
    pass


def _stub_compile():
    real = tilelang.compile

    def _fake(*a, **k):
        kern = real(*a, **k)
        return kern  # 判决主体不抛（我们要拿到真错误），但真编译本身可能抛在 real() 里

    tilelang.compile = _fake


def _gate():
    ascend_env.reset()
    ascend_env.backend_available = lambda *a, **k: True
    ascend_env._npu_present = lambda: True
    ascend_env.active_backend = lambda target=None, device=None: ascend_env.TILELANG


def _interface_dynamic_m() -> int:
    """判决主体：接口路径（动态 m 的 l0tr）——预期受阻于 asc_fill_l1，如实报。"""
    # 不 monkeypatch：让真编译错误经 ascend_env.get_compiled 记进 blockers（返回 False 落回退）
    _gate()
    m, n, k = 32, 64, 64
    dY = torch.randn(m, n, dtype=torch.float32)
    A = torch.randn(m, k, dtype=torch.float32)
    gw = gemm_bwd_dw_kernel.backward(dY, A, out_dtype=torch.float32, target="ascend")
    keys = [x for x in ascend_env.compiled_keys() if x.startswith("ascend|")]
    bl = {x: v for x, v in ascend_env.blockers().items() if x.startswith("ascend|")}
    print(f"[interface·dynamic-m] keys={keys}")
    print(f"[interface·dynamic-m] blockers={bl}")
    if not keys and bl:
        first = next(iter(bl.values()))
        print(f"[interface·dynamic-m] 首错={first[:240]}")
        if "asc_fill_l1" in first:
            print("K-GEMM-BWD-DW ASC-COMPILE-BLOCKED(§12 asc_fill_l1)")
        print("K-GEMM-BWD-DW ASC-COMPILE-FAIL")
        return 1
    if keys and not bl:
        print("K-GEMM-BWD-DW ASC-COMPILE-PASS")
        return 0
    print("K-GEMM-BWD-DW ASC-COMPILE-FAIL")
    return 1


def _static_structure_sanity() -> int:
    """旁证：静态 m(整除 tc) 的同型 l0tr —— 若编过则结构正确、唯动态尾填充挡路。"""
    import tilelang.ascend.language as T
    n, k, tc, bn, bk = 64, 64, 16, 64, 64
    m_static = 32           # 编译期常量，32 % 16 == 0
    nb, nk = n // bn, k // bk
    tiles = nb * nk
    blocks = min(tiles, 8)
    iters = (tiles + blocks - 1) // blocks
    ks = m_static // tc     # 静态整除 → 不发射 asc_fill_l1

    @T.prim_func
    def dw_static(dY: T.Tensor((m_static, n), "float32"),
                  A: T.Tensor((m_static, k), "float32"),
                  GW: T.Tensor((n, k), "float32")):
        with T.Kernel(blocks) as bid:
            dy_l1 = T.alloc_l1((tc, bn), "float32")
            x_l1 = T.alloc_l1((tc, bk), "float32")
            a_l0 = T.alloc_l0a((bn, tc), "float32")
            b_l0 = T.alloc_l0b((bk, tc), "float32")
            acc = T.alloc_l0c((bn, bk), "float32")
            with T.Cube():
                for it in T.serial(iters):
                    tid = bid + it * blocks
                    n_tile = tid // nk
                    k_tile = tid % nk
                    for kt in T.serial(ks):
                        T.copy(dY[kt * tc, n_tile * bn], dy_l1)
                        T.copy(A[kt * tc, k_tile * bk], x_l1)
                        T.copy(dy_l1, a_l0, transpose=True)
                        T.copy(x_l1, b_l0, transpose=True)
                        T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=kt == 0)
                    T.copy(acc, GW[n_tile * bn, k_tile * bk])

    try:
        kern = tilelang.compile(dw_static, target="ascend", out_idx=[])
        src = str(kern.get_kernel_source())
        print(f"[static-sanity] PASS .o~{len(src)}B asc_fill_l1_in_src={'asc_fill_l1' in src} "
              f"mix={'__mix__' in src}")
        print("K-GEMM-BWD-DW STRUCTURE-SANITY-PASS(l0tr 形态可编，唯动态尾填充挡路)")
        return 0
    except Exception as e:
        msg = str(e)
        i = msg.find("error:")
        print(f"[static-sanity] FAIL {msg[i:i+200].replace(chr(10),' ') if i>=0 else msg.splitlines()[0][:200]}")
        return 1


if __name__ == "__main__":
    rc1 = _interface_dynamic_m()
    rc2 = _static_structure_sanity()
    # 判决主体（接口动态 m）为准：受阻即非零退出，如实报；旁证单独打印不计入主判。
    print(f"\n[summary] interface-dynamic-m rc={rc1} (0=PASS/1=BLOCKED-FAIL); "
          f"static-sanity rc={rc2}")
    sys.exit(rc1)
