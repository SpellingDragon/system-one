"""gemm 权重梯度方言件：dW = dYᵀ @ A，沿 token 轴归约（token 数 m 是动态符号）。

【做什么】算"每个连接权该往哪儿挪"的那张表：把一批头信号转置后乘回输入，沿"这批有几条样本"的轴
累加成与权重同形的 (N, K) 结果。前向件的模具把动态轴放在行数上、归约轴是编译期常量，表达不了
本件事，所以另开一模。
【怎么做】① 出口按 (N,K) 分块，网格上界是纯常量（块数），token 轴用 T.serial(ceildiv(m, tc))
   动态上界串行推进——换批大小不重编（实测同一产物连跑 m=8/33 误差都是 0.0，compile_count=1）；
   ② 整数常量必须是被追踪函数自己的参数，且本文件不写 from __future__ import annotations：
   tilelang 取张量注解只认闭包非局部名，PEP 563 会让只在注解里出现的 n/k 失去闭包单元
   （实测 NameError: name 'n' is not defined）；③ 取产物走 tilelang.compile(现场追踪的 PrimFunc)
   而非 tilelang.jit——后者在本件这种写法下只回 Kernel 不执行，且它的 call-form 缓存拿张量做
   等值比较，换一批新张量就抛错；④ 累加器 (bn,bk) 与出口都 fp32，降位由入口层负责。
【为什么】昇腾那份不能照抄 CPU 的 transpose_A=True：L1 输入的乘加路只认"左不翻、右翻"（手册与
   TileKernels 的 GEMM 件同口径），`gemm_mad.py` 对 shared.l1 路直接 assert `not trans_A and
   trans_B`，dW 天然 trans_A（TN）正面撞墙。**910B 合流（P1-4）走 l0tr 主案**：把转置下移到
   DMA 装填——L1 里存 dY/X 的自然 (token块, 列) 布局，装进 L0A/L0B 时 `transpose=True` 转置成
   NT 要求的 (N块,token块)/(K块,token块)，mad 永远吃 NT；累加器 (BN,BK) 与出口同朝向，
   L0C→GM 直出（A-4 已落 asc_copy_l0c2gm），不在 UB 上二次重排（910B 无 SimtVF/T.Parallel，
   且 L0C→UB 硬件不支持）。等价于转置恒等式 dW=(Aᵀ@dY)ᵀ，但把"再转置"消灭在入口 DMA。
   被否方案一：照 950 载体用 SimtVF 在 UB 做转置写回——910B 方言无此构造，COMPILE-FAIL；
   被否方案二：madta（装填不转置、mad 直接吃 MN-major A）——GM→L0 无 DMA 面，codegen 退化
   成标量解引用 `__ca__`，命中 `only __ubuf__/__gm__/local can be dereferenced`（卷宗 G-D2）；
   被否方案三：ntt（GM→L1 dn2nz 转置载入 + shared.l1 融合模板）——编译绿但 compat 的
   asc_copy_gm2l1_dn2nz 只是 nd2nz 同名别名 stub（VERIFY V1），编译绿≠数值绿，前波已裁不投；
   被否方案四：token 数做编译期常量换批重编——训练批形多变，秒级重编吞掉收益。
   ⚠ 已知 §12 缺口（本波实测）：动态 m 下末个 token 块不满 TC(=16) 行时，codegen 会在
   asc_copy_gm2l1_nd2nz 之后发射 `asc_fill_l1` 补 L1 尾区，而 trunk port910b_compat.h 无此件
   → `use of undeclared identifier 'asc_fill_l1'`。静态且 m 整除 TC 不触发（本容器实测原版
   d_dw PASS）。DSL 形态正确，唯此件挡编译；补件需求见 reconcile/compat_patch_K.h，由 §12 owner 合。
云端待验清单（本地不实编）：asc_fill_l1 补齐后的 cube 数值 rel、tc 取值、fp32 操作数是否需 set_hf32_mode。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: token 轴块长：与前向 BLOCK_M 同量级，太小喂不饱并行、太大尾块浪费
BLOCK_TC = 16
#: N/K 轴块宽候选（沿用前向阶梯）；都不能整除则表达不了，交回入口层回退
BLOCK_LADDER = (64, 32, 16)
#: 昇腾侧一次发射占用的立方体核块数与流水深度（与 gemm_asc 同一口径）
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2


def dw_cpu_impl(dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int):
    """CPU(c) 方言正文：dW(N,K) 沿动态 token 轴累加；网格是常量，token 循环是动态上界。

    白话：把梯度表切成固定大小的小方格，每格自己把"这一批所有样本对该格的贡献"一笔笔记到同一个
    账本上；样本有多少条不影响账本格子的大小，记完一次交账。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    m = T.dynamic("m")
    nb, nk = n // bn, k // bk

    @T.prim_func
    def dw_impl(dY: T.Tensor((m, n), "float32"), A: T.Tensor((m, k), "float32"),
                GW: T.Tensor((n, k), "float32")):
        """被追踪的那一层：权重表 (N,K) 是常量，样本行数 M 是活的累加轴。

        白话：CPU 件这一份，把每条样本的头信号竖起来，和当时的输入两两对上记总账；行数多几条
        少几条只是"记的次数"不同，账本形状从头到尾没变，所以换一批不用重新开模具。
        
        """
        with T.Kernel(nb, nk) as (iy, ix):
            Ds = T.alloc_local((tc, bn), "float32")
            As = T.alloc_local((tc, bk), "float32")
            Cs = T.alloc_local((bn, bk), "float32")
            T.clear(Cs)
            for to in T.serial(T.ceildiv(m, tc)):
                T.copy(dY[to * tc, iy * bn], Ds)
                T.copy(A[to * tc, ix * bk], As)
                T.gemm(Ds, As, Cs, transpose_A=True)
            T.copy(Cs, GW[iy * bn, ix * bk])

    return dw_impl


def dw_asc_impl(dY, A, GW, n: int, k: int, tc: int, bn: int, bk: int):
    """昇腾 910B 方言正文（l0tr 主案）：dW=dYᵀ@A 的转置下移到 L1→L0 装填，mad 恒吃 NT。

    白话：卡上的乘加单元只接受"左边不翻、右边翻"这一种摆法。那就把要乘的两片按自然方向先
    搬进近便仓库（L1），临装进乘加寄存器（L0A/L0B）时才把它们竖起来（转置 DMA），账本
    (BN,BK) 与出口同朝向，算完整块直出到大表格——全程不在公共小格子（UB）上二次翻面，
    绕开 910B "立方体累加搬不回 UB" 的硬件限制。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")）。
    """
    import tilelang.ascend.language as T

    m = T.dynamic("m")
    nb, nk = n // bn, k // bk
    tiles = nb * nk
    blocks = min(tiles, NUM_BLOCKS)
    iters = (tiles + blocks - 1) // blocks

    @T.prim_func
    def dw_impl(dY: T.Tensor((m, n), "float32"), A: T.Tensor((m, k), "float32"),
                GW: T.Tensor((n, k), "float32")):
        """被追踪的那一层：出口表形状是常量，行方向串行累加、列方向按块摊开。

        白话：昇腾件这一份，一张大账本按列切成几页，每页自己把这一批的数一笔笔记上去；页数与
        每页多宽都定死，来多少行就记多少轮，不会把别页的数字串了行。
        """
        with T.Kernel(blocks) as bid:
            dy_l1 = T.alloc_l1((tc, bn), "float32")   # L1 自然 (token, n)
            x_l1 = T.alloc_l1((tc, bk), "float32")    # L1 自然 (token, k)
            a_l0 = T.alloc_l0a((bn, tc), "float32")   # NT 要求 (M=BN, K=TC)
            b_l0 = T.alloc_l0b((bk, tc), "float32")   # NT 要求 (N=BK, K=TC)
            acc = T.alloc_l0c((bn, bk), "float32")
            with T.Cube():
                for it in T.serial(iters):
                    tid = bid + it * blocks
                    n_tile = tid // nk
                    k_tile = tid % nk
                    # 归约轴 = 动态 token 数 m；末块不满 tc 触发 §12 asc_fill_l1（见模块 docstring）
                    for kt in T.serial(T.ceildiv(m, tc)):
                        T.copy(dY[kt * tc, n_tile * bn], dy_l1)
                        T.copy(A[kt * tc, k_tile * bk], x_l1)
                        T.copy(dy_l1, a_l0, transpose=True)   # 转置下移到 L1→L0 装填
                        T.copy(x_l1, b_l0, transpose=True)
                        T.gemm(a_l0, b_l0, acc, transpose_B=True, clear_accum=kt == 0)
                    T.copy(acc, GW[n_tile * bn, k_tile * bk])  # L0C→GM 直出，出口自然朝向

    return dw_impl


def _largest_block(total: int) -> int | None:
    """候选块宽里取能整除 total 的最大值；取不到即表达不了（返回 None 交回退）。"""
    for blk in BLOCK_LADDER:
        if total % blk == 0:
            return blk
    return None


def plan(dY: torch.Tensor, A: torch.Tensor, GW: torch.Tensor, target: str | None = None) -> dict[str, Any] | None:
    """这组形状能否交给方言模具：两输入同行数、出口 fp32、N/K 可整除、位宽在该 target 白名单内。

    白话：先看尺寸和记法合不合现成模具；不合就老实说干不了，由入口换普通做法，绝不硬凑。
    """
    name = ascend_env.normalize_target(target)
    if dY.dim() != 2 or A.dim() != 2 or dY.size(0) != A.size(0):
        return None
    if GW.dtype != torch.float32:
        return None
    n, k = int(GW.size(0)), int(GW.size(1))
    if dY.size(1) != n or A.size(1) != k:
        return None
    bn, bk = _largest_block(n), _largest_block(k)
    if bn is None or bk is None:
        return None
    if name == ascend_env.TARGET_CPU:
        if dY.dtype != torch.float32 or A.dtype != torch.float32:
            return None
    elif dY.dtype != A.dtype or dY.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None
    if not (dY.is_contiguous() and A.is_contiguous() and GW.is_contiguous()):
        return None
    kwargs = {"n": n, "k": k, "tc": BLOCK_TC, "bn": bn, "bk": bk}
    impl = dw_asc_impl if name == ascend_env.TARGET_ASCEND else dw_cpu_impl
    key = f"gemm_dw[{name}]|n{n}k{k}tc{BLOCK_TC}b{bn}x{bk}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl}


def run(dY: torch.Tensor, A: torch.Tensor, GW: torch.Tensor, spec: dict[str, Any]) -> bool:
    """真跑内核并把结果写进 GW；返回 False 表示模具没开成，须由入口层回退（不许假装算过）。

    缓存键不含行数：同一份产物服务不同批大小——权重梯度每步都要算，换批重编会把收益吃光。

    白话：模具只在第一次开，之后不管一批里有多少条样本都用同一个；开不了就如实报回去。
    """
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(
            spec["impl"](dY, A, GW, **spec["kwargs"]),
            out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(dY, A, GW)
    return True
