"""gemm 方言件：C = act(A @ Wᵀ + bias)，行数 m 是动态符号，一次编译服务任意批大小。

【做什么】同一套乘加口径写两份方言正文：ascend 那份按 910B 立方体单元（Cube 单工做主乘加、
UB 放累加、标量 serial 收尾）摆；cpu 那份按本机真跑得通的路径摆（列块静态网格 + 行块动态
串行循环）。两份都由 plan() 判形状、run() 负责取编译产物并就地写出口，位宽转换不在方言里做
（出口一律 fp32）。
【怎么做】① 件形是"张量参数在前、整数常量在后"的工厂函数；run() 现场调用工厂拿到追踪好的
   PrimFunc，再交给 tilelang.compile 出产物并按形状键缓存。刻意**不用 tilelang.jit**：这种写法
   属 lazy 风格，jit(...) 只回 Kernel 对象不执行（实测出口保持全零），而它的 call-form 缓存拿
   张量做 `==` 比较上次调用（jit/__init__.py 的 _CallFormCache._matches_last），换一批新张量就
   RuntimeError。② 本文件不写 `from __future__ import annotations`——tilelang 取 prim_func 的
   张量注解时只认闭包非局部名（eager/builder.py 的 get_func_nonlocals），一旦 PEP 563 把注解变成
   字符串，只在注解里出现的形状常量就没有闭包单元，编译期报 NameError: name 'n' is not defined。
   ③ 动态维 m 的去处按后端分：CPU(c) 后端**网格上界必须是编译期常量**，把 m 写进网格会一次都不
   发射（实测 C 全零、误差恰好等于 |A@Wᵀ| 最大值），故 CPU 侧用"列块静态网格 + 行块
   T.serial(ceildiv(m, bm)) 动态循环"；昇腾侧同样只用常量网格（固定 NUM_BLOCKS 个立方体核块），
   m 落在 T.serial 的行块循环上界。④ 收尾（加偏置、按需压负为零）在 UB 累加缓冲上逐元素做，
   最后整块写回；尾块越界交给 T.copy 的自动边界谓词（实测 m=8 对 bm=16 误差为 0，不需手写守卫）。
【为什么】三处口径是被约束逼出来的，不是风格：(a) 权重按 (N,K) 存、必须 transpose_B=True——昇腾
   L1 输入的乘加路只认这个方向（手册与 TileKernels 的 GEMM 件都这么摆）；(b) 出口不在方言里降
   位宽——昇腾 DMA 搬运不允许顺带转类型（tilelang/ascend/analysis/vf_checker.py 明确
   "DMA copies cannot perform type casting"），CPU 侧实测把 fp32 块直接 copy 成 fp16 全局会撞
   生成代码里的 `no matching conversion for C-style cast from 'float4' to 'half4'`，故两边统一
   fp32 出口、入口层降位；(c) 被否方案：CPU 侧也用 T.Parallel 收尾——实测本方言的并行块索引
   local 缓存会被语义检查拒绝（"Local buffer ... is thread-private"），serial 才成立；(d) 被否
   方案：CPU 侧照 P1 的 Metal 件那样把 m 写进网格——Metal 认动态网格，c 后端不认，照搬会得到
   "跑得通、数是零"的静默错，比直接报错更危险，故按后端拆开摆法；(e) 910B 侧（P1-4 合流）：
   早先照 950 载体写的 SimtVF/T.Parallel 收尾在 patched 910B tilelang 下 COMPILE-FAIL（方言里
   根本没有 SimtVF/Parallel），且累加若放 L0C 则搬不回 UB——910B 硬件 fixpipe L0C→UB 不支持
   （port910b 卷宗 b10：FixpipeL0C2UBImpl=assert(false)）。故把操作数与累加都落到 UB(shared)，
   主乘加走 Cube 单工，收尾标量 T.serial，fp32 的 T.clear 也被首轮 clear_accum 取代（fp32 T.clear
   在 UB 上生成坏 float2 store）。云端待验：cube 面单位/V1-V3、清账口径的数值 rel（本波只判编译）。
"""
import tilelang

from typing import Any

import torch

from ascend.kernels import ascend_env

#: 内核侧真正落地的截断口径；其余口径由入口层走普通写法（与 P1 同名同取值）
SUPPORTED_ACTS = ("none", "relu")
ACT_MODE = {"none": 0, "relu": 1}

#: M 轴块高：行块太大会浪费并行度（批大小可能等于 1）
BLOCK_M = 16
#: N/K 轴候选块宽，从大到小取能整除的那个；都不能整除则该形状表达不了
BLOCK_LADDER = (64, 32, 16)
#: 昇腾侧一次发射占用的立方体核块数与流水深度（云端调优项，本地只固定写法）
NUM_BLOCKS = 8
VEC_THREADS = 64
NUM_STAGES = 2


def gemm_cpu_impl(A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int, act_mode: int):
    """CPU(c) 方言正文：全程 fp32，出口 fp32；网格只铺编译期常量列块，行块用动态上界串行推。

    白话：竖着切成固定的几列，每一列自己从上往下把行一段段搬进小格子算；有多少行都不影响切法，
    算完先在公共小格子里加加成、把负数压成零，再整块写回大表格。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    m = T.dynamic("m")
    nb = n // bn  # 列块数：n 已被 plan 判过整除，是编译期常量

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), "float32"), W: T.Tensor((n, k), "float32"),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), "float32")):
        """被追踪的那一层：形状在这里落定，只有行数 M 是活的。

        白话：CPU 件这一份，模具在这一层定型——列切成几块、每块多宽、要不要压负数都写死，
        只有"这批有多少行"留成活的，运行时多长都按同一块模具走，行与行之间不串。
        
        """
        with T.Kernel(nb) as mx:
            As = T.alloc_local((bm, bk), "float32")
            Ws = T.alloc_local((bn, bk), "float32")
            Cs = T.alloc_local((bm, bn), "float32")
            for my in T.serial(T.ceildiv(m, bm)):
                T.clear(Cs)
                for ko in T.serial(T.ceildiv(k, bk)):
                    T.copy(A[my * bm, ko * bk], As)
                    T.copy(W[mx * bn, ko * bk], Ws)
                    T.gemm(As, Ws, Cs, transpose_B=True)
                for i in T.serial(bm):
                    for j in T.serial(bn):
                        Cs[i, j] = Cs[i, j] + bias[mx * bn + j]
                        if act_mode == 1:
                            if Cs[i, j] < 0:
                                Cs[i, j] = 0
                T.copy(Cs, C[my * bm, mx * bn])

    return gemm_impl


def gemm_asc_impl(A, W, bias, C, n: int, k: int, bm: int, bn: int, bk: int, act_mode: int):
    """昇腾 910B 方言正文：Cube 单工做主乘加、UB 上放累加、标量 serial 收尾（加偏置/压负）。

    白话：把乘加交给核上的立方体单元连着算，账先记在同一块公共小格子里（UB），算完就地用
    标量把偏置补上、按需把负数压成零，再整块写回大表格。首轮靠"从头记"的标志位清零，不
    单开清账步；也不碰 L0C/L1 那两层——910B 上 L0C 搬不回 UB（硬件 fixpipe 不支持），
    收尾只能在 UB 上做，故操作数与累加都摆在 UB。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")）。
    """
    import tilelang.ascend.language as T

    m = T.dynamic("m")
    nb = n // bn

    @T.prim_func
    def gemm_impl(A: T.Tensor((m, k), "float32"), W: T.Tensor((n, k), "float32"),
                  bias: T.Tensor((n,), "float32"), C: T.Tensor((m, n), "float32")):
        """被追踪的那一层：网格固定为若干个立方体核块，行数 m 是活的。

        白话：同一本账换到立方体单元上摆——两个操作数和累加都放公共小格子，一次搬多宽、
        竖着切几列都定死，有多少行都不影响切法；末尾不够一整块的部分交给搬运自带的那道闸
        拦住（越界写回被谓词挡下），不需手写守卫。
        """
        with T.Kernel(NUM_BLOCKS) as bx:
            a = T.alloc_shared((bm, bk), "float32")
            w = T.alloc_shared((bn, bk), "float32")
            cl = T.alloc_shared((bm, bn), "float32")
            with T.Cube():
                for chunk in T.serial(T.ceildiv(m, bm * NUM_BLOCKS)):
                    my = chunk * NUM_BLOCKS + bx
                    for mx in T.serial(nb):
                        # K 分块连着乘加，首轮 clear_accum 清零，后续续在同一个 UB 格子里
                        for ko in T.serial(T.ceildiv(k, bk)):
                            T.copy(A[my * bm, ko * bk], a)
                            T.copy(W[mx * bn, ko * bk], w)
                            T.gemm(a, w, cl, transpose_B=True, clear_accum=ko == 0)
                        # 收尾标量化：加偏置、按需压负为零（910B 无 SimtVF/T.Parallel）
                        for i in T.serial(bm):
                            for j in T.serial(bn):
                                v = cl[i, j] + bias[mx * bn + j]
                                if act_mode == 1:
                                    if v < 0:
                                        v = 0.0
                                cl[i, j] = v
                        T.copy(cl, C[my * bm, mx * bn])

    return gemm_impl


def _largest_block(total: int) -> int | None:
    """在候选块宽里取能整除 total 的最大值；取不到说明这个维度表达不了（返回 None）。"""
    for blk in BLOCK_LADDER:
        if total % blk == 0:
            return blk
    return None


def plan(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, C: torch.Tensor,
         act: str, target: str | None = None) -> dict[str, Any] | None:
    """判断这组形状/口径能不能交给方言内核；能则返回编译参数与缓存键，否则返回 None（不硬凑）。

    硬条件三条：截断口径在支持表里；N/K 都能被某个候选块宽整除；位宽组合在该 target 的白名单里
    （CPU 侧只收 fp32，昇腾侧收 fp32/bf16/fp16 同宽操作数 + fp32 偏置与出口）。任一不满足就回
    None，由入口层换普通写法。

    白话：先看这活儿的尺寸和记法合不合现成模具；不合就不勉强开模，直接换手工做法。
    """
    name = ascend_env.normalize_target(target)
    if act not in SUPPORTED_ACTS:
        return None
    n, k = int(W.size(0)), int(W.size(1))
    bn, bk = _largest_block(n), _largest_block(k)
    if bn is None or bk is None:
        return None
    if C.dtype != A.dtype or bias is not None and bias.dtype != torch.float32:
        return None
    if name == ascend_env.TARGET_CPU and A.dtype != torch.float32:
        return None
    if name == ascend_env.TARGET_ASCEND and A.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None
    if A.dtype != W.dtype or not (A.is_contiguous() and W.is_contiguous() and C.is_contiguous()):
        return None
    kwargs = {"n": n, "k": k, "bm": BLOCK_M, "bn": bn, "bk": bk, "act_mode": ACT_MODE[act]}
    impl = gemm_asc_impl if name == ascend_env.TARGET_ASCEND else gemm_cpu_impl
    key = f"gemm[{name}]|n{n}k{k}b{BLOCK_M}x{bn}x{bk}a{act}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "shape": (n, k)}


def run(A: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None, C: torch.Tensor,
        spec: dict[str, Any]) -> bool:
    """真跑一次内核并把结果写进 C；编译产物取不到就返回 False（由入口层落回退）。

    缓存键不含行数：同一份产物服务 8、33、8192 等不同批大小（m 是 PrimFunc 里的动态符号），
    换批重编的秒级开销会吞掉收益。首编用当次张量做形状追踪，之后只喂出口缓冲。

    白话：模具没开成就说没开成，返回一个假字让上层换手工做法，绝不"假装算过"把空表交出去。

    :returns: True 表示 C 已被内核写入。
    """
    zero = bias if bias is not None else torch.zeros(spec["shape"][0], dtype=torch.float32,
                                                     device=A.device)
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](A, W, zero, C, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(A, W, zero, C)
    return True
