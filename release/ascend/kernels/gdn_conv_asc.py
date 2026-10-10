"""GDN 短卷积入口件：channels-first 的 depthwise 因果卷积（k=4）+ SiLU，昇腾侧走 910B 标量面。

【做什么】入口 `conv_forward(x, w, bias=None, out_dtype=None, target)`：
    y[b, c, t] = silu( bias[c] + Σ_{k=0..3} w[c, k] * x[b, c, t-3+k] )   （t-3+k < 0 视作 0）
即 Qwen3-Next / Mamba 系 GDN 里那条"混完 qkv 先卷一把短窗、再上 SiLU"的 conv 位点：
每通道一个私有 4 抽头滤波器（depthwise）、只看过去（causal 左补零 3 格）。
【怎么做】① 载体与递推件同一条 910B 路：`T.Kernel(cores)` + `T.Vector()` + 纯 `T.serial`
   （通道块 → 批次 → 时间步 → 窗内 4 抽头），**零** `T.SimtVF`/`T.Parallel`/`T.Pipelined`；
   通道数 C、窗宽、批数是编译期常量、长度 L 留动态维（缓存键不含 L，与递推件同口径）；C 必须
   能被核块数均分，挑不出整除就回落核数（与 `gdn_asc` 共用一套挑选口径）。② 两档：`scalar`
   （首推，直读 GM 标量面，零 UB、零 MTE——C 波真机数值 6.35e-08 就是这一档）与 `ubstage`
   （整行经 MTE2 进 UB、滑窗读 UB、MTE3 回写：多走搬运面，G-C4/G-C7 的单位/stride 未证真，
   且动态长度下要静态开 UB，本波只当保留档）。③ SiLU 走 `z/(1+exp(-z))`，exp 落到 §11 软件
   expf；`rational` 是"exp 面缺失时保编译"的兜底近似（**不是**对拍真值，别拿它当尺子）。
   ④ 因果左边界用 `T.if_then_else(idx>=0, X[..., max(idx,0)], 0)`：地址先夹到 0，绝不出负下标。
【为什么】这个位点在接口 home 里原本是"在上游之外"——没有入口，落到 torch 的 Conv2D 路径上，
   正是 R7 那一崩（D-int2(b)）。本波把 C 件已证的 910B 形态接成一个正经入口（形状/判据/回退
   口径与其余方言件一致），P2 三栈联调才有东西可替换。被否方案一：继续挂在上游 torch 卷积上
   ——那条路在 910B 上没有自研件可替换，P2 的"GDN 件替换 Conv2D 路径"就成空话；被否方案二：
   把卷积并进递推件 `gdn_asc.py`——递推件吃的是**已经卷过、已经归一**的 q/k/v，两者入参形状
   （(C,L) 通道优先 vs (heads,seq,dk) 时间优先）与生命周期都不同，混在一起会让 plan 的判据变成
   两套语义的并集、出事时无法归因，故分居姊妹件、由同一入口件（`gdn_kernel`）导出。
"""
import os

from typing import Any

import torch
import tilelang

from ascend.kernels import ascend_env, gdn_asc

#: GDN 约定的短卷积窗宽（conv_kernel=4）；别的窗宽本件不开模，直接落回普通写法
CONV_KERNEL = 4
#: SiLU 两种发射面：exp=真值口径（首推）；rational=保编译兜底（非对拍真值）
SILU_MODES = ("exp", "rational")
#: 昇腾侧档位：scalar=直读 GM 标量面（首推，C 波真机数值绿的就是它）；ubstage=整行进 UB 再滑窗
ASC_VARIANTS = ("scalar", "ubstage")
ENV_VARIANT = "SYS1_GDN_CONV_VARIANT"
ENV_SILU = "SYS1_GDN_CONV_SILU"


def _variant(variant: str | None = None) -> str:
    """挑 conv 的昇腾档位：显式参数 > env > scalar；认不得的写法直接报错，不许悄悄换档。"""
    raw = (variant or os.environ.get(ENV_VARIANT) or "scalar").strip().lower()
    if raw not in ASC_VARIANTS:
        raise ValueError(f"未知 conv 档位 variant={raw!r}，只支持 {'/'.join(ASC_VARIANTS)}")
    return raw


def _silu_mode(silu: str | None = None) -> str:
    """挑 SiLU 的发射面（同上，默认 exp=真值口径）。"""
    raw = (silu or os.environ.get(ENV_SILU) or "exp").strip().lower()
    if raw not in SILU_MODES:
        raise ValueError(f"未知 silu 发射面 silu={raw!r}，只支持 {'/'.join(SILU_MODES)}")
    return raw


def _silu(T, z, mode: str):
    """SiLU 的两种发射面（与 attempts/C 同一段算式，逐字对齐口径；T 由调用方正文传入）。"""
    if mode == "exp":
        # z * sigmoid(z) = z / (1 + exp(-z))；T.exp(float32) → codegen expf（compat §11）
        return z / (T.float32(1.0) + T.exp(T.float32(0.0) - z))
    # 兜底（保编译用）：sigmoid(u) ≈ 0.5*(u/(1+|u|)+1)，只用四则 + fabs
    u = T.float32(0.0) - z
    return z * (T.float32(0.5) * (u / (T.float32(1.0) + T.abs(u)) + T.float32(1.0)))


def conv_cpu_impl(X, W, Bias, Y, chans: int, k: int, batch: int):
    """CPU(c) 正文：一核串行扫全部通道×批次×时间步，逐格直读直写。

    白话：一人一桌，从第一个通道开始，一格一格把 4 抽头加出来、加偏置、过 SiLU，再写回去；
    左边不够三格的那几步，缺的当 0。这里不图快，只给 target=cpu 留一条与内核同口径的语义路。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile）。
    """
    import tilelang.cpu.language as T

    length = T.dynamic("length")
    pad = k - 1

    @T.prim_func
    def conv_impl(X: T.Tensor((batch, chans, length), "float32"),
                  W: T.Tensor((chans, k), "float32"),
                  Bias: T.Tensor((chans,), "float32"),
                  Y: T.Tensor((batch, chans, length), "float32")):
        """被追踪的那一层：通道数、窗宽、批数都是常量，只有长度留成活的。

        白话：模具在这一层定型——多少通道、几个抽头、几批写死，这一批有多长运行时再说。
        """
        with T.Kernel(1) as bx:  # noqa: F841  方言要求块索引存在，本件一核扫到底
            acc = T.alloc_var("float32")
            xv = T.alloc_var("float32")
            bi = T.alloc_var("float32")
            for ch in T.serial(chans):
                bi = Bias[ch]
                for b in T.serial(batch):
                    for t in T.serial(length):
                        acc = T.float32(0.0)
                        for kk in T.serial(k):
                            idx = t - pad + kk
                            xv = T.if_then_else(idx >= 0, X[b, ch, T.max(idx, 0)], T.float32(0.0))
                            acc = acc + W[ch, kk] * xv
                        Y[b, ch, t] = _silu(T, acc + bi, "exp")

    return conv_impl


def conv_asc_impl(X, W, Bias, Y, chans: int, k: int, batch: int, cores: int,
                  variant: str = "scalar", silu: str = "exp"):
    """昇腾正文（910B / dav-2201 可编形态）：通道均分给核，每通道整行按 4 抽头滑窗。

    白话：把一摞通道均分给工人，每个工人守着自己那几个通道：逐条记录把 4 个抽头加出来、
    加上本通道的偏置、过一道 SiLU 写回去；最左边不够 3 格的那几步，缺的当 0。

    910B 纪律（凭据 attempts/C/c_gdn_conv_910b.py + RESULT.md，真机数值 6.35e-08）：
      * 只用 AIV 标量面 `T.Kernel` + `T.Vector()` + 纯 `T.serial`；不碰 SimtVF/Parallel/SimdVF
        ——该代际无 SIMT 硬件模型，那些件实测编不出（D-int1）。
      * 累加走 `T.alloc_var`（寄存器），不做 UB 整块 vector-value store（会撞 ScalarDcacheBypass
        的 "vector-valued buffer store not implemented"）。
      * `scalar` 档零 `alloc_shared`：留一个只写不读的死缓冲，codegen 就把它落成命名空句柄
        （G-C8），所以暂存全在手心（var）里；且**零 T.copy**，绕开 MTE2/MTE3 的未证真面。
      * `ubstage` 档才开 UB，并且 `T.copy` 要写**区间**（写 `X[b, ch, 0]` 会退化成只搬一个
        元素，`asc_copy_gm2ub_align` 根本不出现）；该档依赖 G-C4/G-C7 ⇒ 只在要验搬运面时打开。

    :return: 追踪好的 PrimFunc（交给 tilelang.compile(target="ascend")，P1-4 起本地实编判 .o）。
    """
    import tilelang.ascend.language as T

    length = T.dynamic("length")
    if chans % cores:
        raise ValueError(f"通道数 chans={chans} 必须能被 cores={cores} 均分")
    cb = chans // cores
    pad = k - 1

    @T.prim_func
    def conv_impl(X: T.Tensor((batch, chans, length), "float32"),
                  W: T.Tensor((chans, k), "float32"),
                  Bias: T.Tensor((chans,), "float32"),
                  Y: T.Tensor((batch, chans, length), "float32")):
        """被追踪的那一层：通道/窗宽/批数/核数/档位是常量，长度是活的。

        白话：每个工人领几个通道，领到的通道里一条一条走——4 抽头求和、加偏置、过 SiLU、写回。
        """
        with T.Kernel(cores) as core_id:
            acc = T.alloc_var("float32", T.float32(0.0))
            z = T.alloc_var("float32", T.float32(0.0))
            xv = T.alloc_var("float32", T.float32(0.0))
            wt = T.alloc_var("float32", T.float32(0.0))
            bi = T.alloc_var("float32", T.float32(0.0))

            if variant == "ubstage":
                w_ub = T.alloc_shared((k,), "float32")                  # 本通道的 4 抽头
                row_ub = T.alloc_shared((length + pad,), "float32")     # 左端 pad 个 0 = 因果补零
                out_ub = T.alloc_shared((length,), "float32")

            with T.Vector():
                for cblk in T.serial(cb):
                    ch = core_id * cb + cblk
                    bi = Bias[ch]

                    if variant == "ubstage":
                        for kk in T.serial(k):
                            w_ub[kk] = W[ch, kk]
                        for b in T.serial(batch):
                            for kk in T.serial(pad):
                                row_ub[kk] = T.float32(0.0)
                            T.copy(X[b, ch, 0:length], row_ub[pad:pad + length])
                            for t in T.serial(length):
                                acc = T.float32(0.0)
                                for kk in T.serial(k):
                                    acc = acc + w_ub[kk] * row_ub[t + kk]
                                z = acc + bi
                                out_ub[t] = _silu(T, z, silu)
                            T.copy(out_ub[0:length], Y[b, ch, 0:length])
                    else:
                        for b in T.serial(batch):
                            for t in T.serial(length):
                                acc = T.float32(0.0)
                                for kk in T.serial(k):
                                    idx = t - pad + kk
                                    # 左边界：idx<0 ⇒ 补零（地址先夹到 0，不出负下标）
                                    xv = T.if_then_else(idx >= 0, X[b, ch, T.max(idx, 0)],
                                                        T.float32(0.0))
                                    wt = W[ch, kk]
                                    acc = acc + wt * xv
                                z = acc + bi
                                Y[b, ch, t] = _silu(T, z, silu)

    return conv_impl


def plan(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, y: torch.Tensor,
         target: str | None = None, variant: str | None = None, blocks: int | None = None,
         silu: str | None = None) -> dict[str, Any] | None:
    """判据：x 是 (..., C, L) 通道优先、w 是 (C, 4) 或 Conv1d 的 (C,1,4)、全程 fp32 且连续。

    特别地：窗宽不是 4、中间维不是 1（不是 depthwise）、x 与 w 的通道数对不上、或 y 与 x
    不同形，一概不开模——本件只服务 GDN 那一类短卷积，不做通用 conv。昇腾侧另挑一次核块数：
    C 要能被均分，挑不出就退到 1 核。

    白话：先量体裁衣——袖子是不是每只单独一块布（depthwise）、窗是不是 4 格、左边是不是只补
    过去那三格；不是这一件衣服，就别硬上这台缝纫机。
    """
    name = ascend_env.normalize_target(target)
    if x.dim() < 2 or y.shape != x.shape:
        return None
    if w.dim() not in (2, 3) or int(w.size(0)) != int(x.size(-2)) or int(w.size(-1)) != CONV_KERNEL:
        return None
    if w.dim() == 3 and int(w.size(1)) != 1:
        return None
    chans, length = int(x.size(-2)), int(x.size(-1))
    if chans == 0 or length == 0:
        return None
    if bias is not None and (bias.dim() != 1 or int(bias.size(0)) != chans):
        return None
    for one in ((x, w, y) if bias is None else (x, w, bias, y)):
        if one.dtype != torch.float32 or not one.is_contiguous():
            return None
    kwargs: dict[str, Any] = {"chans": chans, "k": CONV_KERNEL,
                             "batch": int(x.numel() // (chans * length))}
    impl = conv_asc_impl if name == ascend_env.TARGET_ASCEND else conv_cpu_impl
    key = f"gdnconv[{name}]|c{chans}k{CONV_KERNEL}b{kwargs['batch']}"
    if name == ascend_env.TARGET_ASCEND:
        cores = gdn_asc._pick_cores(chans, gdn_asc._blocks_limit(blocks))
        var, mode = _variant(variant), _silu_mode(silu)
        kwargs.update({"cores": cores, "variant": var, "silu": mode})
        key += f"|c{cores}{var}{mode[0]}"
    return {"target": name, "key": key, "kwargs": kwargs, "impl": impl, "length": length}


def _args(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, y: torch.Tensor,
          chans: int, device: torch.device) -> tuple:
    """把 (x, w, bias, y) 整成模具要的四路：展平前导批维、补零偏置、w 去掉 depthwise 的中间 1。"""
    x3 = _flatten(x).contiguous()
    y3 = _flatten(y).contiguous()
    w2 = w.reshape(chans, CONV_KERNEL).contiguous() if w.dim() == 3 else w.contiguous()
    b1 = bias.contiguous() if bias is not None else torch.zeros(
        (chans,), dtype=torch.float32, device=device)
    return x3, w2, b1, y3


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """(..., C, L) → (batch, C, L)：前导轴全并进批轴（与递推件的头轴展平同一条路）。"""
    c, l = int(t.size(-2)), int(t.size(-1))
    return t.reshape(-1, c, l)


def run(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None, y: torch.Tensor,
        spec: dict[str, Any]) -> bool:
    """取（或首编）卷积模具并真跑一遍；False = 模具没开成，由入口回退普通写法。

    缓存键不含长度 L：同一份产物服务任意序列长度（与递推件、rope 同一口径）。bias 缺省时
    件内现领一片全零——模具的张量清单里 bias 永远在场，"没有 bias"只是接口的糖。

    白话：模具只开第一次；这一批有多长都照同一个走。偏置没给就当桌上那片是空的。
    """
    args = _args(x, w, bias, y, spec["kwargs"]["chans"], x.device)
    kernel = ascend_env.get_compiled(
        spec["key"],
        lambda: tilelang.compile(spec["impl"](*args, **spec["kwargs"]),
                                 out_idx=[], **ascend_env.compile_kwargs(spec["target"])),
        spec["target"])
    if kernel is None:
        return False
    kernel(*args)
    return True


def conv_forward(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None,
                 out_dtype: torch.dtype | None = None, target: str | None = None,
                 **spec_opts: Any) -> torch.Tensor:
    """GDN 短卷积前向：交回与 x 同形的 (..., C, L)；内核不可用时用同口径的普通写法。

    白话：卷一把短窗、加偏置、过 SiLU，长度不变（只看过去，不看未来）。

    :param w: (C, 4) 或 Conv1d 的 (C, 1, 4)（depthwise 的中间维必须是 1）。
    :param out_dtype: 交回的位宽；不给就按 fp32 交（内核与普通写法都先算 fp32 再收）。
    :raises ValueError: x 不是通道优先、或 w 的通道数/窗宽与件不符时抛出。
    """
    if x.dim() < 2:
        raise ValueError(f"x 至少二维 (..., C, L)，实得 {x.dim()} 维")
    chans, length = int(x.size(-2)), int(x.size(-1))
    if w.dim() == 3 and int(w.size(1)) != 1:
        raise ValueError(f"w 是三维时中间维必须为 1（depthwise），实得 {tuple(w.shape)}")
    if w.dim() not in (2, 3) or int(w.size(0)) != chans or int(w.size(-1)) != CONV_KERNEL:
        raise ValueError(f"w 需为 (C,{CONV_KERNEL}) 或 (C,1,{CONV_KERNEL}) 且 C={chans}，"
                         f"实得 {tuple(w.shape)}")
    if bias is not None and tuple(bias.shape) != (chans,):
        raise ValueError(f"bias 需为 ({chans},)，实得 {tuple(bias.shape)}")
    name = ascend_env.normalize_target(target)
    dtype = out_dtype or torch.float32
    x3, w3, b3 = _flatten(x).float(), w.float(), None if bias is None else bias.float()
    out = torch.empty(tuple(x3.shape), dtype=torch.float32, device=x.device)
    if ascend_env.active_backend(name, x.device) == ascend_env.TILELANG:
        spec = plan(x3, w3, b3, out, name, **spec_opts)
        if spec is not None and run(x3, w3, b3, out, spec):
            return out.to(dtype).reshape(*x.shape[:-2], chans, length)
    return _eager(x3, w3, b3).to(dtype).reshape(*x.shape[:-2], chans, length)


def _eager(x3: torch.Tensor, w2: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """回退路径与尺子：torch 的 depthwise 因果 conv1d + SiLU（fp32），与内核同口径。

    白话：没有模具时手工卷一遍——左边补三格 0、每通道用自己的 4 抽头、加偏置、过 SiLU，
    再把补出来的那三格剪掉。
    """
    import torch.nn.functional as F

    k, pad = int(w2.size(1)), int(w2.size(1)) - 1
    chans = int(w2.size(0))
    w4 = w2.reshape(chans, 1, k).float()
    b = None if bias is None else bias.float()
    xs = F.pad(x3.float(), (pad, 0))                       # 只在左边补零（因果）
    y = F.conv1d(xs, w4, b, groups=chans)
    # pad+valid 已经把长度带回 L（L+pad-k+1 = L），此处**不能再剪**，剪一刀就少 k-1 格
    return F.silu(y)


__all__ = ["ASC_VARIANTS", "CONV_KERNEL", "SILU_MODES", "conv_asc_impl", "conv_cpu_impl",
           "conv_forward", "plan", "run"]
