"""ascend 九件方言内核的**最小混合栈一步冒烟件**（甲路 P2-prep）。

【做什么】把 `ascend.autograd_asc` 的八个可微门面按训练时真正的拓扑串一遍——
`linear → add_ln → rope → attn_sw → linear → gdn_delta → lora_apply → readout`——
前向出 loss、`loss.backward()`、Adam 一步更新，然后打印判决行
`P2-STEP-<target> loss0=<x> loss1=<y>`：本波只验**接线正确性**（loss 是否下降、
是否每个可训参数都拿到非 None 的梯度），**不验内核数值 rel/V***。

【怎么做】① 参数量任意小（d=32, seq=16, heads=2, dim=16），CPU 上毫秒级；② 所有
可微点走 `autograd_asc` 门面（不吃 raw torch ops），确保"如果哪件反向没接对，梯度就
是 None"这条硬判据真能触发；③ 后端由 `--target {cpu, ascend}` 或环境变量
`DMLAYA_ASCEND_TARGET` 决定，卡上只翻 device 与 `--target ascend`，代码不动；④ 用
Adam（`lr=1e-2`）——尺度不变的更新对**单步是否下降**最稳（SGD 在小图上容易因梯度
量级过大一步冲过头）；⑤ 判决前必做**全参非 None 检查**，任一 None 就直接 fail 输出
参数名，方便定位到具体门面。

【为什么】此前 `ascend/` 无任何 `autograd.Function`、`*_kernel` 无模型消费方——R11
消费者枚举缺口。本件就是把消费者接上：**host 上跑通即证明拓扑与反向公式对；卡窗只翻
`--target ascend` 就能真跑一步**。被否方案一：直接在 sys1/model.py 上换后端——把 P1
冻结件拽进 P2 波次，破坏"接口稳定 + 内核合流"的层次；被否方案二：只写单测不写可运行
脚本——上卡那天还得重搭一遍 main，等于把卡窗时间花在写脚本上。

不接的位点（本波边界，如实登记，不假绿）：
- `gdn_conv` 门面不在本件里跑——`gdn_kernel.conv_forward` 无 conv_backward 入口，
  本波 `_GdnConvFn` 的 fp32 闭式兜底已由 `test_ascend_train_step.py` 的 gradcheck
  单独覆盖；把它塞进训练图会引入一条与内核不同源的反向链，混淆"接线对不对"的判据。
- 数值精度域（rel、V*）由 `test_ascend_gradcheck.py` 覆盖，与本件解耦。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# 允许 `python ascend/train_step.py` 直接跑（用户交付指令里的调用形式）：
# 把 release/ 挂进 sys.path，让 import 语义与 `python -m ascend.train_step` / pytest 一致。
_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from ascend import autograd_asc as ops  # noqa: E402
from ascend.kernels import add_ln_kernel, ascend_env  # noqa: E402


def _make_inputs(seq: int, d: int, device: torch.device, gen: torch.Generator):
    """造一批固定种子的输入与目标：seq 个 token、宽度 d；目标是随机投影（回归任务）。

    白话：数据不进优化器（不参与 requires_grad），只当"考卷"；标准答案也一次造好、
    训练全程不变——这样 loss 是否真的下降，只由参数与反向决定，没有噪声可推诿。
    """
    x0 = torch.randn(seq, d, device=device, generator=gen)
    target = torch.randn(seq, d, device=device, generator=gen)
    return x0, target


def _build_params(seq: int, d: int, heads: int, dim: int, rope_theta: float,
                  device: torch.device, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """按拓扑所需准备所有 requires_grad 的原料（不含 cos/sin 与 ids，那三样是常量）。

    白话：`nn.Parameter` 那种"包一层类"的做法留给正式训练；本件只要一张扁平字典，
    优化器直接吃 list(d.values())，读代码时"哪一位是参数"一眼可数，不藏着掖着。
    """
    def _r(*shape, scale: float = 1.0):
        return (torch.randn(*shape, device=device, generator=gen) * scale).requires_grad_(True)

    p: dict[str, torch.Tensor] = {
        "W1": _r(d, d, scale=0.1), "b1": _r(d, scale=0.01),
        "ln_g": torch.ones(d, device=device).requires_grad_(True), "ln_b": _r(d, scale=0.01),
        "Wqkv": _r(3 * heads * dim, d, scale=0.1), "bqkv": _r(3 * heads * dim, scale=0.01),
        "Wout": _r(d, heads * dim, scale=0.1), "bout": _r(d, scale=0.01),
        # GDN 支路：从 proj 再投一次得到 q/k/v/g/beta，全走 linear 门面
        "Wgq": _r(heads * dim, d, scale=0.1), "Wgk": _r(heads * dim, d, scale=0.1),
        "Wgv": _r(heads * dim, d, scale=0.1), "Wgg": _r(heads * dim, d, scale=0.1),
        "Wgb": _r(heads, d, scale=0.1),
        "Wout_g": _r(d, heads * dim, scale=0.1),
        # LoRA 旁路
        "lora_a": _r(8, d, scale=0.05), "lora_b": _r(d, 8, scale=0.05),
    }
    cos = torch.ones(seq, dim // 2, device=device) * (rope_theta ** 0)   # 角度表本件不参与训练
    sin = torch.zeros(seq, dim // 2, device=device)
    # 真正的 RoPE 角度：位置 pos 与半维 j → θ·pos/(10000^(2j/dim))
    pos = torch.arange(seq, device=device).float().unsqueeze(1)
    j = torch.arange(dim // 2, device=device).float().unsqueeze(0)
    inv = rope_theta / (10000.0 ** (2.0 * j / dim))
    angle = pos * inv
    return p, torch.cos(angle), torch.sin(angle)


def _forward(x0: torch.Tensor, p: dict[str, torch.Tensor], cos: torch.Tensor,
             sin: torch.Tensor, seq: int, d: int, heads: int, dim: int,
             target: str) -> torch.Tensor:
    """按拓扑把八个门面都过一遍，交回 loss（fp32、MSE）。

    白话：一层一层照着"乘加→残差+整形→打包 qkv→旋转→切三摞→注意力→投回 d→
    走一遍 GDN→再投回 d→合流→LoRA 旁路→读出"排下来；每一站都走门面、不走 raw
    torch，反向是否接对全看梯度能不能回来——空接会当场报错，不会悄悄通过。
    """
    # 1) linear：x0 (seq, d) → h_lin (seq, d)
    h_lin = ops.linear(x0, p["W1"], p["b1"], out_dtype=torch.float32, target=target)
    # 2) add_ln：把 x0 当残差合进来 → (y, h)
    y, _h = ops.add_ln(h_lin, x0, p["ln_g"], p["ln_b"], eps=add_ln_kernel.DEFAULT_EPS,
                       out_dtype=torch.float32, target=target)
    # 3) linear 到 packed-qkv：(seq, 3·heads·dim)
    qkv_p = ops.linear(y, p["Wqkv"], p["bqkv"], out_dtype=torch.float32, target=target)
    # 4) reshape 到 rope 契约形状 (tokens, 3, heads, dim)
    qkv = qkv_p.reshape(seq, 3, heads, dim)
    # 5) rope（内核就地改，门面内部 clone，训练原件安全）
    qkv = ops.rope(qkv, cos, sin, rotate_slots=rope_slots_default(), target=target)
    # 6) 拆三摞：(heads, seq, dim)
    q = qkv[:, 0].transpose(0, 1).contiguous()
    k = qkv[:, 1].transpose(0, 1).contiguous()
    v = qkv[:, 2].transpose(0, 1).contiguous()
    # 7) 因果滑窗注意力（window ≥ seq ⇒ 等价全因果）
    o = ops.attn_sw(q, k, v, window=seq, out_dtype=torch.float32, target=target)
    # 8) 展平 → linear → proj (seq, d)
    o_flat = o.transpose(0, 1).reshape(seq, heads * dim)
    proj = ops.linear(o_flat, p["Wout"], p["bout"], out_dtype=torch.float32, target=target)
    # 9) GDN 递推支路：从 proj 再投一次得 q/k/v/g/beta，跑 gdn_delta，再 linear 回 d
    qg = ops.linear(proj, p["Wgq"], None, out_dtype=torch.float32, target=target) \
        .reshape(seq, heads, dim).transpose(0, 1).contiguous()
    kg = ops.linear(proj, p["Wgk"], None, out_dtype=torch.float32, target=target) \
        .reshape(seq, heads, dim).transpose(0, 1).contiguous()
    vg = ops.linear(proj, p["Wgv"], None, out_dtype=torch.float32, target=target) \
        .reshape(seq, heads, dim).transpose(0, 1).contiguous()
    # 门：每键通道的对数衰减 ⇒ 取负 softplus，恒 ≤ 0
    gate = ops.linear(proj, p["Wgg"], None, out_dtype=torch.float32, target=target) \
        .reshape(seq, heads, dim).transpose(0, 1)
    gate = -torch.nn.functional.softplus(gate).contiguous()
    # beta：每 (head, token) 一个标量 ∈ (0, 1)，写成 sigmoid
    beta = ops.linear(proj, p["Wgb"], None, out_dtype=torch.float32, target=target) \
        .reshape(seq, heads).transpose(0, 1).sigmoid().contiguous()
    out_g = ops.gdn_delta(qg, kg, vg, gate, beta, out_dtype=torch.float32, target=target)
    out_g_flat = out_g.transpose(0, 1).reshape(seq, heads * dim)
    gdn_out = ops.linear(out_g_flat, p["Wout_g"], None,
                         out_dtype=torch.float32, target=target)
    mix = proj + gdn_out                              # 融合支路（普通加法，走 autograd 图）
    # 10) LoRA 旁路：delta = s·(x@Aᵀ)@Bᵀ；s=1.0
    delta = ops.lora_apply(mix, p["lora_a"], p["lora_b"], scaling=1.0, base=None,
                           out_dtype=torch.float32, target=target)
    h_final = mix + delta
    # 11) 读出：ids 从 h_final 里逐行抽出（等价 identity，但走内核 gather/scatter 路）
    ids = torch.arange(seq, dtype=torch.int64, device=h_final.device)
    logits = ops.readout(h_final, ids, out_dtype=torch.float32, target=target)
    return logits


def rope_slots_default() -> tuple[int, ...]:
    """rope 的槽位默认走 rope_kernel.DEFAULT_SLOTS，与 P1 / ascend 入口同源。"""
    from ascend.kernels import rope_kernel
    return rope_kernel.DEFAULT_SLOTS


def run(target: str, device: torch.device, seed: int = 20261010,
        lr: float = 1e-2, verbose: bool = True) -> tuple[float, float, list[str]]:
    """跑完整一步：前向→backward→检查全参梯度→Adam.step→再前向一次拿 loss1。

    白话：一次前向只够算梯度，不看更新效果；判据要"更新完再前向一次"的 loss1 才算
    一步真的走下去了。全参非 None 检查放在 optimizer.step 之前——step 一旦执行就
    把参数改了，届时"某个梯度漏接"变成"某个参数被更新了一小步"，反而看不出问题。
    返回 (loss0, loss1, missing) 三元组；missing 非空即失败。
    """
    torch.manual_seed(seed)
    gen = torch.Generator(device=device.type)
    gen.manual_seed(seed)

    seq, d, heads, dim = 16, 32, 2, 16
    x0, target_t = _make_inputs(seq, d, device, gen)
    p, cos, sin = _build_params(seq, d, heads, dim, rope_theta=1.0, device=device, gen=gen)

    # 前向 #0：拿 loss0
    logits0 = _forward(x0, p, cos, sin, seq, d, heads, dim, target)
    loss0 = (logits0 - target_t).pow(2).mean()

    # 反向
    for t in p.values():
        if t.grad is not None:
            t.grad = None
    loss0.backward()

    missing = [name for name, t in p.items() if t.grad is None]
    if missing:
        raise RuntimeError(f"以下参数 grad=None，反向接线未到位：{missing}")

    opt = torch.optim.Adam(list(p.values()), lr=lr)
    opt.step()

    # 前向 #1：拿 loss1（同一个 x0/target）
    logits1 = _forward(x0, p, cos, sin, seq, d, heads, dim, target)
    loss1 = (logits1 - target_t).pow(2).mean()

    l0, l1 = float(loss0.detach().item()), float(loss1.detach().item())
    if verbose:
        print(f"P2-STEP-{target} loss0={l0:.6f} loss1={l1:.6f} "
              f"reduced={l1 < l0} params_grad_ok={not missing}")
    return l0, l1, missing


def main(argv: list[str] | None = None) -> int:
    """命令行入口：`--target {cpu, ascend}` 选后端，判决不通过 exit 1（供 CI 硬门）。"""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default=None, choices=list(ascend_env.TARGETS),
                    help=f"编译/发射后端；未指定时读 env {ascend_env.ENV_TARGET}（默认 cpu）")
    ap.add_argument("--lr", type=float, default=1e-2, help="Adam 学习率（默认 1e-2）")
    ap.add_argument("--seed", type=int, default=20261010, help="随机种子（默认 20261010）")
    ap.add_argument("--device", default=None,
                    help="torch 设备；cpu 后端默认 cpu，ascend 后端默认 npu（可覆盖）")
    args = ap.parse_args(argv)

    name = ascend_env.normalize_target(args.target)
    dev_default = "npu" if name == ascend_env.TARGET_ASCEND else "cpu"
    dev = torch.device(args.device or dev_default)
    if name == ascend_env.TARGET_ASCEND and dev.type != "npu":
        print(f"WARN: target=ascend 但 device={dev}，ascend_env.active_backend 会判成 torch_eager",
              file=sys.stderr)

    try:
        l0, l1, missing = run(target=name, device=dev, seed=args.seed, lr=args.lr)
    except Exception as exc:  # noqa: BLE001  顶层脚本：把 traceback 打出来再退码
        print(f"P2-STEP-{name} FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    if missing:
        print(f"P2-STEP-{name} FAIL: grad 缺失 {missing}", file=sys.stderr)
        return 2
    if not (l1 < l0):
        print(f"P2-STEP-{name} FAIL: loss 未下降（{l0} → {l1}）", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
