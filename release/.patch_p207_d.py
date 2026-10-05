"""p2-07 layers/attention.py 定点修订：全注意力层不能传 visible=None（主干的 None 分支不加因果掩码）。"""
from pathlib import Path

P = Path("sys1/layers/attention.py")
s = P.read_text()


def rep(old, new, label):
    global s
    assert s.count(old) == 1, f"锚点失配[{label}]: 命中 {s.count(old)} 次"
    s = s.replace(old, new, 1)
    print("ok", label)


# ① layer_visibles：全注意力层回"纯因果带"而不是 None
rep('''    """逐层的可见位清单；全注意力层给 None（由模型侧走它自己的全因果路）。"""
    windows = plan.windows if isinstance(plan, LongContextPlan) else tuple(plan)
    return [
        None if w is None else sliding_visible(seq, window=w, attention_mask=attention_mask,
                                               batch=batch, device=device)
        for w in windows
    ]''',
    '''    """逐层的可见位清单：滑窗层给窄带，全注意力层给"纯因果带"。

    白话：这里刻意**不**给 None。主干 `CausalAttention` 的 `visible=None` 分支是"任何掩码
    都不加"（双向可见），并不是"全因果"——只有 `Decoder.forward` 才会替它补上因果带。
    逐层清单要能直接喂进 `DecoderBlock`，所以"看全场"这一层也得把因果带画出来。
    """
    windows = plan.windows if isinstance(plan, LongContextPlan) else tuple(plan)
    return [
        sliding_visible(seq, window=w, attention_mask=attention_mask, batch=batch, device=device)
        for w in windows
    ]''',
    "layer_visibles_causal")

# ② forward_with_windows：全注意力层的兜底可见位
rep('''    cos, sin = decoder._tables(seq, input_ids.device)          # 与主干同一张角度表（同真源）
    visible_all = None if attn_mask is None else _full_visible(seq, attn_mask, batch, input_ids.device)
    x = decoder.tok_emb(input_ids)
    for block, w in zip(decoder.blocks, windows):
        vis = visible_all if w is None else sliding_visible(
            seq, window=w, attention_mask=attn_mask, batch=batch, device=input_ids.device)
        x = block(x, cos, sin, vis)
    return decoder.final_norm(x)''',
    '''    cos, sin = decoder._tables(seq, input_ids.device)          # 与主干同一张角度表（同真源）
    if attn_mask is None:
        # 全注意力层用的"纯因果带"，与主干 _visible_mask(seq, None, ...) 逐格同值
        visible_all = window_band(seq, None, device=input_ids.device)
    else:
        visible_all = _full_visible(seq, attn_mask, batch, input_ids.device)
    x = decoder.tok_emb(input_ids)
    for block, w in zip(decoder.blocks, windows):
        vis = visible_all if w is None else sliding_visible(
            seq, window=w, attention_mask=attn_mask, batch=batch, device=input_ids.device)
        x = block(x, cos, sin, vis)
    return decoder.final_norm(x)''',
    "forward_full_causal")

# ③ 模块头的"接进模型"说明与 docstring 里那句 None 的说法同步修正
rep('''    全注意力层传 `None` 时走 `CausalAttention` 的"无掩码"分支，那条分支等价于全因果，
    与 `sys1.model._visible_mask` 在 `attn_mask=None` 时的结果一致；一旦调用方给了批内
    补洞掩码，全注意力层也补一张全因果 + 空洞不可见的纸，语义与主干严格对齐。
    """''',
    '''    窗口 `None`（全注意力）层的可见位取"纯因果带"：与主干 `Decoder.forward` 自己在
    `attn_mask=None` 时算出的那张纸逐格同值（`visible=None` 在主干里是"不加任何掩码"的
    双向分支，绝不能拿来当全注意力，否则因果性当场破了）。副作用是滑窗层的
    `_kernel_ok` 判据（"带掩码就不走内核"）会落回 eager 写法——CPU 档本来就没内核，
    真上 NPU 时要走内核路得让 attn_sw 侧接受带状键值，见本域 C2/云端后续。
    """''',
    "docstring_align")

P.write_text(s)
print("written", P, len(s.splitlines()), "lines")
