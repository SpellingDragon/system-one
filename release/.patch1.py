from pathlib import Path
p = Path("production/backbone/loader.py")
src = p.read_text()
old = '''    qn = conv.tensors["q_norm.weight"].view(lay.heads, lay.head_dim)
    kn = conv.tensors["k_norm.weight"].view(lay.heads, lay.head_dim)
    qkv[:, 0] = rmsnorm_rows(qkv[:, 0], qn, eps)
    qkv[:, 1] = rmsnorm_rows(qkv[:, 1], kn, eps)
'''
new = '''    qn = norm_weight_view(conv.tensors.get("q_norm.weight"), lay)
    kn = norm_weight_view(conv.tensors.get("k_norm.weight"), lay)
    if qn is not None:
        qkv[:, 0] = rmsnorm_rows(qkv[:, 0], qn, eps)
    if kn is not None:
        qkv[:, 1] = rmsnorm_rows(qkv[:, 1], kn, eps)
'''
assert old in src, "anchor missing"
src = src.replace(old, new)
helper = '''

def norm_weight_view(weight: torch.Tensor | None, lay: L.HeadLayout) -> torch.Tensor | None:
    """把逐坐标缩放系数整形成能广播的形状：全头共用则 (hd,)，逐头各配则 (heads, hd)。

    白话：音量旋钮有两种配法——要么一层共用一副、要么每个头自己一副。这里先看副数够不够
    分给每个头，够就一头一副地摆开，不够就原样一副横着放，让后面按行乘的时候自动对上。
    """
    if weight is None:
        return None
    n = int(weight.numel())
    if n == lay.head_dim:
        return weight
    if n == lay.heads * lay.head_dim:
        return weight.view(lay.heads, lay.head_dim)
    raise ValueError(f"逐坐标缩放长度 {n} 既不是 head_dim={lay.head_dim} 也不是 heads*head_dim={lay.heads * lay.head_dim}")
'''
anchor = '\n\ndef self_stack_attention('
assert anchor in src
src = src.replace(anchor, helper + anchor, 1)
p.write_text(src)
print("patched")
