from pathlib import Path
p = Path("production/backbone/loader.py"); s = p.read_text()
old = '''    compute_dtype: torch.dtype = torch.float32,
) -> LayerConversion:'''
new = '''    compute_dtype: torch.dtype = torch.float32,
    norm_plus_one: bool = True,
) -> LayerConversion:'''
assert old in s; s = s.replace(old, new, 1)
old2 = '''    for extra in ("q_norm.weight", "k_norm.weight"):
        if base + extra in raw:
            conv_tensors[extra] = L.permute_norm(raw[base + extra], lay)'''
new2 = '''    for extra in ("q_norm.weight", "k_norm.weight"):
        if base + extra in raw:
            moved = L.permute_norm(raw[base + extra], lay)
            conv_tensors[extra] = L.fold_unit_shift(moved) if norm_plus_one else moved
        if norm_plus_one:
            notes.append("q_norm/k_norm 折算 (1+w)：Qwen3_5RMSNorm 的读数是"零起步加一格"，"
                         "自研栈只认"读数直接乘"，故把垫底的 1 折进系数")'''
assert old2 in s; s = s.replace(old2, new2, 1)
s = s.replace('''同一分片只开一次（一次一层只要 8 块张量，几 MB 而已）。''',
 '''同一分片只开一次（一次一层只要 8 块张量，几 MB 而已）。逐头缩放另有一处折算：Qwen3_5RMSNorm
    的口径是 `norm(x) * (1 + w)`（w 零初始化），而自研栈的融合件只认 `norm(x) * s`，故在换算时
    把 `1 + w` 折进系数里（`layout.fold_unit_shift`），并对拍时按同一口径断言。''', 1)
p.write_text(s); print("patched")
