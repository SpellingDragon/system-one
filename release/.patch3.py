from pathlib import Path
p = Path("production/backbone/loader.py"); s = p.read_text()
bad = '''    for extra in ("q_norm.weight", "k_norm.weight"):
        if base + extra in raw:
            moved = L.permute_norm(raw[base + extra], lay)
            conv_tensors[extra] = L.fold_unit_shift(moved) if norm_plus_one else moved
        if norm_plus_one:
            notes.append("q_norm/k_norm 折算 (1+w)：Qwen3_5RMSNorm 的读数是"零起步加一格"，"
                         "自研栈只认"读数直接乘"，故把垫底的 1 折进系数")'''
good = '''    for extra in ("q_norm.weight", "k_norm.weight"):
        if base + extra in raw:
            moved = L.permute_norm(raw[base + extra], lay)
            conv_tensors[extra] = L.fold_unit_shift(moved) if norm_plus_one else moved
    if norm_plus_one and any(k.endswith("_norm.weight") for k in conv_tensors):
        notes.append("q_norm/k_norm 系数已折算 (1+w)：Qwen3_5RMSNorm 的读数是『零起步加一格』，"
                     "自研栈融合件只认『读数直接乘』，故把垫底的 1 折进系数")'''
assert bad in s, "anchor missing"
p.write_text(s.replace(bad, good, 1)); print("patched")
