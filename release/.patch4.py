from pathlib import Path
p = Path("production/backbone/loader.py"); s = p.read_text()
old = '''    names = [base + n for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight")]'''
new = '''    names = [base + n for n in ATTN_NAMES if base + n in idx["weight_map"]]'''
assert old in s; s = s.replace(old, new, 1)
old2 = '''    if conv.gated:
        gate_src = L.split_gated_q(q_full, lay)[1]'''
new2 = '''    for extra in ("q_norm.weight", "k_norm.weight"):
        if extra in hf_tensors and extra in conv.tensors:
            want = L.fold_unit_shift(L.permute_norm(hf_tensors[extra], lay))
            if not torch.equal(conv.tensors[extra], want):
                report["mismatch"].append(f"{extra} 与『换座次 + 折算 (1+w)』的结果逐元素不等")
    if conv.gated:
        gate_src = L.split_gated_q(q_full, lay)[1]'''
assert old2 in s; s = s.replace(old2, new2, 1)
p.write_text(s); print("patched")
