"""p2-10 B2 补丁 B：把命中下限从派单口径 ≥1 收紧到 spec 口径 ≥2（同 state 连发 3 问）。

跑法：cd release && .venv/bin/python .patch_p210_b2b.py
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "tests" / "test_serving.py"
src = TARGET.read_text(encoding="utf-8")

OLD = '''    st = svc.stats()
    assert st["prefix_hits"] - base["prefix_hits"] >= 1, \\
        f"服务路径没记到命中（三问同 state 至少该命中 1 次）：{st['prefix_hits']}"
'''
NEW = '''    st = svc.stats()
    # spec「命中可观测」的口径是"同 state 连发 3 问 → 命中计数 ≥2"（第一问冷启、后两问复用）；
    # 派单里的 ≥1 只是下限，按 spec 收紧才不会让"只命中一次"的退化实现混过关
    assert st["prefix_hits"] - base["prefix_hits"] >= 2, \\
        f"服务路径命中不足（三问同 state 应命中 ≥2）：{st['prefix_hits']}"
'''

hits = src.count(OLD)
if hits != 1:
    raise SystemExit(f"锚点命中 {hits} 次（要求恰好 1 次）")
TARGET.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
print(f"[patch-b2b] 命中下限收紧至 spec 口径 -> {TARGET}")
