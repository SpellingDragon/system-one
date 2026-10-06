"""p2-03 D3 补丁 D：把 ZH_TRAIN_IDS_TUPLE 挪到 FLAG_TO_IDS 之前（定义顺序，NameError 修）。

跑法：cd release && .venv/bin/python .patch_p203_d3_d.py
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "sys1" / "eval" / "registry.py"
src = TARGET.read_text(encoding="utf-8")

DEF_BLOCK = '''#: 中文 train 语料那一格（D3）：原件档 + 决策化档，都只挂 train 轴，不进六轴评测表
ZH_TRAIN_IDS_TUPLE = ("clue-train-subset", "clue-train-decision")
'''
HEAD = '''#: fetch 的分组开关（CLI 上的 --typed/--intern/... 在这里展开成 id，避免按钮与集名两处维护）
'''

assert src.count(DEF_BLOCK) == 1, src.count(DEF_BLOCK)
assert src.count(HEAD) == 1, src.count(HEAD)
src = src.replace(DEF_BLOCK, "", 1)                       # 从 TRAIN_IDS 那一块里摘出来
src = src.replace(HEAD, DEF_BLOCK + "\n" + HEAD, 1)       # 放回 FLAG_TO_IDS 之前（定义先于引用）
TARGET.write_text(src, encoding="utf-8")
print(f"[patch-d] 定义顺序修正 -> {TARGET}")
