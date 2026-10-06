"""p2-03 D3 补丁 H：取证口径更正（"条目数"被标成"csv 件数"，登记数字跟着失准）。

跑法：cd release && .venv/bin/python .patch_p203_d3_h.py

复核（可逐字复跑）：
    .venv/bin/python -c "import zipfile; n=zipfile.ZipFile('bench/eval_data/raw/cmmlu-subset/cmmlu_v1_0_1.zip').namelist(); \
    print(sum(1 for x in n if x.startswith('dev/')), sum(1 for x in n if x.startswith('dev/') and x.endswith('.csv')))"
    → 68 8 → 实测 dev 条目 68（含目录条目 'dev/'）/ csv 67；test 条目 69 / csv 67；含 train 的路径 0

结论方向不变（CMMLU 上游确无 train 档），但登记数字必须能被一条命令戳穿，故把 csv 数与条目数
分开写清，并把探查脚本的计数式一起改对——否则下一个照这个脚本复跑的人会得到 67/67 而不是 68/69。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── 1. 探查脚本：csv 件数按后缀数，条目数另列 ────────────────────────────
P1 = ROOT / ".probe_p203_d3_train.py"
s = P1.read_text(encoding="utf-8")
OLD = '''    print(f"    csv 件数：dev={sum(1 for n in names if n.startswith('dev/'))} "
          f"test={sum(1 for n in names if n.startswith('test/'))} "
          f"train={sum(1 for n in names if 'train' in n.lower())}")
'''
NEW = '''    def _cnt(prep: str, *, csv_only: bool = True) -> int:
        keep = [n for n in names if n.startswith(f"{prep}/")]
        return sum(1 for n in keep if n.endswith(".csv")) if csv_only else len(keep)
    print(f"    csv 件数：dev={_cnt('dev')} test={_cnt('test')} "
          f"train={sum(1 for n in names if 'train' in n.lower())}"
          f"（条目数另含目录行：dev={_cnt('dev', csv_only=False)} "
          f"test={_cnt('test', csv_only=False)}）")
'''
assert s.count(OLD) == 1, f"probe 锚点命中 {s.count(OLD)} 次"
P1.write_text(s.replace(OLD, NEW, 1), encoding="utf-8")

# ── 2. registry：verdict 与取证第①行改成能逐字复跑的口径 ─────────────────
P2 = ROOT / "sys1" / "eval" / "registry.py"
src = P2.read_text(encoding="utf-8")
edits: list[tuple[str, str]] = []

edits.append((
    '''    "verdict": ("CMMLU 上游无 train 分割：modelscope/cmmlu 仓只有 README.md/cmmlu.py/"
                "cmmlu_v1_0_1.zip 三件，归档内部只有 dev/(68 个 csv) 与 test/(69 个 csv)，"
                "`train` 路径 0 个——故不登记 CMMLU train 项；中文 train 语料只交 "''',
    '''    "verdict": ("CMMLU 上游无 train 分割：modelscope/cmmlu 仓只有 README.md/cmmlu.py/"
                "cmmlu_v1_0_1.zip 三件，归档内部只有 dev/(67 个 csv) 与 test/(67 个 csv)，"
                "`train` 路径 0 个——故不登记 CMMLU train 项；中文 train 语料只交 "''',
))

edits.append((
    '''        "盘上取证（零流量）：`unzip -l bench/eval_data/raw/cmmlu-subset/cmmlu_v1_0_1.zip` 的 "
        "namelist 顶层目录只有 ['dev', 'test']，dev csv=68 / test csv=69 / 含 train 的路径=0",''',
    '''        "盘上取证（零流量）：`python -c \\"import zipfile; "
        "zipfile.ZipFile('bench/eval_data/raw/cmmlu-subset/cmmlu_v1_0_1.zip').namelist()\\"` 的顶层目录"
        "只有 ['dev', 'test']；按后缀数 csv：dev=67 / test=67 / 含 train 的路径=0（按条目数是 "
        "dev=68 / test=69，多出来的是目录行本身——两个口径都记着，免得复跑数字对不上被当成编造）",''',
))

for old, new in edits:
    hits = src.count(old)
    if hits != 1:
        raise SystemExit(f"registry 锚点命中 {hits} 次（要求恰好 1 次）：{old[:60]!r}")
    src = src.replace(old, new, 1)
P2.write_text(src, encoding="utf-8")
print(f"[patch-h] probe 脚本 1 处 + registry {len(edits)} 处口径更正 -> {P1} / {P2}")
