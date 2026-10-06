"""p2-03 D3 补丁 G：报告行把中文 train 那两集逐集列出，免得"4,000 条"被读成可训练条数。

跑法：cd release && .venv/bin/python .patch_p203_d3_g.py

背景：train 轴上中文占了两个集——原件档 clue-train-subset（2,000 条题面，envelope_ready=
False，只当语料凭据）与派生档 clue-train-decision（2,000 条决策信封，训练口真正吃的）。
底账按集累计得 4,000，读者若拿去当训练分母就翻了一倍。逐集明细一列，谁进训练口一目了然
（`load_train_records()` 实测 8,000 = typed 6,000 + 中文决策 2,000，重复 id 0 个）。
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "sys1" / "eval" / "run.py"
src = TARGET.read_text(encoding="utf-8")

OLD = '''        f"（通道 {'、'.join(str(r['channel']) for r in tl['zh_sets'] if r['status'] in ('fetched', 'cached')) or '未拉'}）；"
        f"CMMLU train 探查：{tl['zh_probe']}",
'''
NEW = '''        f"（通道 {'、'.join(str(r['channel']) for r in tl['zh_sets'] if r['status'] in ('fetched', 'cached')) or '未拉'}）"
        f"，逐集 {[(r['id'], r['status'], r['samples']) for r in tl['zh_sets']]}——"
        f"条数按集累计，训练口只吃决策信封（原件档 envelope_ready=False，不入分母）；"
        f"CMMLU train 探查：{tl['zh_probe']}",
'''

hits = src.count(OLD)
if hits != 1:
    raise SystemExit(f"锚点命中 {hits} 次（要求恰好 1 次）")
TARGET.write_text(src.replace(OLD, NEW, 1), encoding="utf-8")
print(f"[patch-g] run.py 报告行补逐集明细 -> {TARGET}")
