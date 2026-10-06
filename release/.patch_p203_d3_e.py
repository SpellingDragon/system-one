"""p2-03 D3 补丁 E：把登记行数更正为盘上实测值（tnews 53360 / ocnli 50437）。

跑法：cd release && .venv/bin/python .patch_p203_d3_e.py

背景：D3 初稿 note/注释写的是"53760/10047"，属凭记忆估写；用 pyarrow 读盘上 parquet
元数据复核得 tnews train = 53360 行、ocnli train = 50437 行。字节数（3399930/2440405）
是 HubApi 列举的实测值，保持不动。登记数字必须能被复跑命令戳穿，故一律改成实测值。
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "sys1" / "eval" / "registry.py"
src = TARGET.read_text(encoding="utf-8")
edits: list[tuple[str, str]] = []

# ── 1. ZH_TRAIN_SUBSET 上方的注释 ────────────────────────────────────────
edits.append((
    '''#: 中文 train 语料的自持上限（D3）：上游 CLUE tnews/ocnli 的 train 档实测 53760/10047 行，''',
    '''#: 中文 train 语料的自持上限（D3）：上游 CLUE train 档盘上实测 tnews 53360 行 / ocnli 50437 行''',
))

edits.append((
    '''#: 本域按固定 seed 只留这一段（副本体积与装配内存都按这个数封顶）；要全量请显式改 sampler''',
    '''#: （`pyarrow.parquet.ParquetFile(...).metadata.num_rows`）：本域按固定 seed 只留这一段''',
))

edits.append((
    '''ZH_TRAIN_SUBSET = MIN_SUBSET * 10''',
    '''#: （副本体积与装配内存都按这个数封顶）；要全量请显式改 sampler
ZH_TRAIN_SUBSET = MIN_SUBSET * 10''',
))

# ── 2. clue-train-subset 的 note：改列实测行数并给出可复跑口径 ────────────
edits.append((
    '''             "2440405 字节（`HubApi.get_dataset_files` 列举，盘上装出实测 53760/10047 行）。只取"''',
    '''             "2440405 字节（`HubApi.get_dataset_files` 列举）；盘上装出实测 tnews 53360 行、"
             "ocnli 50437 行（`pyarrow.parquet.ParquetFile(f).metadata.num_rows`）。只取"''',
))

for old, new in edits:
    hits = src.count(old)
    if hits != 1:
        raise SystemExit(f"锚点命中 {hits} 次（要求恰好 1 次）：{old[:70]!r}")
    src = src.replace(old, new, 1)

TARGET.write_text(src, encoding="utf-8")
print(f"[patch-e] registry.py 数字更正 {len(edits)} 处 -> {TARGET}")
