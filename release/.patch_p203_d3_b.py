"""p2-03 D3 补丁 B：chinese.py 补中文 train 档的决策化去向与装配钩子。

跑法：cd release && .venv/bin/python .patch_p203_d3_b.py
只加不改口径：转写规则与考卷那一格完全同一条，分开的只是"档位登记在哪一格"。
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "sys1" / "eval" / "chinese.py"
src = TARGET.read_text(encoding="utf-8")
edits: list[tuple[str, str]] = []

# ── 1. 模块 docstring【怎么做】第③条补一句 train 档 ───────────────────────
edits.append((
    '''       那套唯一的记账代码写，本模块不自记一份，免得两处底账互相改写。转写口径变了就升
       `ZH_DECISION_VERSION`，并把版本号写进派生 pin 的 note——报告里的中文分据此能追到规则版本。
''',
    '''       registry 那套唯一的记账代码写，本模块不自记一份，免得两处底账互相改写。转写口径变了就升
       `ZH_DECISION_VERSION`，并把版本号写进派生 pin 的 note——报告里的中文分据此能追到规则版本。
       D3 再补一路**训练档**：`clue-train-subset → clue-train-decision`（registry 的 train 轴，
       经 `load_train_records()` 消费）。折法一条不改，只把档位另登一格——同一条规则分别喂
       train 与 validation，训测隔离才成立；CMMLU 经探查上游确无 train 分割（取证见
       `registry.ZH_TRAIN_PROBE`），本模块不拿 dev 档改名凑数。
''',
))

# ── 2. 常量：train 档的去向单独一格，与考卷那对分开登记 ───────────────────
edits.append((
    '''#: 源副本集 id → 决策信封集 id（registry 里成对登记；派生集零流量，按左边那份现推）。
SOURCE_TO_DECISION = {"cmmlu-subset": "cmmlu-decision", "clue-subset": "clue-decision"}
DECISION_TO_SOURCE = {v: k for k, v in SOURCE_TO_DECISION.items()}
''',
    '''#: 源副本集 id → 决策信封集 id（registry 里成对登记；派生集零流量，按左边那份现推）。
SOURCE_TO_DECISION = {"cmmlu-subset": "cmmlu-decision", "clue-subset": "clue-decision"}
DECISION_TO_SOURCE = {v: k for k, v in SOURCE_TO_DECISION.items()}
#: 中文 **train 语料**那一格（D3）：与上面那对考卷分开设表——`--zh` 按钮、quality 轴的账都
#: 按考卷那对走，混进同一张表会把评测面一起改了；两张表合并成 `ALL_SOURCE_TO_DECISION` 供
#: 名字换算与转写分派用（换算是一张表，记账是两张表，各管各的）。
TRAIN_SOURCE_TO_DECISION = {"clue-train-subset": "clue-train-decision"}
ALL_SOURCE_TO_DECISION = {**SOURCE_TO_DECISION, **TRAIN_SOURCE_TO_DECISION}
ALL_DECISION_TO_SOURCE = {v: k for k, v in ALL_SOURCE_TO_DECISION.items()}
#: CLUE 系的两个档位（考卷 validation / 语料 train）：转写折法完全同一条，只是原件来自哪一档。
CLUE_SOURCE_PIDS = ("clue-subset", "clue-train-subset")
#: 各源副本缺件时的补救命令（把"先敲哪条"写进报错里，别让下游拿空表算出 0 分当实测）。
FETCH_HINT = {"cmmlu-subset": "fetch --cn", "clue-subset": "fetch --cn",
              "clue-train-subset": "fetch --zh-train"}
''',
))

# ── 3. 名字换算改走合并表（考卷与语料都能换算，报错仍列全表）──────────────
edits.append((
    '''    if source_pid not in SOURCE_TO_DECISION:
        raise ChineseTranscribeError(
            f"没有为 {source_pid!r} 登记决策化去向；已登记的源副本：{sorted(SOURCE_TO_DECISION)}")
    return SOURCE_TO_DECISION[source_pid]
''',
    '''    if source_pid not in ALL_SOURCE_TO_DECISION:
        raise ChineseTranscribeError(
            f"没有为 {source_pid!r} 登记决策化去向；已登记的源副本：{sorted(ALL_SOURCE_TO_DECISION)}")
    return ALL_SOURCE_TO_DECISION[source_pid]
''',
))

edits.append((
    '''    if decision_pid not in DECISION_TO_SOURCE:
        raise ChineseTranscribeError(
            f"{decision_pid!r} 不是本模块登记的中文决策集；可选：{sorted(DECISION_TO_SOURCE)}")
    return DECISION_TO_SOURCE[decision_pid]
''',
    '''    if decision_pid not in ALL_DECISION_TO_SOURCE:
        raise ChineseTranscribeError(
            f"{decision_pid!r} 不是本模块登记的中文决策集；可选：{sorted(ALL_DECISION_TO_SOURCE)}")
    return ALL_DECISION_TO_SOURCE[decision_pid]
''',
))

# ── 4. load_subset：补救命令按档位给（train 档的按钮是 --zh-train）─────────
edits.append((
    '''    if not path.is_file():
        raise ChineseTranscribeError(
            f"中文子集副本不在盘上：{path}（先跑 `python -m sys1.eval.registry fetch --cn`）")
''',
    '''    if not path.is_file():
        raise ChineseTranscribeError(
            f"中文子集副本不在盘上：{path}（先跑 `python -m sys1.eval.registry "
            f"{FETCH_HINT.get(source_pid, 'fetch --cn')}`）")
''',
))

# ── 5. 转写分派：CLUE 两档同一条折法 ─────────────────────────────────────
edits.append((
    '''    if source_pid == "cmmlu-subset":
        return [cmmlu_to_envelope(r) for r in rows]
    if source_pid == "clue-subset":
        return [clue_to_envelope(r) for r in rows]
    raise ChineseTranscribeError(f"未登记的中文源副本集 {source_pid!r}；可选：{sorted(SOURCE_TO_DECISION)}")
''',
    '''    if source_pid == "cmmlu-subset":
        return [cmmlu_to_envelope(r) for r in rows]
    if source_pid in CLUE_SOURCE_PIDS:          # validation 考卷与 train 语料共用这一条折法
        return [clue_to_envelope(r) for r in rows]
    raise ChineseTranscribeError(
        f"未登记的中文源副本集 {source_pid!r}；可选：{sorted(ALL_SOURCE_TO_DECISION)}")
''',
))

# ── 6. registry 装配钩子：train 档那一格 ─────────────────────────────────
edits.append((
    '''def load_decision_records(decision_pid: str, *, limit: int = 0) -> list[dict[str, Any]]:
''',
    '''def assemble_clue_train_decision(raw: Path) -> list[dict[str, Any]]:
    """registry 装配口（clue-train-decision，D3）：读 train 档原件，交同一条折法出训练信封。

    白话：和 `assemble_clue_decision` 是同一门手艺，只是题目从 train 那份副本里读。这样
    "练的题"与"考的题"由同一条规则各写一遍，差别只在档位——若两份信封由两套规则写出来，
    训测之间的可比性当场没了，而分数上看不出这种破口。装配失败仍报成 registry 的装配错，
    让 fetch 把这个集记进 failed 栏，绝不静默出一半。
    """
    del raw
    try:
        rows = load_subset("clue-train-subset")
    except ChineseTranscribeError as exc:
        raise registry.AssembleError(str(exc)) from exc
    return transcribe_subset("clue-train-subset", rows)


def load_decision_records(decision_pid: str, *, limit: int = 0) -> list[dict[str, Any]]:
''',
))

# ── 7. __all__ 补三枚新名字 ──────────────────────────────────────────────
edits.append((
    '''    "assemble_clue_decision",
    "assemble_cmmlu_decision",
''',
    '''    "ALL_DECISION_TO_SOURCE",
    "ALL_SOURCE_TO_DECISION",
    "CLUE_SOURCE_PIDS",
    "TRAIN_SOURCE_TO_DECISION",
    "assemble_clue_decision",
    "assemble_clue_train_decision",
    "assemble_cmmlu_decision",
''',
))

for old, new in edits:
    hits = src.count(old)
    if hits != 1:
        raise SystemExit(f"锚点命中 {hits} 次（要求恰好 1 次）：{old[:70]!r}")
    src = src.replace(old, new, 1)

TARGET.write_text(src, encoding="utf-8")
print(f"[patch-b] chinese.py 落笔 {len(edits)} 处 -> {TARGET}")
