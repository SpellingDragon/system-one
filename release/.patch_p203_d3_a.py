"""p2-03 D3 补丁 A：registry 侧登记中文 train 语料（CLUE train 档）+ CMMLU 无 train 取证结论。

跑法：cd release && .venv/bin/python .patch_p203_d3_a.py
每处替换都要求锚点**恰好命中一次**，命不中即报错退出（不静默跳过）。
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "sys1" / "eval" / "registry.py"
src = TARGET.read_text(encoding="utf-8")
edits: list[tuple[str, str]] = []

# ── 1. 常量：中文 train 语料的自持上限 ─────────────────────────────────────
edits.append((
    '''#: 训练数据专用登记轴（D1）：不并进六轴评测表——训练复用评测题面=训测同集假分（p2-05 教训）
TRAIN_AXIS = "train"
''',
    '''#: 训练数据专用登记轴（D1）：不并进六轴评测表——训练复用评测题面=训测同集假分（p2-05 教训）
TRAIN_AXIS = "train"
#: 中文 train 语料的自持上限（D3）：上游 CLUE tnews/ocnli 的 train 档实测 53760/10047 行，
#: 本域按固定 seed 只留这一段（副本体积与装配内存都按这个数封顶）；要全量请显式改 sampler
ZH_TRAIN_SUBSET = MIN_SUBSET * 10
''',
))

# ── 2. CMMLU 原件那一格补 train 探查结论指针 ──────────────────────────────
edits.append((
    '''             "就一个 1.08MB 的 zip（含全学科 csv）。"),
''',
    '''             "就一个 1.08MB 的 zip（含全学科 csv）。train 分割探查（2026-10-06，D3）：仓内三件与"
             "归档内部都只有 dev/test 两档，查无 train——不登记 CMMLU train 项、不拿 dev 改名凑数，"
             "取证见 `ZH_TRAIN_PROBE`（run 底账原样引用）。"),
''',
))

# ── 3. 中文 train 语料两个登记项（挂在 train 轴上）────────────────────────
edits.append((
    '''             "不凭记忆补名——这条局限随中文 acc 一起披露）。档位随原件取 validation。"),
    "mmbench-cn-subset": Pin(
''',
    '''             "档位随原件取 validation。"),
    # ── 中文 train 语料（D3 · p2-09 隐患裁决 C5 前置闸之二：训练档与考卷必须两格各占一行）──
    "clue-train-subset": Pin(
        id="clue-train-subset", kind="modelscope", repo="opencompass/clue", revision="master",
        split="train", full="opencompass-clue-parquet", seed=NEEDLE_SEED,
        sampler=f"seeded-subset:{ZH_TRAIN_SUBSET}", qtypes=("choice", "noul"), axis=TRAIN_AXIS,
        sources=("modelscope://opencompass/clue",),
        patterns=("tnews/train-*.parquet", "ocnli/train-*.parquet"), assembler="clue-train",
        note="CLUE 题面原件的 **train 档**（D3 补登）：上游确有三个档位——实测仓内 "
             "tnews/train-00000-of-00001.parquet 3399930 字节、ocnli/train-00000-of-00001.parquet "
             "2440405 字节（`HubApi.get_dataset_files` 列举，盘上装出实测 53760/10047 行）。只取"
             f"固定 seed 的 {ZH_TRAIN_SUBSET} 条自持（原件只当语料凭据，绝不当考卷），决策化交"
             "派生集 clue-train-decision。与 validation 那两份（clue-subset/clue-decision）分属两"
             "轴，id 里的档位段（train/validation）天然互斥，隔离由 split_isolation 用例把关。"),
    "clue-train-decision": Pin(
        id="clue-train-decision", kind="derived", repo="sys1:eval.chinese",
        revision=ZH_DECISION_VERSION, split="train", full="opencompass-clue-parquet",
        seed=NEEDLE_SEED, sampler="derived:clue-train-subset",
        qtypes=("choice", "noul"), axis=TRAIN_AXIS,
        sources=(f"derived://{ZH_DECISION_VERSION}/clue-train-subset",), patterns=(),
        assembler="zh-clue-train", derived_from="clue-train-subset",
        note="CLUE train 档决策化出来的训练信封（零流量，按 clue-train-subset 那份原件现推）：折法"
             "与考卷那一格完全同规则（tnews→choice k=15、ocnli→noul），只有档位不同——同一套规则"
             "分别喂 train 与 validation，才谈得上「练的题与考的题不是同一批」。挂 train 轴、经 "
             "load_train_records() 消费，永不进六轴评测表（p2-05 训测同集假分的教训在这里同样生效）。"),
    "mmbench-cn-subset": Pin(
''',
))

# ── 4. 中文 train 探查结论（同 D1 的 Intern 先例：查无即如实记）────────────
edits.append((
    '''    "probed_at": "2026-10-05",
}

#: fetch 的分组开关''',
    '''    "probed_at": "2026-10-05",
}

#: 中文集 train 分割探查结论（D3 · 2026-10-06）：CMMLU 查无 train 档，CLUE 有 train 档。
#: 三条取证原样抄自 `.probe_p203_d3_train.py` 的运行输出（一条命令一条事实，可复跑），
#: run 底账引用 verdict；不登记 CMMLU train 项，也不拿 dev 档改名冒充 train——dev 是少样本
#: 示例档，改名叫 train 就是给考卷同源的数据刷上"练习册"的标签，训测隔离当场失效。
ZH_TRAIN_PROBE = {
    "verdict": ("CMMLU 上游无 train 分割：modelscope/cmmlu 仓只有 README.md/cmmlu.py/"
                "cmmlu_v1_0_1.zip 三件，归档内部只有 dev/(68 个 csv) 与 test/(69 个 csv)，"
                "`train` 路径 0 个——故不登记 CMMLU train 项；中文 train 语料只交 "
                "CLUE(tnews/ocnli) train 档（已登记 clue-train-subset / clue-train-decision）"),
    "absent": ("cmmlu",),
    "present": ("clue-tnews", "clue-ocnli"),
    "evidence": [
        "盘上取证（零流量）：`unzip -l bench/eval_data/raw/cmmlu-subset/cmmlu_v1_0_1.zip` 的 "
        "namelist 顶层目录只有 ['dev', 'test']，dev csv=68 / test csv=69 / 含 train 的路径=0",
        "上游列举（魔搭 API）：`HubApi().get_dataset_files(repo_id='modelscope/cmmlu', "
        "revision='master', recursive=True)` 回 3 件：README.md 406 B、cmmlu.py 5066 B、"
        "cmmlu_v1_0_1.zip 1078656 B —— 仓内没有任何 train 数据件，也没有第二个归档",
        "上游列举（魔搭 API）：`get_dataset_files(repo_id='opencompass/clue', revision='master', "
        "recursive=True)` 回 49 件，其中 11 个任务带 train 档，本域要的两件是 tnews/"
        "train-00000-of-00001.parquet（3399930 B）与 ocnli/train-00000-of-00001.parquet"
        "（2440405 B）——CLUE 确有 train 档，据此登记并真拉装配",
    ],
    "probed_at": "2026-10-06",
}

#: fetch 的分组开关''',
))

# ── 5. 分组开关 / id 清单 ────────────────────────────────────────────────
edits.append((
    '''    "zh": ("cmmlu-decision", "clue-decision"),
''',
    '''    "zh": ("cmmlu-decision", "clue-decision"),
    "zh-train": ZH_TRAIN_IDS_TUPLE,          # D3 中文 train 语料（原件 + 决策信封两格）
''',
))

edits.append((
    '''THREE_SHEET_IDS = ("typed-decisions", "intern-decision", "jevbench")
TRAIN_IDS = ("typed-decisions-train",)        # 训练侧登记集（D1）：fetch --all 也带上，底账才完整
''',
    '''THREE_SHEET_IDS = ("typed-decisions", "intern-decision", "jevbench")
#: 中文 train 语料那一格（D3）：原件档 + 决策化档，都只挂 train 轴，不进六轴评测表
ZH_TRAIN_IDS_TUPLE = ("clue-train-subset", "clue-train-decision")
#: 训练侧登记集（D1 交 typed train，D3 补中文档）：fetch --all 一并带上，底账才完整
TRAIN_IDS = ("typed-decisions-train",) + ZH_TRAIN_IDS_TUPLE
''',
))

# ── 6. CLUE 装配口参数化档位上限 + train 档装配器 ────────────────────────
edits.append((
    '''def _assemble_clue(raw: Path) -> list[dict[str, Any]]:
    """CLUE 镜像 parquet（tnews/ocnli）→ 固定 seed 子集，并逐条标好决策化的目标题型。
''',
    '''def _assemble_clue(raw: Path, *, n: int = MIN_SUBSET * 2) -> list[dict[str, Any]]:
    """CLUE 镜像 parquet（tnews/ocnli）→ 固定 seed 子集，并逐条标好决策化的目标题型。
''',
))

edits.append((
    '''    副本先钉住四百条起，免得日后抽得更小却还挂着同一句覆盖充分。
    """
''',
    '''    副本先钉住四百条起，免得日后抽得更小却还挂着同一句覆盖充分。`n` 是这一档的自持上限：
    考卷那一格取 400（validation），train 那一格取 `ZH_TRAIN_SUBSET`——同一套抄法只换个上限，
    才不至于出现"两份副本由两套规则抄出来"这种事。
    """
''',
))

edits.append((
    '''    if not out:
        raise AssembleError(f"CLUE 副本里没有带人工答案的 parquet 行（无真值的档一律不收，"
                            f"跳过 {skipped_unlabeled} 行）：{raw}")
    return _seeded_subset(out, MIN_SUBSET * 2, NEEDLE_SEED)
''',
    '''    if not out:
        raise AssembleError(f"CLUE 副本里没有带人工答案的 parquet 行（无真值的档一律不收，"
                            f"跳过 {skipped_unlabeled} 行）：{raw}")
    return _seeded_subset(out, n, NEEDLE_SEED)


def _assemble_clue_train(raw: Path) -> list[dict[str, Any]]:
    """CLUE train 档（tnews/ocnli 的 train parquet）→ 题面原件自持副本（D3 补登）。

    白话：还是 `_assemble_clue` 那门抄题的手艺，只是从 train 那一份文件里取件、上限换成
    `ZH_TRAIN_SUBSET`；每条的 id 里带着档位段（train），与 validation 那份天然撞不上号。
    这一格交的是"题面长什么样"的原件凭据，能直接喂训练侧的决策信封在派生集那一格。
    """
    return _assemble_clue(raw, n=ZH_TRAIN_SUBSET)
''',
))

# ── 7. 中文 train 决策装配钩子 + ASSEMBLERS 表 ───────────────────────────
edits.append((
    '''ASSEMBLERS = {"typed": _assemble_typed, "intern": _assemble_intern, "jev": _assemble_jev,
              "typed-train": _assemble_typed_train,
              "cmmlu": _assemble_cmmlu, "clue": _assemble_clue, "mmbench": _assemble_mmbench,
              "longbench": _assemble_longbench, "needle": _assemble_needle,
              "zh-cmmlu": _assemble_zh_cmmlu, "zh-clue": _assemble_zh_clue}
''',
    '''def _assemble_zh_clue_train(raw: Path) -> list[dict[str, Any]]:
    """中文 CLUE **train 档** → 训练信封：折法住在 p2-09 的 `chinese.py`，这里只挂个名（D3）。"""
    from sys1.eval import chinese                           # 懒 import：与同族钩子一致

    return chinese.assemble_clue_train_decision(raw)


ASSEMBLERS = {"typed": _assemble_typed, "intern": _assemble_intern, "jev": _assemble_jev,
              "typed-train": _assemble_typed_train,
              "cmmlu": _assemble_cmmlu, "clue": _assemble_clue, "mmbench": _assemble_mmbench,
              "longbench": _assemble_longbench, "needle": _assemble_needle,
              "zh-cmmlu": _assemble_zh_cmmlu, "zh-clue": _assemble_zh_clue,
              "clue-train": _assemble_clue_train, "zh-clue-train": _assemble_zh_clue_train}
''',
))

# ── 8. CLI：加 --zh-train 按钮，并把带连字符的分组名映射到 argparse dest ──
edits.append((
    '''                      ("zh", "装配中文决策信封（零流量，派生自 --cn 那份题面原件）"),
''',
    '''                      ("zh", "装配中文决策信封（零流量，派生自 --cn 那份题面原件）"),
                      ("zh-train", "拉中文 train 语料（CLUE tnews/ocnli train 档 + 现推决策信封；"
                                   "只挂 train 轴，不入六轴评测表）"),
''',
))

edits.append((
    '''    groups = tuple(flag for flag in FLAG_TO_IDS if getattr(args, flag, False))
''',
    '''    # argparse 把 --zh-train 的 dest 写成 zh_train（连字符换下划线），而 FLAG_TO_IDS 认的是
    # 按钮本名；不按这个规矩换算，带连字符的分组按下去就是块死键（只在表里加键，没人理它）
    groups = tuple(flag for flag in FLAG_TO_IDS
                   if getattr(args, flag.replace("-", "_"), False))
''',
))

for old, new in edits:
    hits = src.count(old)
    if hits != 1:
        raise SystemExit(f"锚点命中 {hits} 次（要求恰好 1 次）：{old[:70]!r}")
    src = src.replace(old, new, 1)

TARGET.write_text(src, encoding="utf-8")
print(f"[patch-a] registry.py 落笔 {len(edits)} 处 -> {TARGET}")
