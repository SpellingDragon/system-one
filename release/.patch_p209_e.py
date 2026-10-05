"""一次性补丁脚本（p2-09 · A1 registry pin 收口）：给 registry 加"派生集"这条零流量路。

【做什么】按任务书"转写产物入 bench/eval_data/assembled/ 并注册进 registry"的要求，在
    `sys1/eval/registry.py` 上落六处改动：① `Pin` 增字段 `derived_from`（并摊进 `as_dict`）；
    ② 新增派生路下载器 `_fetch_derived`（零流量：不下载，按已 pin 的源副本现推，版本凭证沿用
    源集）并挂进 `FETCHERS`；③ 登记两个派生集 `cmmlu-decision`(choice/test) 与
    `clue-decision`(choice+noul/validation)；④ 挂上两个装配钩子 `zh-cmmlu` / `zh-clue`（函数内
    懒 import `sys1.eval.chinese` 避循环）；⑤ CLI 开关 `--zh` 与 `FLAG_TO_IDS["zh"]`，并把两集
    计入 `EXTRA_IDS`（于是 `--all` 一键连中文信封也能重建）；⑥ 源集 note 收口指向派生集。
【怎么做】一张 (原文 → 新文) 清单，逐条断言原文在全文**恰好出现一次**再替换；全部命中才写盘。
    不用行号删改，不顺手重排别处。写后立刻跑三连（wc / compileall / 锚点 grep）与
    `tests/test_registry.py` 全量回归（26 条基线必须仍全绿）。
【为什么】中文信封必须经 registry 唯一的记账口（`_store_assembled`）落盘，本域才不自记一份
    底账；而"派生"这件事若不进注册表，就只能靠 chinese.py 自己写文件——那正是【为什么】里
    被否的"两处各记必然漂移"。走 derived 而不是再下一遍数据，是因为题面原件已经 pin 过一次，
    重复下载既花流量又给出两个版本凭证。
"""
import sys
from pathlib import Path

TARGET = Path("sys1/eval/registry.py")

NEW_PINS = '''    # ── 中文决策信封（p2-09 派生集：零流量，按上面两份题面原件现推）─────────────
    "cmmlu-decision": Pin(
        id="cmmlu-decision", kind="derived", repo="sys1:eval.chinese", revision=ZH_DECISION_VERSION,
        split="test", full="cmmlu_v1_0_1", seed=NEEDLE_SEED, sampler="derived:cmmlu-subset",
        qtypes=("choice",), axis="quality", sources=(f"derived://{ZH_DECISION_VERSION}/cmmlu-subset",),
        patterns=(), assembler="zh-cmmlu", derived_from="cmmlu-subset",
        note="CMMLU 200 题决策化成 choice 信封（k=4，criteria=A/B/C/D 选项文本，问法换中文知识问法）。"
             "转写口径见 `sys1/eval/chinese.py`；题面原件与版本凭证都在 cmmlu-subset 那一格，本集"
             "不重复下载、不另钉版本。"),
    "clue-decision": Pin(
        id="clue-decision", kind="derived", repo="sys1:eval.chinese", revision=ZH_DECISION_VERSION,
        split="validation", full="opencompass-clue-parquet", seed=NEEDLE_SEED,
        sampler="derived:clue-subset", qtypes=("choice", "noul"), axis="quality",
        sources=(f"derived://{ZH_DECISION_VERSION}/clue-subset",), patterns=(), assembler="zh-clue",
        derived_from="clue-subset",
        note="CLUE 400 题决策化：tnews→choice(15 类)、ocnli→noul(蕴含=true，中立/矛盾=false)。"
             "候选文字沿用源 parquet 自带的类别码 \\"100\\"..\\"116\\"（盘上没给码↔类别名对照，"
             "不凭记忆补名——这条局限随中文 acc 一起披露）。档位随原件取 validation。"),
'''

FETCH_DERIVED = '''def _fetch_derived(pin: Pin, dst: Path) -> dict[str, Any]:
    """派生路不下载：只按已 pin 的源副本现推，把"从哪份原件、哪一版推的"写成一页凭据落盘。

    白话：这一格的题不是从网上再拿一遍，而是从盘上那份题面原件誊出来的（中文子集已经 fetch
    并钉过版本，再下一遍只得多花流量、还多出第二个版本凭证谁也不认）。所以这里零字节是事实：
    要证明的只有两件事——原件在不在（不在就报命令让人先去 fetch --cn）、原件是哪一版（散列抄下来
    存进 derived_from.json）。装配规则变了不要指望这条路自动重跑：幂等照旧，改过口径请 --force。
    """
    src = REGISTRY.get(pin.derived_from)
    if src is None:
        raise FetchError(f"派生集 {pin.id} 的源副本 {pin.derived_from!r} 未登记，无从现推")
    origin = DATA_DIR / ASSEMBLED_DIR_NAME / src.id / f"{src.id}.jsonl"
    if not origin.is_file():
        raise FetchError(f"派生集 {pin.id} 的题面原件不在盘上：{origin}（先跑 "
                         f"`python -m sys1.eval.registry fetch --sets {src.id}`）")
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "derived_from.json"
    out.write_text(json.dumps({"from": src.id, "revision": src.revision, "revision_full": src.full,
                               "split": src.split, "seed": src.seed, "assembler": pin.assembler,
                               "origin_bytes": origin.stat().st_size, "origin_sha256": _sha256(origin)},
                              ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    return {"resolved": f"derived:{src.id}@{src.revision}", "bytes": 0, "cached": True, "files": 1,
            "endpoint": "local", "source": str(origin), "sha256": {out.name: _sha256(out)}}


'''

ASSEMBLE_WRAPPERS = '''def _assemble_zh_cmmlu(raw: Path) -> list[dict[str, Any]]:
    """中文 CMMLU → 决策信封：规则与版号住在 p2-09 的 `sys1/eval/chinese.py`，这里只挂个名。"""
    from sys1.eval import chinese                           # 懒 import：chinese 反向 import 本模块

    return chinese.assemble_cmmlu_decision(raw)


def _assemble_zh_clue(raw: Path) -> list[dict[str, Any]]:
    """中文 CLUE → 决策信封（tnews 多选题 + ocnli 是非题）：同上，口径只在一处。"""
    from sys1.eval import chinese                           # 懒 import：避免模块级循环引用

    return chinese.assemble_clue_decision(raw)


'''

PAIRS = [
    # ① 模块 docstring：来源路数由五条变六条（含派生路）
    ("    ② 拉取按来源分派五条路：git 路走",
     "    ② 拉取按来源分派六条路（第六条是 p2-09 的派生路：不下载，按已 pin 的副本现推）：git 路走"),

    # ② 下载区段标题
    ("# ---------------------------------------------------------------- 下载（五条路 + 兜底候选源）",
     "# ---------------------------------------------------------------- 下载（六条路 + 兜底候选源）"),

    # ③ 中文轨的转写版号（registry 侧的单一字面量；与 chinese.ZH_DECISION_VERSION 由用例对拍）
    ('MIN_SUBSET = 200                      # 子集自持副本的题量下限（design §子集策略）',
     'MIN_SUBSET = 200                      # 子集自持副本的题量下限（design §子集策略）\n'
     '#: 中文决策信封的转写口径版号（规则住在 sys1/eval/chinese.py；两侧不一致即有用例判红）\n'
     'ZH_DECISION_VERSION = "zh_decision_v1"'),

    # ④ Pin 增字段
    ('    assembler: str = ""\n    note: str = ""',
     '    assembler: str = ""\n    note: str = ""\n'
     '    derived_from: str = ""      # kind="derived" 时填：题面原件的集 id（零流量，按它现推）'),

    # ⑤ as_dict 摊开新字段
    ('            "assembler": self.assembler, "note": self.note,',
     '            "assembler": self.assembler, "note": self.note,\n'
     '            "derived_from": self.derived_from,'),

    # ⑥ 源集 note 收口（cmmlu）
    ('        note="中文多选题；决策化成信封由 p2-09 承接，本域只登记 + 产自持子集副本。实测该仓"',
     '        note="中文多选题的题面原件；决策化信封已交派生集 cmmlu-decision（转写口径 '
     'zh_decision_v1），本集只留原件与版本凭证。实测该仓"'),

    # ⑦ 源集 note 收口（clue）+ 紧跟其后插入两个派生 pin
    ('             "（实测文件清单只有 clue.py/dataset_infos.json/README），故取 opencompass/clue 的 parquet 镜像。"),\n',
     '             "（实测文件清单只有 clue.py/dataset_infos.json/README），故取 opencompass/clue 的 parquet 镜像。"\n'
     '             "决策化信封见派生集 clue-decision（zh_decision_v1）。"),\n' + NEW_PINS),

    # ⑧ CLI 分组按钮
    ('    "cn": ("cmmlu-subset", "clue-subset"),',
     '    "cn": ("cmmlu-subset", "clue-subset"),\n'
     '    "zh": ("cmmlu-decision", "clue-decision"),'),

    # ⑨ 扩展集清单（--all 覆盖到中文决策信封）
    ('EXTRA_IDS = ("cmmlu-subset", "clue-subset", "mmbench-cn-subset", "longbench-zh", "needle-synthetic")',
     'EXTRA_IDS = ("cmmlu-subset", "clue-subset", "cmmlu-decision", "clue-decision",\n'
     '             "mmbench-cn-subset", "longbench-zh", "needle-synthetic")'),

    # ⑩ 派生下载器 + FETCHERS 注册
    ('FETCHERS = {"git": _fetch_git, "archive": _fetch_archive, "hf": _fetch_hf,\n'
     '            "modelscope": _fetch_modelscope, "synthetic": _fetch_synthetic}',
     FETCH_DERIVED + 'FETCHERS = {"git": _fetch_git, "archive": _fetch_archive, "hf": _fetch_hf,\n'
     '            "modelscope": _fetch_modelscope, "synthetic": _fetch_synthetic,\n'
     '            "derived": _fetch_derived}'),

    # ⑪ 装配钩子 + ASSEMBLERS 注册
    ('ASSEMBLERS = {"typed": _assemble_typed, "intern": _assemble_intern, "jev": _assemble_jev,\n'
     '              "typed-train": _assemble_typed_train,\n'
     '              "cmmlu": _assemble_cmmlu, "clue": _assemble_clue, "mmbench": _assemble_mmbench,\n'
     '              "longbench": _assemble_longbench, "needle": _assemble_needle}',
     ASSEMBLE_WRAPPERS + 'ASSEMBLERS = {"typed": _assemble_typed, "intern": _assemble_intern, "jev": _assemble_jev,\n'
     '              "typed-train": _assemble_typed_train,\n'
     '              "cmmlu": _assemble_cmmlu, "clue": _assemble_clue, "mmbench": _assemble_mmbench,\n'
     '              "longbench": _assemble_longbench, "needle": _assemble_needle,\n'
     '              "zh-cmmlu": _assemble_zh_cmmlu, "zh-clue": _assemble_zh_clue}'),

    # ⑫ CLI 上真的加一个 --zh 按钮（FLAG_TO_IDS 有键而 argparse 没按钮，按下去也不会生效）
    ('                      ("jev", "只拉 jevbench"), ("cn", "拉中文扩展子集（CMMLU+CLUE）"),',
     '                      ("jev", "只拉 jevbench"), ("cn", "拉中文扩展子集（CMMLU+CLUE 题面原件）"),\n'
     '                      ("zh", "装配中文决策信封（零流量，派生自 --cn 那份题面原件）"),'),
]


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    for i, (old, _new) in enumerate(PAIRS, start=1):
        n = text.count(old)
        assert n == 1, f"锚点 #{i} 命中 {n} 次（期望恰好 1 次），拒绝盲改：{old.splitlines()[0][:60]!r}"
    for old, new in PAIRS:
        text = text.replace(old, new, 1)
    for probe in ("derived_from", "def _fetch_derived", '"derived": _fetch_derived',
                  '"zh-cmmlu": _assemble_zh_cmmlu', '"zh": ("cmmlu-decision", "clue-decision")',
                  '"cmmlu-decision": Pin(', '"clue-decision": Pin(', 'ZH_DECISION_VERSION = "zh_decision_v1"'):
        assert probe in text, f"改完却找不到关键件：{probe!r}"
    TARGET.write_text(text, encoding="utf-8")
    print(f"[patch-e] registry 派生路 + 两 pin + 两钩子 + --zh 落定，行数 {len(text.splitlines())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
