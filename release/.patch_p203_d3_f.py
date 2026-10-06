"""p2-03 D3 补丁 F：split_isolation 两用例扩至中文对（train 语料 vs validation 考卷）。

跑法：cd release && .venv/bin/python .patch_p203_d3_f.py
验证：.venv/bin/python -m pytest tests/test_registry.py -k split_isolation -q
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "tests" / "test_registry.py"
src = TARGET.read_text(encoding="utf-8")
edits: list[tuple[str, str]] = []

# ── 1. 结构层：中文 train 两格必须存在、只挂 train 轴；CMMLU 查无 train 就不得凭空登记 ──
edits.append((
    '''    st = evalrun.collect_axes("all", manifest=_manifest())
    assert evalrun.TRAIN_AXIS not in st, "六轴表上不该出现 train 轴"
''',
    '''    st = evalrun.collect_axes("all", manifest=_manifest())
    assert evalrun.TRAIN_AXIS not in st, "六轴表上不该出现 train 轴"
    # D3 中文对：train 语料那两格必须在册且只挂 train 轴（与 typed train 同一道闸）
    for pid in registry.ZH_TRAIN_IDS_TUPLE:
        assert pid in registry.REGISTRY, f"D3 中文 train 登记项缺席：{pid}"
        assert registry.REGISTRY[pid].split == "train", f"{pid} 档位必须钉在 train"
        assert registry.REGISTRY[pid].axis == evalrun.TRAIN_AXIS, f"{pid} 必须挂 train 轴"
        assert pid in train_ids, f"{pid} 没进 train 轴消费表：{sorted(train_ids)}"
    # 探查结论必须在案（无 train 档的集只许留结论，不许留一个改名凑数的 pin）
    assert registry.ZH_TRAIN_PROBE["verdict"], "D3 中文 train 探查结论缺席"
    assert registry.ZH_TRAIN_PROBE["evidence"], "D3 探查取证行为空（结论不可复跑即无效）"
    assert "cmmlu" in registry.ZH_TRAIN_PROBE["absent"], "CMMLU 无 train 这条事实得挂在探查表上"
    faked = [pid for pid, pin in registry.REGISTRY.items()
             if pin.axis == evalrun.TRAIN_AXIS and "cmmlu" in pid]
    assert not faked, f"CMMLU 被凭空登记了 train 档（dev 改名凑 train？）：{faked}"
''',
))

# ── 2. 事实层：中文 train 记录要被真 scoop 到，且与中文考卷 id 零交集 ──────────────
edits.append((
    '''def test_split_isolation_train_ids_disjoint_from_quality():
    """事实层：任何 train 记录 id 不得出现在 quality 评测轴集内；qtype 三口一致、split 显式。"""
    _need_disk("typed-decisions", "typed-decisions-train")
    train = evalrun.load_train_records()
    assert train, "train 消费口 scoop 出空表（先跑 fetch --train）"
    assert all(r.get("split") == "train" for r in train), "数据类记录必须显式带 split 语义"
    assert {"choice", "noul", "score"} <= {r.get("qtype") for r in train}, "train 档 qtype 三口不全"
    quality = evalrun.load_axis_records("quality")
    overlap = {r["id"] for r in train} & {r["id"] for r in quality}
    assert not overlap, f"训测同集：{len(overlap)} 个 train id 出现在 quality 评测轴集内 {sorted(overlap)[:5]}"
''',
    '''def test_split_isolation_train_ids_disjoint_from_quality():
    """事实层：任何 train 记录 id 不得出现在 quality 评测轴集内；qtype 三口一致、split 显式。

    白话：这一条是 D1/D3 共用的那道隔离闸的"真数据版"——上面那条结构层只保证登记面上
    train 轴没串进六轴，本条要把盘上的信封真读一遍：训练口 scoop 出来的每一条都得写着
    split=train，且这些 id 与考卷（quality 轴，含中文 validation 那 400 题）一个都不能重。
    中文那半单独再断一次：train 语料得真在训练口里（C5 前置闸要的"有米可下锅"），而且
    "练的题"（id 里带 :train:）与"考的题"（id 里带 :validation:）不能是同一批——重了一个，
    训练分与中文评测分就互相漏题，报告上却看不出破口。
    """
    _need_disk("typed-decisions", "typed-decisions-train", *registry.ZH_TRAIN_IDS_TUPLE)
    train = evalrun.load_train_records()
    assert train, "train 消费口 scoop 出空表（先跑 fetch --train）"
    assert all(r.get("split") == "train" for r in train), "数据类记录必须显式带 split 语义"
    assert {"choice", "noul", "score"} <= {r.get("qtype") for r in train}, "train 档 qtype 三口不全"
    quality = evalrun.load_axis_records("quality")
    overlap = {r["id"] for r in train} & {r["id"] for r in quality}
    assert not overlap, f"训测同集：{len(overlap)} 个 train id 出现在 quality 评测轴集内 {sorted(overlap)[:5]}"
    # ── 中文对（D3）：训练口里得有中文档，且与中文考卷逐条互斥 ──
    zh_train = [r for r in train if r["id"].startswith("clue:")]
    assert zh_train, "load_train_records() 里没有中文 train 语料（D3 交付缺口，先跑 fetch --zh-train）"
    assert all(r.get("split") == "train" for r in zh_train), "中文 train 记录必须带 split=train"
    assert all(":train:" in r["id"] for r in zh_train), "中文 train 记录 id 的档位段必须是 train"
    zh_exam = {r["id"] for r in quality if r["id"].startswith("clue:")}
    assert zh_exam, "中文考卷缺席——零交集断言就退化成空断言（先跑 fetch --cn）"
    clash = {r["id"] for r in zh_train} & zh_exam
    assert not clash, f"中文训测同集：{len(clash)} 条 train 语料出现在考卷里 {sorted(clash)[:5]}"
''',
))

for old, new in edits:
    hits = src.count(old)
    if hits != 1:
        raise SystemExit(f"锚点命中 {hits} 次（要求恰好 1 次）：{old[:70]!r}")
    src = src.replace(old, new, 1)

TARGET.write_text(src, encoding="utf-8")
print(f"[patch-f] test_registry.py 落笔 {len(edits)} 处 -> {TARGET}")
