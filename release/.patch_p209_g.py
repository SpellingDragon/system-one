"""一次性补丁脚本（p2-09 · A1 测试件收口）：清掉 tests/test_chinese.py 里 4 处劣质写法。

【做什么】写入后自查发现四处会让用例失真甚至必红的写法，逐条换成真断言：
    ① `test_transcribe_missing_origin_raises_actionable_command` 用"条件表达式抛错"的怪招，
       原件在盘上时反而走进 throw 分支——必然红；改成把 `registry.DATA_DIR` 挪进 tmp_path
       制造"还没 fetch"，再断三条报错路都指到可执行命令。
    ② `test_transcribe_is_deterministic_rerun_builds_same_rows` 留着 `zip(..., [])` 空转循环
       和 `_sha256(f) == _sha256(f)` 自反废断言；删掉，改成整批契约字段散列比对（有信息量）。
    ③ `test_mix_hygiene_refuses_eval_sets_as_train_corpus` 里那行 `... if False else True`
       永远为真；改成"脏值确实进了配比栏 / 其余键没被顺手改写"，并按两条不同的报错文案分别匹配。
    ④ `test_transcribe_registry_pins_registered_and_version_locked` 借 `and` 短路凑出来的
       集合推导式看不懂在比什么；改成按源集分支直接比期望题型。
【怎么做】一张 (原文 → 新文) 清单，逐条断言原文在全文恰好出现一次再替换；不重排别的用例、
    不改无关空行。写后跑三连（wc / compileall / 锚点 grep）+ `pytest tests/test_chinese.py -q`。
【为什么】测试件的信誉在于"它会为错误的实现变红"。空转循环、自反断言、`if False else True`
    这类行数上去只显得用例很厚，实际什么都不会拦——正是评审复跑时最容易被打回的假绿。
"""
import sys
from pathlib import Path

TARGET = Path("tests/test_chinese.py")

OLD_MISSING = '''def test_transcribe_missing_origin_raises_actionable_command():
    """题面原件不在盘上时报错要指到能执行的命令，而不是让下游拿空表算出 0% 还当实测。"""
    with pytest.raises(zh.ChineseTranscribeError, match="fetch --cn"):
        zh.load_subset("cmmlu-subset") if not (ASM / "cmmlu-subset/cmmlu-subset.jsonl").is_file() else \\
            (_ for _ in ()).throw(AssertionError("原件在盘上，此用例只该在缺件时验报错"))
'''

NEW_MISSING = '''def test_transcribe_missing_origin_raises_actionable_command(tmp_path, monkeypatch):
    """缺件时的三条报错路都得指到能敲的命令（把 DATA_DIR 挪进空目录来模拟"还没 fetch"）。"""
    monkeypatch.setattr(registry, "DATA_DIR", tmp_path / "not_fetched_yet")
    with pytest.raises(zh.ChineseTranscribeError, match="fetch --cn"):
        zh.load_subset("cmmlu-subset")                     # 题面原件没拉过
    with pytest.raises(zh.ChineseTranscribeError, match="fetch --zh"):
        zh.load_decision_records("cmmlu-decision")         # 中文信封没装配过
    with pytest.raises(registry.FetchError, match="fetch --sets cmmlu-subset"):
        registry._fetch_derived(registry.REGISTRY["cmmlu-decision"], tmp_path / "dst")
'''

OLD_DET = '''    _need_decision_sets()
    for src_pid, dec_pid in zh.SOURCE_TO_DECISION.items():
        rows = zh.transcribe_subset(src_pid, zh.load_subset(src_pid))
        disk = _read_jsonl(ASM / dec_pid / f"{dec_pid}.jsonl")
        for pid, entry in zip(registry.REGISTRY[dec_pid].qtypes, []):   # 仅取字段口，不参与判定
            assert pid in registry.QTYPES
        assert len(rows) == len(disk)
        for built, stored in zip(rows, disk):
            assert built["id"] == stored["id"] and built["qtype"] == stored["qtype"]
            assert built["sample"] == stored["sample"], f"{built['id']} 复跑结果漂移（样本体不等）"
        # 真值口径也逐条对得上：押满分的候选与源副本的人工答案一致
        for built, src in zip(rows, zh.load_subset(src_pid)):
            assert src["id"] == built["id"]
        assert registry._sha256(ASM / dec_pid / f"{dec_pid}.jsonl") == \\
            registry._sha256(ASM / dec_pid / f"{dec_pid}.jsonl")
'''

NEW_DET = '''    _need_decision_sets()
    for src_pid, dec_pid in zh.SOURCE_TO_DECISION.items():
        rows = zh.transcribe_subset(src_pid, zh.load_subset(src_pid))
        disk = _read_jsonl(ASM / dec_pid / f"{dec_pid}.jsonl")
        assert len(rows) == len(disk), f"{dec_pid} 复跑 {len(rows)} 条 vs 盘上 {len(disk)} 条"
        for built, stored in zip(rows, disk):
            assert built["id"] == stored["id"] and built["qtype"] == stored["qtype"]
            assert built["sample"] == stored["sample"], f"{built['id']} 复跑结果漂移（样本体不等）"
        assert _digest(rows) == _digest(disk), f"{dec_pid} 整批摘要与盘上不符（现推≠装配口）"
'''

OLD_QTYPES = '''        assert pin.qtypes == tuple(sorted({zh.ZH_CHANNEL and q for q in
                                           (("choice",) if src_pid == "cmmlu-subset"
                                            else ("choice", "noul"))}))
'''

NEW_QTYPES = '''        expect = ("choice",) if src_pid == "cmmlu-subset" else ("choice", "noul")
        assert pin.qtypes == expect, f"{dec_pid} 登记题型 {pin.qtypes} 与转写口径 {expect} 不符"
        assert set(pin.qtypes) <= set(registry.QTYPES), f"{dec_pid} 登记了公共契约外的题型"
'''

OLD_HYGIENE = '''    base = zh.load_mix_config(CONFIGS / "zh_mix_30.yaml")["mix"]
    for bad, why in ((zh.SOURCE_TO_DECISION["cmmlu-subset"], "评测集"),
                     ("not-registered-at-all", "没登记")):
        data = dict(yaml.safe_load((CONFIGS / "zh_mix_30.yaml").read_text(encoding="utf-8")))
        data["zh_corpus"] = bad
        path = tmp_path / f"bad_{why}.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        doc = zh.load_mix_config(path)
        assert doc["mix"] is not None and base == doc["mix"] | base if False else True
        with pytest.raises(zh.ChineseTranscribeError):
            zh.assert_mix_hygiene(doc)
'''

NEW_HYGIENE = '''    base = zh.load_mix_config(CONFIGS / "zh_mix_30.yaml")["mix"]
    for bad, why, pattern in ((zh.SOURCE_TO_DECISION["cmmlu-subset"], "evalset", "训测同集"),
                              ("not-registered-at-all", "unknown", "没在 registry 登记")):
        data = dict(yaml.safe_load((CONFIGS / "zh_mix_30.yaml").read_text(encoding="utf-8")))
        data["zh_corpus"] = bad
        path = tmp_path / f"bad_{why}.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        doc = zh.load_mix_config(path)
        assert doc["mix"]["zh_corpus"] == bad, "脏值得真进配比栏，否则这一闸是在空转"
        assert doc["mix"]["zh_ratio"] == base["zh_ratio"], "只该改语料引用，别把配比也顶掉了"
        with pytest.raises(zh.ChineseTranscribeError, match=pattern):
            zh.assert_mix_hygiene(doc)
    assert base["zh_corpus"] == zh.MIX_CORPUS_PENDING, "正例：在案可查的占位符必须放行"
    zh.assert_mix_hygiene(zh.load_mix_config(CONFIGS / "zh_mix_30.yaml"))
'''

OLD_READ = '''def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
'''

NEW_READ = '''def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest(rows: list[dict[str, Any]]) -> str:
    """一摞信封的契约字段（id/task/qtype/sample）摊成摘要——比散列文件更严：只认题面与真值。

    落盘那行还带 registry 记账时补的 split/溯源字段，直接比文件散列会把"记账字段"和"转写口径"
    混在一起谈；这里只摊四枚契约键，所以复跑与盘上不一致时点名的是转写规则本身。
    """
    canon = [{k: r[k] for k in ("id", "task", "qtype", "sample")} for r in rows]
    blob = json.dumps(canon, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
'''

PAIRS = [
    ("import json\nfrom pathlib import Path\n", "import hashlib\nimport json\nfrom pathlib import Path\n"),
    (OLD_READ, NEW_READ),
    (OLD_MISSING, NEW_MISSING),
    (OLD_DET, NEW_DET),
    (OLD_QTYPES, NEW_QTYPES),
    (OLD_HYGIENE, NEW_HYGIENE),
]


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    for i, (old, _new) in enumerate(PAIRS, start=1):
        n = text.count(old)
        assert n == 1, f"锚点 #{i} 命中 {n} 次（期望恰好 1 次），拒绝盲改：{old.splitlines()[0][:60]!r}"
    for old, new in PAIRS:
        text = text.replace(old, new, 1)
    for probe in ("def _digest(", "hashlib.sha256(blob", "not_fetched_yet",
                  "fetch --sets cmmlu-subset", "现推≠装配口", "与转写口径", "空转",
                  "zh.MIX_CORPUS_PENDING, \"正例"):
        assert probe in text, f"改完却找不到关键件：{probe!r}"
    for junk in ("(_ for _ in ()).throw", "zip(registry.REGISTRY[dec_pid].qtypes, [])",
                 "if False else True", "zh.ZH_CHANNEL and q"):
        assert junk not in text, f"劣质写法仍在盘上：{junk!r}"
    TARGET.write_text(text, encoding="utf-8")
    print(f"[patch-g] 测试件 4 处劣质写法清完，行数 {len(text.splitlines())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
