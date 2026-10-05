"""p2-07 tests：B1c 弃缓存用例改为"冷启先入库、命中时判负"，并补记冷启路同样受护栏覆盖。"""
from pathlib import Path

P = Path("tests/test_longctx.py")
s = P.read_text()
old = '''def test_prefix_parity_over_threshold_quarantines_and_recomputes():
    engine = FakeEngine(perturb=0.9)                 # 漂移 9 分、分数尺度 ~200 → 相对漂移超阈值
    runner = PrefixRunner(engine, readout=readout, verify="always")
    runner.ask(PREFIX, SUFFIXES[0])
    res = runner.ask(PREFIX, SUFFIXES[0])
    report = runner.verdicts[-1]
    assert report.passed is False and res.drift_ratio > runner.drift_ratio_threshold
    assert res.recomputed is True and res.full_tokens_fed == len(PREFIX) + len(SUFFIXES[0])
    assert res.key in runner.stats()["quarantined"], "判负的 state 必须进黑名单"
    assert runner.cache.get(res.key) is None, "脏 past 要立刻从库里撕掉"
    again = runner.ask(PREFIX, SUFFIXES[1])
    assert again.hit is False and again.recomputed is True
    assert again.prefix_tokens_fed == 0 and again.full_tokens_fed == len(PREFIX) + len(SUFFIXES[1]), \\
        "拉黑之后一律整段重算，绝不拿脏小抄凑答案"'''
assert s.count(old) == 1
new = '''def test_prefix_parity_over_threshold_quarantines_and_recomputes():
    """漂移超限 → 当次就改用重算结果、拉黑该 state、脏 past 立即出库，此后一律重算。"""
    engine = FakeEngine(perturb=0.9)                 # 分数尺度 ≈0.17，漂移 0.9 → 相对漂移远超阈值
    runner = PrefixRunner(engine, readout=readout, verify="off")
    runner.ask(PREFIX, SUFFIXES[0])                  # 先冷启入库，把"命中路判负"这条路走实
    res = runner.ask(PREFIX, SUFFIXES[0], verify="always")
    report = runner.verdicts[-1]
    assert res.hit is True and res.drift is not None
    assert report.passed is False and res.drift_ratio > runner.drift_ratio_threshold
    assert res.recomputed is True, "判负当次就该交重算结果，而不是把脏复用的答案递出去"
    assert res.full_tokens_fed == len(PREFIX) + len(SUFFIXES[0])
    assert res.key in runner.stats()["quarantined"], "判负的 state 必须进黑名单"
    assert runner.cache.get(res.key) is None, "脏 past 要立刻从库里撕掉"
    again = runner.ask(PREFIX, SUFFIXES[1])
    assert again.hit is False and again.recomputed is True
    assert again.prefix_tokens_fed == 0 and again.full_tokens_fed == len(PREFIX) + len(SUFFIXES[1]), \\
        "拉黑之后一律整段重算，绝不拿脏小抄凑答案"


def test_prefix_parity_cold_two_step_path_is_also_guarded():
    """冷启路（前缀前向 + 问题段增量）同样用了 past，护栏必须一并覆盖，不许只验命中路。"""
    cold_first = PrefixRunner(FakeEngine(perturb=0.9), readout=readout, verify="always")
    first = cold_first.ask(PREFIX, SUFFIXES[0])
    assert first.hit is False and first.drift is not None, "第一问（两段式）就该被验一次"
    assert first.recomputed is True and len(cold_first.stats()["quarantined"]) == 1'''
P.write_text(s.replace(old, new, 1))
print("written", P, len(new.splitlines()))
