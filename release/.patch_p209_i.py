"""一次性补丁脚本（p2-09 · B2 失败样本用例的预测键形状修正）。

【做什么】`test_g3_failure_samples_pick_wrong_items_and_truncate_state` 判红：`preds` 传的是
    裸题号 `cmmlu:x:test/0`，而公共打分件 `scoring.rows_from_envelopes` 拆出来的行 id 是
    `"<题号>/<qid>"`（实测 `cmmlu:x:test/0/q`）——挂列函数按打分层的 id 查预测，两边必须同形。
    被测函数没错（现网预测出口 `predictions.jsonl` 就是这个键），错的是用例编的输入形状。
【怎么做】把构造的预测键改成 `f".../{zh.QID}"` 这一契约形状，并补两条断言：挑出来的样本 id
    带 `/q` 后缀、全部来自真正答错的那几题（把其中两题改成答对，就只该剩那一题被挑中）。
【为什么】这一格的凭据价值在于"它跟分数表用的是同一套判定"。用例若自造一种预测键形状，
    绿了也不能证明挂列接得上真预测——正是评审要复跑戳穿的那种"自证的绿"。
"""
import sys
from pathlib import Path

TARGET = Path("tests/test_chinese.py")

OLD = '''    wrong = {f"cmmlu:x:test/{i}": {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1} for i in range(5)}
    picked = zh.pick_failure_samples(records, wrong)
    assert len(picked) == zh.FAILURE_SAMPLE_SLOTS, "spec 要 3 例，多给少给都不算达标"
    assert all(s["gold"] == "A" and s["pred"] == "B" and s["k"] == 4 for s in picked)
    assert all(len(s["state"]) <= zh.FAILURE_STATE_CHARS + 1 for s in picked)
    right = {f"cmmlu:x:test/{i}": {"A": 0.9, "B": 0.03, "C": 0.03, "D": 0.04} for i in range(5)}
    assert zh.pick_failure_samples(records, right) == []
    assert zh.pick_failure_samples(records, {}) == []
'''

NEW = '''    # 预测键取打分层拆出来的 "<题号>/<qid>" 形状（与 predictions.jsonl 同形，不是本用例自造的）
    wrong = {f"cmmlu:x:test/{i}/{zh.QID}": {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1} for i in range(5)}
    picked = zh.pick_failure_samples(records, wrong)
    assert len(picked) == zh.FAILURE_SAMPLE_SLOTS, "spec 要 3 例，多给少给都不算达标"
    assert all(s["gold"] == "A" and s["pred"] == "B" and s["k"] == 4 for s in picked)
    assert all(s["id"].endswith(f"/{zh.QID}") for s in picked), "题号须与分数表同形，便于逐题追问"
    assert all(len(s["state"]) <= zh.FAILURE_STATE_CHARS + 1 for s in picked)
    assert picked[0]["state"].endswith("…"), "题面按字数截断并留省略号（不整段外抄）"
    right = {f"cmmlu:x:test/{i}/{zh.QID}": {"A": 0.9, "B": 0.03, "C": 0.03, "D": 0.04}
             for i in range(5)}
    assert zh.pick_failure_samples(records, right) == []
    assert zh.pick_failure_samples(records, {}) == []
    mixed = dict(right, **{f"cmmlu:x:test/2/{zh.QID}": {"A": 0.1, "B": 0.8, "C": 0.05, "D": 0.05}})
    only_one = zh.pick_failure_samples(records, mixed)
    assert [s["id"] for s in only_one] == [f"cmmlu:x:test/2/{zh.QID}"], "答对的题不该混进失败样本"
'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    n = text.count(OLD)
    assert n == 1, f"锚点命中 {n} 次（期望恰好 1 次），拒绝盲改"
    text = text.replace(OLD, NEW, 1)
    for probe in ('f"cmmlu:x:test/{i}/{zh.QID}"', "答对的题不该混进失败样本", 'endswith("…")'):
        assert probe in text, f"改完却找不到关键件：{probe!r}"
    assert 'wrong = {f"cmmlu:x:test/{i}"' not in text, "裸题号的预测键没改干净"
    TARGET.write_text(text, encoding="utf-8")
    print(f"[patch-i] 失败样本用例改用契约预测键，行数 {len(text.splitlines())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
