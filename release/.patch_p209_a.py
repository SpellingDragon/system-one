"""一次性补丁脚本（p2-09 · A1）：修 sys1/eval/chinese.py 自查出的四处瑕疵。

【做什么】把 chinese.py 首版里四处"能跑但难看/绕圈"的写法按锚点替换掉：
    ① `_finish` 重复调用 `_revalidate` 且先 update 再覆盖再 pop 的混乱拼装；
    ② `zh_source=SOURCE_TO_DECISION and "cmmlu-subset"` 这种借 and 短路蒙身份的怪表达式；
    ③ `load_decision_records` 里 `registry.__dict__["json"] and [...]` 的无意义 hack；
    ④ 恒为 None 的假常量 `TNEWS_INSTRUCTIONS`（没人读它，留着像"有个问法开关"）。
【怎么做】每处一个 (锚点原文 → 新文) 对，替换前断言锚点在文件里**恰好出现一次**，
    任一处对不上就整体不写盘并报错——绝不做行号删除，也绝不"没匹配上就当改好了"。
【为什么】编辑工具一次写整文件容易把别处一起改了；小块带断言的替换能证明"只动了这四段"。
"""
import sys
from pathlib import Path

TARGET = Path("sys1/eval/chinese.py")

PAIRS = [
    # ① _finish 的行拼装
    ('''    row = {"id": env["id"], "task": channel, "qtype": q["type"]}
    row.update(_revalidate(env)["sample"])          # sample（已过契约）取回校验后的那份
    row["sample"] = _revalidate(env)["sample"]
    row.pop("state", None)
    row.pop("questions", None)
    row.pop("targets", None)
    row.update(extra)                               # subject / task_name 之类的溯源字段原样带上
    return row''',
     '''    sample = _revalidate(env)["sample"]             # 动过文字，重新过一遍样本契约
    row = {"id": env["id"], "task": channel, "qtype": q["type"], "sample": sample}
    row.update(extra)                               # subject / task_name 之类的溯源字段原样带上
    return row'''),

    # ② CMMLU 的溯源字段
    ('''                   subject=str(row.get("subject") or "unknown"),
                   zh_source=SOURCE_TO_DECISION and "cmmlu-subset")''',
     '''                   subject=str(row.get("subject") or "unknown"),
                   zh_source="cmmlu-subset")'''),

    # ③ 决策信封读盘
    ('''    rows = registry.__dict__["json"] and [json.loads(x) for x in
                                          path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[:limit] if limit else rows''',
     '''    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return rows[:limit] if limit else rows'''),

    # ④ 去掉恒为 None 的假常量
    ('''#: tnews 是主题归类，转写既有件的默认问法（"这段话属于哪一类？"）本就对味，不另写一句。
TNEWS_INSTRUCTIONS = None''',
     '''# tnews 是主题归类，转写既有件的默认问法（"这段话属于哪一类？"）本就对味：故 `tnews_to_envelope`
# 不传 instructions，这里也不留一个恒为 None 的假常量（看着像开关，实则没人读它）。'''),
]


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(PAIRS, start=1):
        n = text.count(old)
        assert n == 1, f"锚点 #{i} 出现 {n} 次（期望恰好 1 次），拒绝盲改：{old.splitlines()[0][:60]!r}"
    for old, new in PAIRS:
        text = text.replace(old, new, 1)
    assert "TNEWS_INSTRUCTIONS" not in text, "假常量仍在文件里"
    assert "SOURCE_TO_DECISION and" not in text, "怪表达式仍在"
    assert 'registry.__dict__' not in text, "registry 内省 hack 仍在"
    assert text.count("_revalidate(env)") == 1, f"_revalidate 调用数应为 1，实得 {text.count('_revalidate(env)')}"
    TARGET.write_text(text, encoding="utf-8")
    print(f"[patch-a] {TARGET} 四处替换完成，行数 {len(text.splitlines())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
