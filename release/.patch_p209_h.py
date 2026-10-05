"""一次性补丁脚本（p2-09 · A1 测试件解包顺序修正）。

【做什么】`test_transcribe_assembled_ledger_three_mouths_agree` 把
    `zh.SOURCE_TO_DECISION.items()` 解成了 `for dec_pid, src_pid`——这个映射是"源集 → 派生集"，
    于是两个变量名与实际内容对调，用例实际在拿**题面原件**（cmmlu-subset / clue-subset）走
    "三口一致"，判到 `entry["envelope_ready"] is True` 时才红（原件那一格 envelope_ready=False，
    因为 p2-03 明确不决策化）。红得正好：证明这一闸有效。改成正确解包，让三口一致查的是
    中文决策集本身。
【怎么做】单点锚替换（断言原文恰好出现一次）；顺带把循环里没用上的源集变量做一次实际用途——
    比对派生集与原件的 split 是否同档（这条是 A1"split 显式"要守的第二处）。
【为什么】这条用例是本域"账面=盘上"的核心凭据；查错对象会给出全绿但毫无意义的结果，
    比直接失败更危险。留在测试件里当反面教材不合适，必须当场改掉而不是放宽断言。
"""
import sys
from pathlib import Path

TARGET = Path("tests/test_chinese.py")

OLD = '''    for dec_pid, src_pid in zh.SOURCE_TO_DECISION.items():
        entry = man[dec_pid]'''

NEW = '''    for src_pid, dec_pid in zh.SOURCE_TO_DECISION.items():
        assert registry.REGISTRY[dec_pid].split == registry.REGISTRY[src_pid].split, \\
            f"{dec_pid} 的档位与题面原件 {src_pid} 脱钩"
        entry = man[dec_pid]'''


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    n = text.count(OLD)
    assert n == 1, f"锚点命中 {n} 次（期望恰好 1 次），拒绝盲改"
    assert "for dec_pid, src_pid in" not in NEW
    text = text.replace(OLD, NEW, 1)
    assert "for dec_pid, src_pid in zh.SOURCE_TO_DECISION" not in text, "对调解包没改干净"
    TARGET.write_text(text, encoding="utf-8")
    print(f"[patch-h] 三口一致用例改查派生集本身，行数 {len(text.splitlines())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
