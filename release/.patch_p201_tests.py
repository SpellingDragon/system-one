"""p2-01 测试修正：接缝③"开关扳不动"那型必须让关闭态先过尾部检查，否则测的是上一型。

跑法：cd release && .venv/bin/python .patch_p201_tests.py
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "tests" / "test_assets.py"
OLD = '    same = _THINK_NOT_CLOSED + "x"'
NEW = '    same = _TPL_OFF + "x"'      # 尾部对得上（走第二型），开启态与关闭态同文才是"开关是摆设"

text = TARGET.read_text(encoding="utf-8")
assert text.count(OLD) == 1, f"锚点命中 {text.count(OLD)} 次"
TARGET.write_text(text.replace(OLD, NEW), encoding="utf-8")
print("patched tests/test_assets.py: think_switch_noop 改用 _TPL_OFF（尾部正确的第二型）")
