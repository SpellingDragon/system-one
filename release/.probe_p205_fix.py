"""p2-05 测试修补②：玩具注意力补齐四枚投影 + 两处算式写正（一次性脚本）。"""
import pathlib
import sys

P = pathlib.Path("tests/test_prod_sft.py")
src = P.read_text(encoding="utf-8")

PAIRS = [
    ('    def forward(self, x):\n        return self.o_proj(torch.tanh(self.q_proj(x)))\n',
     '    def forward(self, x):\n'
     '        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)\n'
     '        return self.o_proj(q * torch.tanh(k) + v)      # 四枚都进图，注入面才算全覆盖\n'),
    ('    assert counts["trainable"] == 8 * (4 * HID)               # 每枚 A(4,16)+B(16,4)\n',
     '    assert counts["trainable"] == len(loras) * 2 * (4 * HID)  # 每枚垫片 A(4,16)+B(16,4)\n'),
    ('    base_w = torch.randn(8, HID)\n', '    base_w = torch.randn(HID, HID)\n'),
    ('    x = torch.randn(2, 3, HID)\n\n    holder_mine = _ToyAttn()',
     '    x = torch.randn(2, 3, HID)\n\n    holder_mine = _ToyAttn()'),
]

fail = []
for old, new in PAIRS:
    n = src.count(old)
    if n != 1:
        fail.append((n, old.strip().splitlines()[0][:50]))
        continue
    src = src.replace(old, new)
P.write_text(src, encoding="utf-8")
print("MISS:", fail)
sys.exit(0 if not fail else 1)
