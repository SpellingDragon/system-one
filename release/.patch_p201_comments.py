"""一次性补齐 p2-01 半成品的 R2『白话:』段落（check_comments 门禁 8 处违例）。

跑法：cd release && ../scratch/.venv/bin/python .patch_p201_comments.py   （在仓库根跑亦可）
口径：只加 docstring 段落，不动任何逻辑；每处替换都要求原文唯一命中，命中不到就报错退出。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EDITS: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- production/assets.py
    ("production/assets.py",
     '        """摊成能直接写进 run config 的扁平字典（键名带 backbone_ 前缀，避免与训练参数撞名）。"""',
     '        """摊成能直接写进 run config 的扁平字典（键名带 backbone_ 前缀，避免与训练参数撞名）。\n'
     '\n'
     '        白话：把这次搬东西的账目摊平成一行一项的条目——从哪家搬的、哪个版本、搬进多少、\n'
     '        用了多久、落在那个柜子的哪一格——抄进实验记录本就是一份能复查的凭据。"""'),
    ("production/assets.py",
     '        """摊平成 run config 字段（接缝证据跟着 run 走，报告里能指回来）。"""',
     '        """摊平成 run config 字段（接缝证据跟着 run 走，报告里能指回来）。\n'
     '\n'
     '        白话：四道关卡各自量到的数——二十六个字母的号、挑出来的行有多宽、套壳尾巴的指纹\n'
     '        与长度、贴图那几个记号的号——一并摊平，报告里每句结论都能指回这里核对。"""'),
    ("production/assets.py",
     '        """run config 要的溯源 + 接缝字典（repo/revision 必含，spec "来源可溯"场景）。"""',
     '        """run config 要的溯源 + 接缝字典（repo/revision 必含，spec "来源可溯"场景）。\n'
     '\n'
     '        白话：把"东西从哪家搬来、搬了多少、过了哪几道检查"合成一张说明单一次交齐，\n'
     '        省得写记录时东抄一行西抄一行，最要紧的出处反而漏掉。"""'),
    # ---------------------------------------------------------------- production/backbone/loader.py
    ("production/backbone/loader.py",
     '    """读 model.safetensors.index.json（weight_map + 总规模）；没有分片 index 时回空表。"""',
     '    """读 model.safetensors.index.json（weight_map + 总规模）；没有分片 index 时回空表。\n'
     '\n'
     '    白话：先把这本"页码目录"翻开——哪个数记在哪一页、那一页又在哪一本书里。后面点名\n'
     '    取数全照着它走；没有目录时不猜，交一张空表出去，让调用侧自己决定要不要另想办法。"""'),
    ("production/backbone/loader.py",
     '        """一行文本凭据（进 run notes：层号 / 最大差 / 相对差 / 判定，事后可复跑核对）。"""',
     '        """一行文本凭据（进 run notes：层号 / 最大差 / 相对差 / 判定，事后可复跑核对）。\n'
     '\n'
     '        白话：把这层的比账压成一行字——第几层、两边最大差多少、参考侧平均多重、折算成\n'
     '        倍数是多少、允许的限度划在哪儿、最后判过没过——一行就能直接进记录本。"""'),
    # ---------------------------------------------------------------- production/backbone/layout.py
    ("production/backbone/layout.py",
     '        """一个大头摊到几个小头上（分组注意力的复用倍数；MHA 时为 1）。"""',
     '        """一个大头摊到几个小头上（分组注意力的复用倍数；MHA 时为 1）。\n'
     '\n'
     '        白话：问的一方有八个头、答的一方只备了两套，那就让每套顶着四家用；这个"顶几回"\n'
     '        的倍数从这里算。除不尽说明这副身板的头数配不成组，当场说清，不留含糊。"""'),
    ("production/backbone/layout.py",
     '        """内核配对的另一半起点（head_dim 的一半）。"""',
     '        """内核配对的另一半起点（head_dim 的一半）。\n'
     '\n'
     '        白话：一个头里的坐标排成前后两排坐，前排第几位与后排同列那位结对子；这里交出\n'
     '        的就是后排的起点，也就是头宽的一半。头宽是奇数就两两配不上对，直接说清。"""'),
    ("production/backbone/layout.py",
     '    """只补小头、不换座次的行号列子（值投影用；也用于"原始坐次"的对照实验）。"""',
     '    """只补小头、不换座次的行号列子（值投影用；也用于"原始坐次"的对照实验）。\n'
     '\n'
     '    白话：跟换了座次那条只差一步：这里只把"两套顶八家用"的连号点名做完（0,0,0,0,1,1,1,1\n'
     '    这样），每个头内部一个座次都不动。不参与转角的那一路本来就该照原样坐，拿来当对照也合适。"""'),
]


def main() -> int:
    cache: dict[str, str] = {}
    for rel, old, new in EDITS:
        text = cache.get(rel) or (ROOT / rel).read_text(encoding="utf-8")
        n = text.count(old)
        if n != 1:
            print(f"FAIL {rel}: 原文命中 {n} 次（需恰好 1 次）：{old[:60]}...")
            return 1
        cache[rel] = text.replace(old, new)
    for rel, text in cache.items():
        (ROOT / rel).write_text(text, encoding="utf-8")
        print(f"patched {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
