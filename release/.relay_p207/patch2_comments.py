# -*- coding: utf-8 -*-
"""p2-07 接力补丁②：注释门（R2 白话段落）—— sys1/eval/longctx.py 6 处 + sys1/layers/attention.py 2 处。

用法：cd release && .venv/bin/python .relay_p207/patch2_comments.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path("/Users/pengweiye/Documents/codes/system-one/release")

EDITS: list[tuple[str, str, str]] = [
    # ---------------- sys1/eval/longctx.py ----------------
    (
        "sys1/eval/longctx.py",
        '    """按档位分组并把档内按 `pos_frac` 升序排（正文按插入顺序拼，序乱则位置失真）。"""',
        '    """按档位分组并把档内按 `pos_frac` 升序排（正文按插入顺序拼，序乱则位置失真）。\n'
        '\n'
        '    白话：先把同一档长度的考卷归成一摞，再把这摞里的几根针按"插在文章几成处"从头排到尾。\n'
        '    正文是一个字往后写出来的，针的顺序一乱，落点就全错位——落点错了，"中段比开头难捞"这种\n'
        '    诊断价值也就没了。\n'
        '    """',
    ),
    (
        "sys1/eval/longctx.py",
        '    """凑错选项：先取同篇其它针的编号（最像的干扰），不够再从 100..999 里补，绝不与正确值同值。"""',
        '    """凑错选项：先取同篇其它针的编号（最像的干扰），不够再从 100..999 里补，绝不与正确值同值。\n'
        '\n'
        '    白话：给正确答案配几个像样的干扰项。就近取材最狠——同一篇里别的针的编号也是三位数、也\n'
        '    真在文中出现过，蒙不中就说明是真没捞到；凑不齐才去 100..999 里补，补进来的绝不与正确项\n'
        '    撞值（撞值就是两道正确答案）。\n'
        '    """',
    ),
    (
        "sys1/eval/longctx.py",
        '    """给这批正文称个重：条数、总字数、逐档字节与整批 sha256 前 16 位（复跑对得上才算确定）。"""',
        '    """给这批正文称个重：条数、总字数、逐档字节与整批 sha256 前 16 位（复跑对得上才算确定）。\n'
        '\n'
        '    白话：把这批正文整个压成一个"重量级暗号"，再按档位分开记体量。两次构建暗号逐字相同，\n'
        '    才配说这份考卷可复现；暗号一变就说明拼装口径被动过，报告里的召回曲线就得重跑。\n'
        '    """',
    ),
    (
        "sys1/eval/longctx.py",
        '    """1M 那一行的双列口径：没实测就 `measured: —`，配置是否备好另说（防虚报的制度化）。"""',
        '    """1M 那一行的双列口径：没实测就 `measured: —`，配置是否备好另说（防虚报的制度化）。\n'
        '\n'
        '    白话：报告里 1M 那一格，本机到底跑没跑过？没跑过就把"实测"留空，只在"配置"那一列写\n'
        '    ready。两列各说各话，读的人才不会把"配置备好了"误读成"量出来的召回"——这正是父变更\n'
        '    反复强调的"不虚报"落到表头上的写法。\n'
        '    """',
    ),
    (
        "sys1/eval/longctx.py",
        '    """CLI：`build` 出正文副本，`selfcheck` 只报针位表与计划是否同源（不写盘）。"""',
        '    """CLI：`build` 出正文副本，`selfcheck` 只报针位表与计划是否同源（不写盘）。\n'
        '\n'
        '    白话：命令行就两个动作。`build` 按针位表把正文造出来、落盘；`selfcheck` 一个字都不写，\n'
        '    只回答一个问题——"盘上那张座位表"和"代码照种子摇出来的那张"是不是同一张。\n'
        '    """',
    ),
    (
        "sys1/eval/longctx.py",
        '    """CLI 出口：selfcheck 打印指纹；build 落盘并回 sha256 与条数。"""',
        '    """CLI 出口：selfcheck 打印指纹；build 落盘并回 sha256 与条数。\n'
        '\n'
        '    白话：`selfcheck` 把同源核对打成一行，方便人一眼看清"能不能拿代码推的那张当题"；\n'
        '    `build` 造完正文后把条数、逐档台账、指纹一起报出来——报的数字全来自刚生成的那批行，\n'
        '    不另算一遍，也就不存在"报的与写的不是同一份"。\n'
        '    """',
    ),
    # ---------------- sys1/layers/attention.py ----------------
    (
        "sys1/layers/attention.py",
        '        """摊进 run 档案：窗上限与夹了几层必须留痕，否则"封顶"只是嘴上封顶。"""',
        '        """摊进 run 档案：窗上限与夹了几层必须留痕，否则"封顶"只是嘴上封顶。\n'
        '\n'
        '        白话：把这张排班表抄成一份能读的账——每层看多宽、封顶闸门设在哪、夹住了几层，一行\n'
        '        都不落。日后追问"你那次到底封没封顶"，账上就有答案，不用回头翻代码。\n'
        '        """',
    ),
    (
        "sys1/layers/attention.py",
        '    白话：这里刻意**不**给 None。主干 `CausalAttention` 的 `visible=None` 分支是"任何掩码\n'
        '    都不加"（双向可见），并不是"全因果"——只有 `Decoder.forward` 才会替它补上因果带。\n'
        '    逐层清单要能直接喂进 `DecoderBlock`，所以"看全场"这一层也得把因果带画出来。\n',
        '    白话：这里刻意**不留空不发**。按主干的规矩，"这张纸不发"等于"谁都能看见谁"，因果当场\n'
        '    漏掉；所以连"看全场"那一层也要发一张满宽的"只许回头看"，才敢直接逐层喂下去。\n'
        '\n'
        '    技术注：`visible=None` 在 `CausalAttention` 里是"任何掩码都不加"的双向分支，并不是\n'
        '    "全因果"——只有 `Decoder.forward` 才替它补上因果带；逐层清单要能直接喂进 `DecoderBlock`，\n'
        '    所以"看全场"这一层也得把因果带画出来。\n',
    ),
]


def main() -> None:
    cache: dict[str, str] = {}
    for rel, old, new in EDITS:
        src = cache.get(rel)
        if src is None:
            src = (ROOT / rel).read_text(encoding="utf-8")
        n = src.count(old)
        assert n == 1, f"补丁②锚点命中 {n} 次（需 1 次）@{rel}：{old[:60]!r}"
        cache[rel] = src.replace(old, new)
    for rel, text in cache.items():
        (ROOT / rel).write_text(text, encoding="utf-8")
        print(f"补丁②落地：{rel}")
    print(f"共 {len(EDITS)} 处")


if __name__ == "__main__":
    main()
