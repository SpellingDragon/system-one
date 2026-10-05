"""一次性补丁脚本（p2-09 · A1 注释门）：把 `# 白话：` 段搬进各公共函数的 docstring。

【做什么】`scratch/tools/check_comments.py` 的 R2 只认 **docstring 里**以「白话：」起头的段落
    （`extract_plain_para` 走 `ast.get_docstring`），而 chinese.py 首版把这段解释写成了函数体
    里的行注释——读起来一样，工具判红。本脚本把每个公共函数 docstring 之后紧跟的那一整块
    `# 白话…` 连续注释搬进 docstring（单行式改写成"摘要 + 空行 + 白话段"，多行式在白话段
    前补一个空行后追加到结尾），并给 `source_id` / `fmt` 两个原本没写的补上一段。
【怎么做】按 `^def 名字(` 定位函数（名字以下划线开头的私有件整段跳过，R2 本就豁免它们）→
    定位其 docstring 首尾行 → 收下紧随其后的连续 `    # ` 注释块（仅当首行含「白话」才搬）→
    重写这一段。形状对不上（docstring 不闭合、段落末字符是引号会造出四连引号）即断言中止，
    不写盘；搬完用真工具 `check_comments.py` 复核，不靠脚本自说自话。
【为什么】注释门的判红不能靠"关掉工具"绕过；把白话段搬进 docstring 既过门又不减信息，
    密度口径（R3 把注释与 docstring 一起算）不受影响。私有函数留在体内的行注释读的是实现
    细节（怎么拼装、为什么这样兜底），不是对外语义，故原样保留。
"""
import re
import sys
from pathlib import Path

TARGET = Path("sys1/eval/chinese.py")
TOP = '"""'
PUBLIC = re.compile(r"^def (?!_)\w+\(")

EXTRA = {
    # source_id：原本一句 docstring，按 R2 要求补一段白话
    '    """决策信封集 id → 它的源副本集 id（回溯"这批题面是从哪份 pin 誊来的"）。"""':
        '    """决策信封集 id → 它的源副本集 id（回溯"这批题面是从哪份 pin 誊来的"）。\n'
        '\n'
        '    白话：分数要能追问出处，就得能从"考卷"倒回"题面原件"：这一集是从 p2-03 钉过版本的\n'
        '    那份副本誊出来的，源副本的 revision、取的是哪一档、总共几条，都记在原件那一格账上。\n'
        '    名字没登记过就当场报错，绝不猜一个看着顺眼的原件。\n'
        '    """',
    # fmt：表上占位牌的读法，原本只有 docstring 一句
    '    """把一个指标摊成表里的字符串；没实测的一律成占位牌，不当 0 也不当空。"""':
        '    """把一个指标摊成表里的字符串；没实测的一律成占位牌，不当 0 也不当空。\n'
        '\n'
        '    白话：G3 这张表上，"还没跑"与"跑了是 0"必须一眼分得开。所以不是数字的东西（没值、\n'
        '    空着、压根是个字符串）统统换成那块写着缺口的牌子，而不是 0——0 会被读成一个真分数。\n'
        '    """',
}


def _take_para(lines: list[str], start: int) -> tuple[list[str], int]:
    """从 start 行起收下连续的 `    # ` 注释块（去掉 `# ` 前缀），回（段落行, 下一个行号）。"""
    para, j = [], start
    while j < len(lines) and lines[j].startswith("    # "):
        para.append("    " + lines[j][6:])
        j += 1
    return para, j


def _single(lines: list[str], i: int) -> list[str] | None:
    """单行式 docstring + 其后白话块 → 改写；形状不符就回 None（原样不动）。"""
    first = lines[i].rstrip()
    body = first.strip()[3:-3].rstrip()
    para, _nxt = _take_para(lines, i + 1)
    if not para or "白话" not in para[0]:
        return None
    assert not body.endswith('"') and not para[-1].endswith('"'), f"会拼出四连引号：{body[-8:]!r}"
    return [f"    {TOP}{body}", "    ", *para, f"    {TOP}"]


def main() -> int:
    lines = TARGET.read_text(encoding="utf-8").split("\n")
    out: list[str] = []
    i = 0
    moved = 0
    while i < len(lines):
        ln = lines[i]
        if ln in EXTRA:
            out.extend(EXTRA[ln].split("\n"))
            moved += 1
            i += 1
            continue
        if PUBLIC.match(ln) and lines[i + 1].strip().startswith(TOP):
            if lines[i + 1].rstrip().endswith(TOP) and len(lines[i + 1].strip()) > 6:      # 单行式
                block = _single(lines, i + 1)
                if block is not None:
                    _, nxt = _take_para(lines, i + 2)
                    out.extend(block)
                    moved += 1
                    i = nxt
                    continue
            else:                                             # 多行式：先找收尾 """
                j = i + 2
                while j < len(lines) and lines[j].strip() != TOP:
                    j += 1
                assert j < len(lines), f"docstring 未闭合（{ln}）"
                para, nxt = _take_para(lines, j + 1)
                if para and "白话" in para[0]:
                    assert not para[-1].endswith('"'), f"段落末字符是引号：{para[-1][-8:]!r}"
                    out.extend(lines[i:j])
                    out.append("    ")
                    out.extend(para)
                    out.append(lines[j])
                    moved += 1
                    i = nxt
                    continue
        out.append(ln)
        i += 1
    TARGET.write_text("\n".join(out), encoding="utf-8")
    print(f"[patch-cmt] 搬运/补齐 {moved} 处白话段进 docstring，行数 {len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
