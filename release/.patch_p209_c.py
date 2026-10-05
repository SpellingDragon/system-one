"""一次性补救脚本（p2-09 · A1）：接回被 `.patch_p209_b.py` 吃掉的 15 行 `def`，并清掉空白行尾巴。

【做什么】上一支搬运脚本有个真 bug：处理"单行 docstring + 其后白话注释块"那种形状时，只把
    改写后的 docstring 块写回，漏写了函数自己的 `def …:` 行——15 个函数的签名行因此丢了
    （`return` outside function 就是这么来的）。本脚本按"docstring 摘要行"逐一定位，把丢失的
    签名行接回原位；顺带把搬运时插入的分隔行由 4 个空格改成真正的空行（尾随空格是 lint 债）。
【怎么做】一张 (签名行 → 摘要行) 对照表，签名逐字抄自改动前的原始文件（会话里那份 Read 输出）：
    ① 断言摘要行在全文**恰好出现一次**；② 断言它的前一行不是该签名（若已是说明无需补）；
    ③ 在摘要行前插入签名。全部命中后才写盘。再把 `^\\s+$` 的行换成空行。
【为什么】靠"重新手写整个文件"来救会更贵也更险（可能顺手改掉别的段）；按锚点补回丢的那一行，
    改动范围可被逐条断言证明。签名不许凭记忆编：写错一个参数默认值，调用面就悄悄变了。
"""
import sys
from pathlib import Path

TARGET = Path("sys1/eval/chinese.py")

RESTORE = [
    ("def decision_id(source_pid: str) -> str:",
     '    """源副本集 id → 决策信封集 id（没登记过的名字当场报错，不静默回一个怪 id）。'),
    ("def load_subset(source_pid: str) -> list[dict[str, Any]]:",
     '    """读回原样子集副本的每一行；文件不在就抛带补救命令的错，不交一张空表冒充"读过了"。'),
    ("def cmmlu_to_envelope(row: dict[str, Any]) -> dict[str, Any]:",
     '    """CMMLU 一行（题干 + 四候选 + 正确字母）→ choice 信封（k=4，criteria=选项文本）。'),
    ("def tnews_to_envelope(row: dict[str, Any]) -> dict[str, Any]:",
     '    """CLUE tnews 一行（标题 + 类别下标）→ choice 信封（k=15，候选=源头给的类别码）。'),
    ("def ocnli_to_envelope(row: dict[str, Any]) -> dict[str, Any]:",
     '    """CLUE ocnli 一行（两句 + 三分类下标）→ noul 信封（蕴含=true，其余=false）。'),
    ("def clue_to_envelope(row: dict[str, Any]) -> dict[str, Any]:",
     "    \"\"\"CLUE 一行按它自带的 `task_name` 分派（tnews→choice / ocnli→noul）。"),
    ("def transcribe_subset(source_pid: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:",
     '    """一份原样子集副本 → 一摞决策信封（逐条转写，一条折不动就整批停下来说清是哪条）。'),
    ("def assemble_clue_decision(raw: Path) -> list[dict[str, Any]]:",
     '    """registry 装配口（clue-decision）：读 p2-03 那份 CLUE 副本，tnews/ocnli 各按各的折法。'),
    ("def load_decision_records(decision_pid: str, *, limit: int = 0) -> list[dict[str, Any]]:",
     '    """读回已落盘的中文决策信封（同 harness 同打分口的消费入口）。'),
    ("def load_mix_config(path: str | Path) -> dict[str, Any]:",
     '    """读一份中文配比档，拆成"现在就合法的配置键"与"待 p2-05 接入的配比键"两栏。'),
    ("def assert_mix_hygiene(doc: dict[str, Any]) -> None:",
     '    """配比档卫生闸：训练语料不得引用任何评测轴上的集（训测同集会出假分）。'),
    ("def compare_mix_configs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:",
     "    \"\"\"两档对照：除 `zh_ratio` 外必须逐键相同（消融只许动一个因子）。"),
    ("def build_ledger() -> dict[str, Any]:",
     '    """装配中文决策信封（走 registry 唯一的下载+装配+记账口），回账本与逐集底账。'),
    ("def build_parser() -> argparse.ArgumentParser:",
     '    """搭出中文轨 CLI 的参数表（build / g3 / check-mix 三个子命令）。'),
    ("def main(argv: list[str] | None = None) -> int:",
     '    """CLI 入口：build / g3 / check-mix 三条路，失败一律非零退出并列出原因。'),
]


def main() -> int:
    lines = TARGET.read_text(encoding="utf-8").split("\n")
    inserted = 0
    for sig, summary in RESTORE:
        hits = [i for i, ln in enumerate(lines) if ln == summary]
        assert len(hits) == 1, f"摘要行命中 {len(hits)} 次，无法定位：{summary[:50]!r}"
        i = hits[0]
        if lines[i - 1] == sig:
            continue                                          # 已在位（幂等复跑）
        assert not lines[i - 1].startswith("def "), f"前一行是别的 def：{lines[i - 1][:60]!r}"
        lines.insert(i, sig)
        inserted += 1
    blanked = 0
    for i, ln in enumerate(lines):
        if ln.strip() == "" and ln != "":
            lines[i] = ""
            blanked += 1
    text = "\n".join(lines)
    n_def = len([ln for ln in lines if ln.startswith("def ")])
    assert n_def == 27, f"def 行数 {n_def}，预期 27（15 补回 + 12 原本在位）；不符即不落盘"
    TARGET.write_text(text, encoding="utf-8")
    print(f"[patch-c] 接回 {inserted} 行 def 签名，清掉 {blanked} 行尾随空格，行数 {len(lines)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
