"""p2-07 定点修订：longctx 金标键 bug + prefix_cache 字节/属性补齐。"""
from pathlib import Path


def patch(path, pairs):
    P = Path(path)
    s = P.read_text()
    for old, new, label in pairs:
        assert s.count(old) == 1, f"锚点失配[{path}:{label}]: 命中 {s.count(old)} 次"
        s = s.replace(old, new, 1)
        print("ok", label)
    P.write_text(s)
    print("written", path, len(s.splitlines()), "lines")


# ---------------------------------------------------------------- ① A1 金标键 bug
# 现象：rng.shuffle(texts) 之后再拿 texts[0] 当正确值 —— 洗牌后 texts[0] 已是随机干扰项，
# 金标键会指到错误选项上（判分 recall 上不去的直接原因）。正确做法：洗牌前先记住正确值文字。
patch("sys1/eval/longctx.py", [
    ('''    texts = [str(probe.value), *[str(v) for v in picks]]
    rng.shuffle(texts)
    criteria = {str(i + 1): t for i, t in enumerate(texts)}       # 代号 1..k 按数值升序 → 字母 A..
    gold_key = next(k for k, t in criteria.items() if t == texts[0])''',
     '''    gold_text = str(probe.value)                                  # 洗牌前先把正确值钉住
    texts = [gold_text, *[str(v) for v in picks]]
    rng.shuffle(texts)
    criteria = {str(i + 1): t for i, t in enumerate(texts)}       # 代号 1..k 按数值升序 → 字母 A..
    gold_key = next(k for k, t in criteria.items() if t == gold_text)''',
     "gold_key_before_shuffle"),
])

# ---------------------------------------------------------------- ② B1a/B1b 补齐
patch("serving/prefix_cache.py", [
    # estimate_past_bytes 认不出 dict 形态的 past（假引擎就是这个形状）→ 回 0，字节预算形同虚设
    ('''    layers = getattr(past, "layers", None)
    if layers is None:
        layers = past if isinstance(past, (list, tuple)) else []
    total = 0''',
     '''    layers = getattr(past, "layers", None)
    if layers is None:
        layers = past if isinstance(past, (list, tuple)) else []
    if isinstance(past, dict):                       # 自建 past（测试替身/非 HF 引擎）常是 dict
        total = 0
        for item in past.values():
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
            elif isinstance(item, (list, tuple)):
                total += sum(x.numel() * x.element_size() for x in item if isinstance(x, torch.Tensor))
        return total
    total = 0''',
     "estimate_bytes_dict"),
    # AskResult 补一个真属性（as_dict 里那个键得有对应实体，测试才断言得到）
    ('''    def as_dict(self) -> dict[str, Any]:
        """摊成可写进 run 档案的窄表（张量不进产物文件）。"""''',
     '''    @property
    def tokens_are_suffix_only(self) -> bool:
        """B1b 的硬断言位：本次没为证据段付一个符号，且为问题段付的恰是问题段长度。"""
        return self.prefix_tokens_fed == 0 and self.suffix_tokens_fed == self.suffix_len

    def as_dict(self) -> dict[str, Any]:
        """摊成可写进 run 档案的窄表（张量不进产物文件）。"""''',
     "askresult_property"),
    ('''            "tokens_are_suffix_only": self.prefix_tokens_fed == 0
            and self.suffix_tokens_fed == self.suffix_len,''',
     '''            "tokens_are_suffix_only": self.tokens_are_suffix_only,''',
     "as_dict_use_property"),
])
