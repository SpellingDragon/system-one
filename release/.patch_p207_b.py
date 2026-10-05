"""p2-07 prefix_cache 三处定点修订：元组层字节算法、克隆回退、护栏覆盖冷启路。"""
from pathlib import Path

P = Path("serving/prefix_cache.py")
s = P.read_text()


def rep(old, new, label):
    global s
    assert s.count(old) == 1, f"锚点失配[{label}]: 命中 {s.count(old)} 次"
    s = s.replace(old, new, 1)
    print("ok", label)


# ① 老口径 past 是 [(keys, values), ...]：lay["keys"] 会 TypeError；改成按形态取对
rep('''    total = 0
    for lay in layers:
        for name in ("keys", "values"):
            item = lay[name] if isinstance(lay, (list, tuple)) and len(lay) >= 2 else None
            if item is None:
                item = getattr(lay, name, None) if not isinstance(lay, (list, tuple)) else None
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
    return total''',
    '''    total = 0
    for lay in layers:
        for item in _past_pair(lay):
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
    return total


def _past_pair(lay: Any) -> tuple[Any, Any]:
    """从"一层"里取出 `(keys, values)`：兼容 `(k, v)` 元组老口径与 `DynamicLayer` 新口径。"""
    if isinstance(lay, (list, tuple)):
        return (lay[0], lay[1]) if len(lay) >= 2 else (None, None)
    return getattr(lay, "keys", None), getattr(lay, "values", None)''',
    "past_bytes_pair")

# ② 克隆回退：老口径列表直接逐层 clone，别再用 type(past)(...) 猜构造签名
rep('''        if layers is None:                              # 退回 (keys, values) 元组列表的老口径
            clone = type(past)([(k.clone(), v.clone()) for k, v in past]) if isinstance(past, list) else past
            return clone''',
    '''        if layers is None:                              # 退回 (keys, values) 元组列表的老口径
            if isinstance(past, list):
                return [(_maybe_clone(k), _maybe_clone(v)) for k, v in past]
            return past''',
    "clone_list_fallback")

# ③ 护栏覆盖面：冷启路同样是"用了 past 的两段式前向"，漂移风险与命中路同源，必须一起验
rep('''        need_check = (
            hit
            and not force_full
            and self.readout is not None
            and (mode == "always" or (mode == "once" and key not in self._checked))
        )''',
    '''        used_past = (not force_full) and (key not in self._quarantine)
        need_check = (
            used_past
            and self.readout is not None
            and (mode == "always" or (mode == "once" and key not in self._checked))
        )''',
    "verify_covers_cold")

P.write_text(s)
print("written", P, len(s.splitlines()), "lines")
