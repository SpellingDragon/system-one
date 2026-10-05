"""补插入 _maybe_clone 定义（被 clone_past 的列表回退引用）。"""
from pathlib import Path

P = Path("serving/prefix_cache.py")
s = P.read_text()
old = "# ---------------------------------------------------------------- 引擎协议\n@dataclass(frozen=True)\nclass EngineOut:"
assert s.count(old) == 1, s.count(old)
new = ('def _maybe_clone(value: Any) -> Any:\n'
       '    """张量就 `.clone()`，其他（None、标量、缓存壳）原样交回。"""\n'
       '    return value.clone() if isinstance(value, torch.Tensor) else value\n'
       '\n'
       '\n' + old)
P.write_text(s.replace(old, new, 1))
print("inserted _maybe_clone")
