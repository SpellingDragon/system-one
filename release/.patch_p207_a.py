"""p2-07 serving/prefix_cache.py 定点补丁：修中文引号造成的语法错 + 清掉 serve() 里的废记账。"""
from pathlib import Path

P = Path("serving/prefix_cache.py")
s = P.read_text()


def rep(old, new, label):
    global s
    assert s.count(old) == 1, f"锚点失配[{label}]: {s.count(old)}"
    s = s.replace(old, new, 1)
    print("ok", label)


# ① 中文引号在双引号串里造成字符串提前结束
rep('raise CacheError("对话模板不是"逐轮拼接"的形态，无法安全截出 state 壳")',
    'raise CacheError("对话模板不是「逐轮拼接」的形态，无法安全截出 state 壳")',
    "quote_fix")

# ② serve() 里的占位垃圾记账：改成如实的"证据段被省下的 token 数"
rep('''        rows: list[dict[str, Any]] = []
        for req in requests:
            res = self.ask(req["prefix_ids"], req["suffix_ids"], namespace=namespace)
            rows.append({"request_id": req.get("request_id"), **res.as_dict()})
        hits = sum(1 for r in rows if r["hit"])
        saved = sum(r["suffix_len"] * 0 for r in rows)   # 占位：真正的节省量看 prefix_tokens_fed
        avoided = sum(
            len(r["prefix_tokens_fed"]) if False else 0 for r in rows
        )                                                # noqa: F841 （见下方 prefix_avoided）
        return {
            "rows": rows,
            "requests": len(rows),
            "hits": hits,
            "prefix_avoided": sum(1 for r in rows if r["hit"] and r["prefix_tokens_fed"] == 0),
            "tokens_fed_total": sum(r["tokens_fed"] for r in rows),
            "cache": self.cache.stats(),
            "counter": self.counter.as_dict(),
            "unused": saved + avoided,
        }''',
    '''        rows: list[dict[str, Any]] = []
        for req in requests:
            prefix_ids, suffix_ids = list(req["prefix_ids"]), list(req["suffix_ids"])
            res = self.ask(prefix_ids, suffix_ids, namespace=namespace)
            # prefix_len 由调用侧的入参给出（不塞进 AskResult，免得缓存层凭空"知道"长度）
            rows.append({"request_id": req.get("request_id"), "prefix_len": len(prefix_ids),
                         **res.as_dict()})
        hits = sum(1 for r in rows if r["hit"])
        # 省下的证据段 token = 命中且本次没为前缀付费的那些题，各自的前缀长度之和
        avoided = sum(r["prefix_len"] for r in rows if r["hit"] and r["prefix_tokens_fed"] == 0)
        return {
            "rows": rows,
            "requests": len(rows),
            "hits": hits,
            "hit_ratio": round((hits / len(rows)), 4) if rows else 0.0,
            "tokens_fed_total": sum(r["tokens_fed"] for r in rows),
            "tokens_avoided_in_prefix": avoided,
            "all_hits_suffix_only": all(
                r["prefix_tokens_fed"] == 0 and r["suffix_tokens_fed"] == r["suffix_len"]
                for r in rows if r["hit"]
            ),
            "cache": self.cache.stats(),
            "counter": self.counter.as_dict(),
        }''',
    "serve_ledger")

P.write_text(s)
print("written", P, len(s.splitlines()), "lines")
