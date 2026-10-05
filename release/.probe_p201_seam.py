"""p2-01 探针：Qwen3.5-0.8B tokenizer 接缝实测（字母单 token / 边界 / 思考关闭前缀）。"""
import os, string, time, json
os.environ.setdefault("MODELSCOPE_CACHE", "bench/ms_models")
from modelscope import snapshot_download

t0 = time.time()
p = snapshot_download(
    "Qwen/Qwen3.5-0.8B",
    cache_dir="bench/ms_models",
    allow_patterns=["*.json", "*.txt", "*.jinja", "tokenizer*", "vocab*", "merges*"],
)
dt = time.time() - t0
print("snapshot dir:", p)
print("elapsed tokenizer-only: %.1fs" % dt)
print("files:", sorted(os.listdir(p)))
