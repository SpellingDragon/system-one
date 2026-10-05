"""p2-01 B2 真实路径重测：走 production.assets.fetch_snapshot 向 ModelScope 全新拉一份 0.8B（非缓存预置）。

跑法：cd release && .venv/bin/python .probe_p201_redownload.py
口径：cache_dir 指到一个空目录 bench/p201_dl_fresh，于是"这次真搬进来多少字节"必须等于快照规模，
      不能拿自家柜子早就有的那份 1.6G 冒充下载量（P1 教训：远端路径 mock 通过≠路径可用）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from production.assets import fetch_snapshot  # noqa: E402

if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "bench" / "p201_dl_fresh"
    t0 = time.time()
    snap = fetch_snapshot("qwen3.5-0.8b", cache_dir=root)
    print("WALL_SECONDS %.1f" % (time.time() - t0))
    print("PROVENANCE_JSON " + json.dumps(snap.as_config(), ensure_ascii=False))
