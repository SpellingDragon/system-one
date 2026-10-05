"""p2-01 真下载入口（走 production.assets.fetch_snapshot，非 monkeypatch）：拉 Qwen3.5-0.8B 全量快照。

跑法：cd release && .venv/bin/python .probe_p201_fetch.py
产物：控制台打印 Snapshot 溯源 JSON（字节量/耗时/端点/文件数），供 run notes 抄录。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from production.assets import fetch_snapshot  # noqa: E402

if __name__ == "__main__":
    snap = fetch_snapshot("qwen3.5-0.8b")
    print("PROVENANCE_JSON " + json.dumps(snap.as_config(), ensure_ascii=False))
