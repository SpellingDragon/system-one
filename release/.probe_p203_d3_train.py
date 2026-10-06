"""p2-03-D3 上游 train 档探查取证（中文语料登记的前置证据，同 D1 的 Intern 先例）。

跑法：cd release && .venv/bin/python .probe_p203_d3_train.py
产物：控制台逐条打印探查命令与输出原文，供 registry 的 ZH_TRAIN_PROBE 与 run notes 抄录。
禁 MPS；纯 CPU + 只读网络列举（不下载数据件）。
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parent
ZIP = BASE / "bench/eval_data/raw/cmmlu-subset/cmmlu_v1_0_1.zip"

print("=" * 72)
print("[1] 盘上取证：CMMLU pin 归档内部结构（zip 名列举，零流量）")
print(f"    命令：python -c \"zipfile.ZipFile('{ZIP.relative_to(BASE)}').namelist()\"")
if ZIP.is_file():
    names = zipfile.ZipFile(ZIP).namelist()
    tops = sorted({n.split('/')[0] for n in names if n})
    print(f"    顶层目录：{tops}")
    def _cnt(prep: str, *, csv_only: bool = True) -> int:
        keep = [n for n in names if n.startswith(f"{prep}/")]
        return sum(1 for n in keep if n.endswith(".csv")) if csv_only else len(keep)
    print(f"    csv 件数：dev={_cnt('dev')} test={_cnt('test')} "
          f"train={sum(1 for n in names if 'train' in n.lower())}"
          f"（条目数另含目录行：dev={_cnt('dev', csv_only=False)} "
          f"test={_cnt('test', csv_only=False)}）")
else:
    print(f"    [缺件] {ZIP} —— 先跑 fetch --cn")

print("=" * 72)
print("[2] 上游列举：modelscope/cmmlu 仓文件清单（看是否另有含 train 的归档）")
try:
    from modelscope.hub.api import HubApi

    api = HubApi()
    for repo in ("modelscope/cmmlu", "opencompass/clue"):
        print(f"    GET {repo}  ->  get_dataset_files(repo_id, revision='master', recursive=True)")
        files = api.get_dataset_files(repo_id=repo, revision="master", recursive=True,
                                      page_size=200)
        rows = [(f.get("Path"), f.get("Size")) for f in files] if files else []
        print(f"    件数={len(rows)}")
        for path, size in sorted(rows):
            print(f"      {path}  ({size} B)")
        trains = [p for p, _ in rows if "train" in str(p).lower()]
        print(f"    含 'train' 的路径：{trains if trains else '无'}")
        print("-" * 72)
except Exception as exc:  # noqa: BLE001  网络/授权形态原样报出，不猜结论
    print(f"    [探查失败] {type(exc).__name__}: {exc}")
print("=" * 72)
