"""p2-03 D3 补丁 C：run.py 把中文 train 语料的账与缺口写进 train 底账 / run notes。

跑法：cd release && .venv/bin/python .patch_p203_d3_c.py
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).resolve().parent / "sys1" / "eval" / "run.py"
src = TARGET.read_text(encoding="utf-8")
edits: list[tuple[str, str]] = []

# ── 1. load_train_records 的 docstring：口径从"只有 typed train"改成"含中文档" ──
edits.append((
    '''    """训练数据消费口：只回 `split=='train'` 的统一信封（typed-decisions train 档）。
''',
    '''    """训练数据消费口：只回 `split=='train'` 的统一信封（typed train + 中文 train 语料，D3）。
''',
))

# ── 2. train_ledger：分列出中文那一格 + 带上中文探查结论 ──────────────────
edits.append((
    '''    man = manifest if manifest is not None else registry.load_manifest()
    sets = man.get("sets") or {}
    rows = []
    for pid, pin in registry.REGISTRY.items():
        if pin.axis != TRAIN_AXIS:
            continue
        s = sets.get(pid) or {}
        rows.append({"id": pid, "status": s.get("status", "absent"),
                     "samples": int(s.get("samples") or 0),
                     "bytes": int(s.get("bytes_downloaded") or s.get("bytes") or 0),
                     "qtype_counts": s.get("qtype_counts") or {},
                     "channel": s.get("endpoint") or s.get("source_url") or "-"})
    return {"sets": rows, "samples": sum(r["samples"] for r in rows),
            "bytes": sum(r["bytes"] for r in rows),
            "intern_probe": registry.INTERN_TRAIN_PROBE["verdict"]}
''',
    '''    man = manifest if manifest is not None else registry.load_manifest()
    sets = man.get("sets") or {}
    rows = []
    for pid, pin in registry.REGISTRY.items():
        if pin.axis != TRAIN_AXIS:
            continue
        s = sets.get(pid) or {}
        rows.append({"id": pid, "status": s.get("status", "absent"),
                     "samples": int(s.get("samples") or 0),
                     "bytes": int(s.get("bytes_downloaded") or s.get("bytes") or 0),
                     "qtype_counts": s.get("qtype_counts") or {},
                     "channel": s.get("endpoint") or s.get("source_url") or "-"})
    # 中文那一格单独列出（D3）：C5 的 zh_corpus 要指着它，缺口也必须能单独读出来——
    # 混在总数里就看不出"typed 有了、中文还欠着 CMMLU 那一档"这句实话
    zh_rows = [r for r in rows if r["id"] in registry.ZH_TRAIN_IDS_TUPLE]
    return {"sets": rows, "samples": sum(r["samples"] for r in rows),
            "bytes": sum(r["bytes"] for r in rows),
            "zh_sets": zh_rows, "zh_samples": sum(r["samples"] for r in zh_rows),
            "zh_bytes": sum(r["bytes"] for r in zh_rows),
            "intern_probe": registry.INTERN_TRAIN_PROBE["verdict"],
            "zh_probe": registry.ZH_TRAIN_PROBE["verdict"]}
''',
))

# ── 3. 报告模板：train 底账行补中文语料与缺口 ────────────────────────────
edits.append((
    '''        f"Intern-Decision train 探查：{tl['intern_probe']}",
''',
    '''        f"Intern-Decision train 探查：{tl['intern_probe']}",
        f"- 中文 train 语料（D3 · C5 前置闸之二）：{len(tl['zh_sets'])} 集 / "
        f"{tl['zh_samples']:,} 条 / 真实下载 {tl['zh_bytes']:,} 字节"
        f"（通道 {'、'.join(str(r['channel']) for r in tl['zh_sets'] if r['status'] in ('fetched', 'cached')) or '未拉'}）；"
        f"CMMLU train 探查：{tl['zh_probe']}",
''',
))

# ── 4. run notes：观察行补一句中文语料底账与缺口上报 ──────────────────────
edits.append((
    '''            f"Intern-Decision train 探查结论：{tl['intern_probe']}"
            f"（取证见 registry.INTERN_TRAIN_PROBE）——查无即如实记，不硬造分区凑数",
        ])
''',
    '''            f"Intern-Decision train 探查结论：{tl['intern_probe']}"
            f"（取证见 registry.INTERN_TRAIN_PROBE）——查无即如实记，不硬造分区凑数",
            f"中文 train 语料底账（D3，C5 前置闸之二）："
            f"{[(r['id'], r['status'], r['samples']) for r in tl['zh_sets']]} / "
            f"真实下载 {tl['zh_bytes']:,} 字节；中文档 id 与考卷 id 逐条互斥"
            f"（档位段 train vs validation/test 写在 id 里，断言见 tests/test_registry.py "
            f"-k split_isolation）；语料缺口上报：{tl['zh_probe']}"
            f"（取证见 registry.ZH_TRAIN_PROBE）——查无即如实记，不拿 dev 档改名凑 train",
        ])
''',
))

for old, new in edits:
    hits = src.count(old)
    if hits != 1:
        raise SystemExit(f"锚点命中 {hits} 次（要求恰好 1 次）：{old[:70]!r}")
    src = src.replace(old, new, 1)

TARGET.write_text(src, encoding="utf-8")
print(f"[patch-c] run.py 落笔 {len(edits)} 处 -> {TARGET}")
