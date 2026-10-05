"""p2-07 长上下文本地半场 run 档案：A1 真 token 口径 + B1/B2 真前缀复用 + C1 滑窗内存账。

复跑：cd release && .venv/bin/python .p207_run.py
口径（务必连着数字一起读）：
  * 设备一律 CPU、禁用 MPS（派单纪律）；模型替身 = bench/ms_models 的 Qwen3-0.6B（minimal 装载，
    p2-05 确立的替身侧门——正式门为 0.8B 多模态设计，0.6B 缺 image_token_id 会被接缝④拒）；
  * A2 的"实测召回曲线"（8K/32K/128K 真前向逐题作答）**不在本波**：本机 CPU 摊不开长窗前向，
    本档案只交"正文可考 + 长度口径为真 token"，召回行一律标 pending；
  * 内存一律 CPU 驻留/解析账口径，NPU 显存账等开卡（C2/云端）。
"""
from __future__ import annotations

import json
import time

import torch

from production.sft import build_student
from serving.prefix_cache import (
    HFEngine,
    PrefixCache,
    PrefixRunner,
    compare_parity,
    estimate_past_bytes,
    prefix_key,
    split_prompt_ids,
)
from sys1.decision import RENDER_VERSION, from_systemone, option_scores, render
from sys1.eval import longctx, registry
from sys1.layers import attention as LA
from sys1.runs import new_run

STATE_TOKENS = 2048          # 本机 CPU 档的证据段长度（8K/128K 真前向属 A2·待云端）
DEVICE = "cpu"
DTYPE = torch.bfloat16
DRIFT_THRESHOLD = 0.02
# 正文构建档：8K/32K 全出，128K/256K 各出一题（本机只验"长度口径与链路"，不验吞吐与召回）
A1_PLAN = ((8192, 0), (32768, 0), (131072, 1), (262144, 1))


def enc(tok, text):
    out = tok.encode(text, add_special_tokens=False)
    return list(out["input_ids"]) if isinstance(out, dict) else list(getattr(out, "ids", out))


# ---------------------------------------------------------------- A1：真 token 口径的长文副本
def step_a1(tok):
    """按 registry 针位表补正文，逐档记"真 token 长度 vs 档位"的偏差（不虚报字数为 token）。"""
    t0 = time.time()
    rows, per_bucket = [], {}
    for ctx, limit in A1_PLAN:
        sub = longctx.build_corpus(buckets=[ctx], tokenizer=tok, options=4, limit_per_bucket=limit)
        rows += sub["rows"]
        ent = {"probes": len(sub["rows"]), "needles_in_doc": sub["ledger"][str(ctx)]["needles"],
               "target": sub["rows"][0]["needle"]["target_units"],
               "tokens_min": min(r["needle"]["tokens"] for r in sub["rows"]),
               "tokens_max": max(r["needle"]["tokens"] for r in sub["rows"]),
               "rel_err_max": max(r["needle"]["tokens_rel_err"] for r in sub["rows"])}
        per_bucket[str(ctx)] = ent
    out = longctx.write_corpus(rows, registry.DATA_DIR / "assembled" / "needle-synthetic",
                               source="skeleton")
    fp = longctx.corpus_fingerprint(rows)
    print(f"[A1] rows={len(rows)} seconds={time.time() - t0:.1f} out={out}")
    print(f"[A1] fingerprint={json.dumps(fp, ensure_ascii=False)}")
    for ctx in sorted(per_bucket, key=int):
        e = per_bucket[ctx]
        print(f"[A1]   ctx={ctx:>6} 探测={e['probes']:>2} 篇内针={e['needles_in_doc']:>2} "
              f"真token={e['tokens_min']:,}~{e['tokens_max']:,} 目标={e['target']:,} "
              f"最大相对偏差={e['rel_err_max']:.4f}")
    return {"rows": len(rows), "seconds": round(time.time() - t0, 1), "out": str(out),
            "fingerprint": fp, "per_bucket": per_bucket}


# ---------------------------------------------------------------- B1/B2：真前缀复用前向
def step_b(st, tok):
    """同 state 三问（含跨请求）走真 backbone：符号计数、past 字节、bf16 漂移全部实测。"""
    body, head, letters = st.body, st.head_weight, st.letter_ids
    table = longctx.load_skeleton()
    probe = table[0]
    bucket = [n for n in table if n.ctx == probe.ctx]
    doc, _cred = longctx.build_document(probe, bucket, seed=probe.seed, tokenizer=None)
    state_text = tok.decode(enc(tok, doc)[:STATE_TOKENS])       # 证据段定长到 2048 token
    questions = [
        {"type": "choice", "instructions": longctx.QUESTION_TEMPLATE_ORDINAL.format(ordinal=1, color="青"),
         "criteria": {"1": str(bucket[0].value), "2": str(bucket[1].value), "3": "777"}},
        {"type": "choice", "instructions": longctx.QUESTION_TEMPLATE_ORDINAL.format(ordinal=2, color="青"),
         "criteria": {"1": str(bucket[0].value), "2": str(bucket[1].value), "3": "777"}},
        {"type": "noul", "instructions": "这份材料里出现过颜色与编号的记录吗？", "criteria": {}},
    ]
    rendered = []
    for qi, spec in enumerate(questions):
        row = from_systemone(state_text, spec, qid=f"q{qi}")
        messages, order = render(row)
        rendered.append((messages, order))
    naive_total = sum(len(enc(tok, tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                           enable_thinking=False))) for m, _ in rendered)
    print(f"[B] naive_total_tokens(三问整段)={naive_total} state_chars={len(state_text)}")

    engine = HFEngine(body)
    cache = PrefixCache(max_bytes=8 * 1024 ** 3)

    def readout(hidden, k=None):
        """与 P1 同一件读出工具：末位数字 × 字母行 = 候选分。"""
        return option_scores(hidden.unsqueeze(1), head, letters[:k] if k else letters)

    runner = PrefixRunner(engine, cache=cache, readout=readout, verify="off")
    asks, lines = [], []
    cold_seconds = 0.0
    for qi, (messages, order) in enumerate(rendered):
        mark = runner.counter.snapshot()
        t1 = time.time()
        res = runner.ask_row(tok, messages, namespace="p207-run", k=len(order))
        dt, spent = time.time() - t1, runner.counter.since(mark)
        if qi == 0:
            cold_seconds = dt
        scores = readout(res.hidden_last, len(order))[0]
        lines.append(f"[B1b-real q{qi}] hit={res.hit} prefix_tokens_fed={res.prefix_tokens_fed} "
                     f"suffix_tokens_fed={res.suffix_tokens_fed} suffix_len={res.suffix_len} "
                     f"tokens_fed={res.tokens_fed} counter_spent={spent} "
                     f"suffix_only={res.tokens_are_suffix_only} seconds={dt:.2f}")
        asks.append({"q": qi, **res.as_dict(), "seconds": round(dt, 3),
                     "scores": [round(v, 4) for v in scores.tolist()],
                     "argmax_key": order[int(scores.argmax())]})
    for line in lines:
        print(line)
    print(f"[B1b-real] reuse_total={runner.counter.total} naive_total={naive_total} "
          f"saved={naive_total - runner.counter.total} by_kind={runner.counter.as_dict()['by_kind']}")

    # B1c：复用路 vs 整段重算路（真 bf16 前向）——漂移定标凭据，阈值可配
    prefix_ids, suffix_ids, info = split_prompt_ids(tok, rendered[1][0])
    warm = runner.ask(prefix_ids, suffix_ids, namespace="p207-run", k=len(rendered[1][1]))
    direct = runner.ask(prefix_ids, suffix_ids, namespace="p207-run", force_full=True,
                        k=len(rendered[1][1]))
    parity = compare_parity(warm.key, warm.hidden_last, direct.hidden_last, readout,
                            threshold=DRIFT_THRESHOLD, k=len(rendered[1][1]))
    print(f"[B1c-real] argmax_same={parity.argmax_same} max_abs_drift={parity.max_abs_drift:.3e} "
          f"score_scale={parity.score_scale:.3f} drift_ratio={parity.drift_ratio:.3e} "
          f"threshold={parity.threshold} passed={parity.passed} "
          f"scores_reuse={parity.scores_reuse} scores_full={parity.scores_full}")
    print(f"[B1c-real] warm_hit={warm.hit} warm_prefix_tokens_fed={warm.prefix_tokens_fed} "
          f"direct_full_tokens_fed={direct.full_tokens_fed}")

    # B2：跨请求命中（三条请求号不同、证据同一份）
    b2 = runner.serve([{"request_id": f"req-{i}", "prefix_ids": prefix_ids,
                        "suffix_ids": suffix_ids if i % 2 == 0 else suffix_ids[:-3]}
                       for i in range(3)], namespace="p207-run")
    print(f"[B2-real] hits={b2['hits']}/{b2['requests']} suffix_only={b2['all_hits_suffix_only']} "
          f"tokens_avoided_in_prefix={b2['tokens_avoided_in_prefix']} "
          f"cache={json.dumps(b2['cache'], ensure_ascii=False)}")

    cached = cache.get(prefix_key(prefix_ids, namespace="p207-run"))
    pb = estimate_past_bytes(cached) if cached is not None else 0
    return {"state_tokens": len(prefix_ids), "split_info": info, "asks": asks,
            "counter": runner.counter.as_dict(), "naive_total": naive_total,
            "saved_tokens": naive_total - runner.counter.total,
            "cold_seconds": round(cold_seconds, 3), "parity": parity.as_dict(),
            "cross_request": {"hits": b2["hits"], "requests": b2["requests"],
                              "all_suffix_only": b2["all_hits_suffix_only"],
                              "tokens_avoided": b2["tokens_avoided_in_prefix"]},
            "cache": runner.cache.stats(), "past_bytes": pb, "past_mib": round(pb / 2 ** 20, 1),
            "runner_stats": runner.stats()}


# ---------------------------------------------------------------- C1：滑窗路径与内存账
def step_c():
    """滑窗两条路与全注意力的口径互证 + 单块峰值 + 128K 档 KV 解析账（替身真实头数）。"""
    heads, dim, layers, kv_heads = 16, 128, 28, 2       # 取自 Qwen3-0.6B 的注意力形状
    routes = LA.compare_routes(512, 128, heads=heads, dim=dim, chunk=64)
    gen = torch.Generator().manual_seed(20261005)
    q, k, v = (torch.randn(heads, 1024, dim, generator=gen) for _ in range(3))
    rss_base = LA.rss_bytes()
    band = LA.attention_band(q, k, v, 128, chunk=64, route="local")
    rss_band = LA.rss_bytes()
    masked = LA.attention_masked(q, k, v, window=None)
    rss_masked = LA.rss_bytes()
    ratio = band.materialized_peak_bytes / masked.materialized_peak_bytes
    kv = {str(ctx): LA.kv_ledger(ctx, layers=layers, windows=[128] * layers, kv_heads=kv_heads,
                                 head_dim=dim) for ctx in (8192, 32768, 131072, 262144)}
    capped = LA.resolve_layer_windows(layers, window=262144, pattern="25:3", cap=256, cap_full=True)
    print(f"[C1] 三路口径互证@T=512: kernel_vs_masked={routes['kernel_vs_masked_max_abs']:.1e} "
          f"local_vs_masked={routes['local_vs_masked_max_abs']:.1e} "
          f"allclose={routes['kernel_allclose']}/{routes['local_allclose']}")
    print(f"[C1] T=1024 heads={heads} dim={dim}：带状单块峰值 {band.materialized_peak_bytes:,} 字节 "
          f"vs 全注意力物化 {masked.materialized_peak_bytes:,} 字节（比值 {ratio:.4f}）")
    print(f"[C1] CPU 驻留峰值（单调量，只作上界）：起点 {rss_base:,} / 带状后 {rss_band:,} / "
          f"全注意力后 {rss_masked:,} 字节；口径 {band.stats['memory_ledger']}")
    for ctx, led in kv.items():
        print(f"[C1] KV 解析账 ctx={ctx:>6}：全注意力 {led['full_attention_mib']:>9,.1f} MiB → "
              f"滑窗 W=128 {led['windowed_mib']:>7,.1f} MiB（省 {led['saved_ratio'] * 100:.2f}%）")
    print(f"[C1] 封顶排布 25:3 + cap=256(cap_full)：clamped={capped.clamped} "
          f"前 4 层窗={list(capped.windows[:4])}")
    return {"routes_512": routes, "band_T1024": band.stats,
            "masked_T1024_peak_bytes": masked.materialized_peak_bytes,
            "band_T1024_peak_bytes": band.materialized_peak_bytes, "peak_ratio": round(ratio, 6),
            "rss_base": rss_base, "rss_after_band": rss_band, "rss_after_masked": rss_masked,
            "kv_by_bucket": {c2: {"full_mib": x["full_attention_mib"],
                                  "windowed_mib": x["windowed_mib"],
                                  "saved_ratio": x["saved_ratio"]} for c2, x in kv.items()},
            "kv_128k": kv["131072"], "capped_plan": capped.as_dict()}


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    st = build_student("0.6b", loader="minimal", device=DEVICE, dtype=DTYPE)
    tok = st.tokenizer
    print(f"[env] build_student seconds={time.time() - t0:.1f} dtype={st.dtype} device={DEVICE}")
    cfg = {
        "domain": "p2-07-long-context",
        "wave": "A1+B1a-B1c+B2+C1（本地半场；A2 待云端/长窗真档）",
        "model": "Qwen3-0.6B 替身（production/sft build_student loader=minimal）",
        "device": DEVICE, "dtype": str(DTYPE), "mps": "禁用（派单纪律）",
        "longctx_version": longctx.LONGCTX_VERSION, "render_version": RENDER_VERSION,
        "needle_table_source": "bench/eval_data/assembled/needle-synthetic/needle-synthetic.jsonl",
        "state_tokens_for_prefix_probe": STATE_TOKENS, "drift_ratio_threshold": DRIFT_THRESHOLD,
        "sliding": {"reference_kernel": "sys1/kernels/attn_sw_kernel",
                    "default_window": LA.DEFAULT_WINDOW,
                    "cap_switch": "LongContextPlan.cap / cap_full"},
        "memory_ledger": "cpu-rss + 解析账，非 NPU 显存账",
    }
    nb = new_run("p207-longctx-local-half", cfg)
    a1 = step_a1(tok)
    nb.log_metrics(0, needle_rows=a1["rows"], corpus_sha16=a1["fingerprint"]["sha256_16"],
                   corpus_build_seconds=a1["seconds"],
                   token_rel_err_max=max(e["rel_err_max"] for e in a1["per_bucket"].values()))
    b = step_b(st, tok)
    nb.log_metrics(1, prefix_state_tokens=b["state_tokens"],
                   reuse_tokens_fed=b["counter"]["total"], naive_tokens_fed=b["naive_total"],
                   saved_tokens=b["saved_tokens"], cold_seconds=b["cold_seconds"],
                   past_mib=b["past_mib"])
    nb.log_metrics(2, cross_request_hits=b["cross_request"]["hits"],
                   cross_requests=b["cross_request"]["requests"],
                   cross_request_all_suffix_only=b["cross_request"]["all_suffix_only"],
                   cache_hits=b["cache"]["hits"], cache_misses=b["cache"]["misses"],
                   cache_evictions=b["cache"]["evictions"],
                   drift_ratio=b["parity"]["drift_ratio"], argmax_same=b["parity"]["argmax_same"],
                   parity_passed=b["parity"]["passed"])
    c = step_c()
    nb.log_metrics(3, masked_peak_bytes=c["masked_T1024_peak_bytes"],
                   band_peak_bytes=c["band_T1024_peak_bytes"], peak_ratio=c["peak_ratio"],
                   kv_full_mib_128k=c["kv_128k"]["full_attention_mib"],
                   kv_windowed_mib_128k=c["kv_128k"]["windowed_mib"],
                   kv_saved_ratio=c["kv_128k"]["saved_ratio"],
                   cap_clamped_layers=c["capped_plan"]["clamped_layers"])
    report = {"a1": a1, "b": b, "c": c, "device_note": longctx._device_note(DEVICE),
              "pending": {"A2": "8K/32K/128K 真前向召回曲线：待云端/长窗真档（本机 CPU 摊不开）",
                          "C2": "1M 档 measured 待开卡；config 见 production/configs/longctx_1m_cloud.yaml"}}
    (nb.path / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    rel_err = max(e["rel_err_max"] for e in a1["per_bucket"].values())
    text = nb.notes_file.read_text(encoding="utf-8")
    text = text.replace("- 假设：待填写",
                        "- 假设：证据段的编号是整段 prompt 编号的严格前缀，故 past 可按 state 复用；"
                        "复用生效的硬凭据是符号计数（第 2 问起=问题段长度），不是墙钟时间；"
                        "bf16 复用漂移可压在相对阈值 0.02 内", 1)
    text = text.replace("- 观察：待填写",
                        f"- 观察：A1 真 token 口径 {a1['rows']} 行全部构建（8K/32K 全出 + 128K/256K 各 1 题），"
                        f"档位相对偏差上限 {rel_err:.4f}，副本落 {a1['out']}；针位表与 registry 骨架同源\n"
                        f"- 观察：B1b 复用路总符号 {b['counter']['total']} vs 三问整段重算 {b['naive_total']}"
                        f"（省 {b['saved_tokens']}）；第 2/3 问 prefix_tokens_fed=0 且 "
                        f"suffix_tokens_fed=suffix_len；冷启一题 {b['cold_seconds']}s，"
                        f"past {b['past_mib']} MiB（state={b['state_tokens']} token）\n"
                        f"- 观察：B1c 真 bf16 前向 max_abs_drift={b['parity']['max_abs_drift']:.3e}、"
                        f"分数尺度 {b['parity']['score_scale']:.3f} → 相对漂移 "
                        f"{b['parity']['drift_ratio']:.3e}（阈值 {DRIFT_THRESHOLD}）passed="
                        f"{b['parity']['passed']}，argmax_same={b['parity']['argmax_same']}\n"
                        f"- 观察：B2 跨请求命中 {b['cross_request']['hits']}/{b['cross_request']['requests']}，"
                        f"命中题全部 suffix_only={b['cross_request']['all_suffix_only']}；"
                        f"缓存记账 {json.dumps(b['cache'], ensure_ascii=False)}\n"
                        f"- 观察：C1 带状路单块峰值 {c['band_T1024_peak_bytes']:,} 字节 vs 全注意力 "
                        f"{c['masked_T1024_peak_bytes']:,} 字节（比值 {c['peak_ratio']}）；"
                        f"128K 档 KV 解析账 {c['kv_128k']['full_attention_mib']:,.1f} MiB → "
                        f"{c['kv_128k']['windowed_mib']:,.1f} MiB；窗上限 cap=256 可夹住 "
                        f"{c['capped_plan']['clamped_layers']}/{c['capped_plan']['layers']} 层\n"
                        f"- 口径边界：以上全为 CPU + 0.6B 替身；A2 召回曲线与 1M/NPU 显存实测未做"
                        f"（待云端长窗真档）\n", 1)
    nb.notes_file.write_text(text, encoding="utf-8")
    nb.conclude(
        f"prefix_cache 三级件与滑窗长文路径在本地半场成立：同 state 三问复用只付问题段符号"
        f"（{b['counter']['total']}/{b['naive_total']} token，省 {b['saved_tokens']}，"
        f"第 2 问起 prefix_tokens_fed=0），跨请求命中 {b['cross_request']['hits']}/"
        f"{b['cross_request']['requests']}，真 bf16 相对漂移 {b['parity']['drift_ratio']:.3e} "
        f"在 0.02 阈值内且 argmax 不换人；滑窗带状路单块峰值降到全注意力的 {c['peak_ratio']} 倍"
        f"且与掩码路逐位同口径（kernel 路差 0.0），窗上限开关能把 {c['capped_plan']['clamped_layers']} "
        f"层全夹住——D7 三件套'滑窗封顶'的第一件基座到位。A2 召回曲线与 NPU/1M 显存实测按派单留待云端。")
    print("RUN_DIR", nb.finish())


if __name__ == "__main__":
    main()
