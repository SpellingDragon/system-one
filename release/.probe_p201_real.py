"""p2-01 真路径探针：真载 Qwen3.5-0.8B（CPU/fp16）后一次跑完四道接缝、布局对拍、换脑冒烟、因果不变性与峰值内存。

跑法：cd release && .venv/bin/python .probe_p201_real.py
约束：只用 CPU（MPS 被一阶段训练占用）；不改 sys1/decision/；不进任何生成循环。
"""
from __future__ import annotations
import json, resource, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from production.assets import load_backbone              # noqa: E402
from production.backbone import loader as LD             # noqa: E402
from sys1.decision import LETTERS, from_systemone, readout, render  # noqa: E402

QUESTIONS = [
    ("q1", "weather: clear sky at 07:00, humidity 40%",
     {"type": "choice", "instructions": "Which option fits the evidence?",
      "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"}}),
    ("q2", "server status: cpu 12%, memory 30%, disk 45%",
     {"type": "choice", "instructions": "Is the server healthy?",
      "criteria": {"a": "healthy", "b": "degraded", "c": "down"}}),
    ("q3", "review: the code passes all tests and has no lint errors",
     {"type": "choice", "instructions": "What is the review verdict?",
      "criteria": {"a": "approve", "b": "request changes", "c": "comment only", "d": "reject"}}),
]


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def swap_brain(bb) -> list[dict]:
    out = []
    for qid, state, spec in QUESTIONS:
        row = from_systemone(state, spec, qid=qid)
        ids, order = bb.encode_prompt(row)
        t = torch.tensor([ids], dtype=torch.long)
        hidden = bb.forward(t)
        ro = readout(hidden, bb.model.get_output_embeddings().weight.float(),
                     list(bb.letter_ids[:len(order)]), lengths=[len(ids)], qtypes=[row["type"]])
        pick = int(ro.probs[0].argmax())
        out.append({"qid": qid, "tokens": len(ids), "order": order,
                    "letter": LETTERS[pick], "code": order[pick],
                    "probs": [round(v, 4) for v in ro.probs[0].tolist()],
                    "temperatures": [round(v, 4) for v in ro.temperatures]})
    return out


def causal_invariance(bb) -> dict:
    """右补齐前向对照：同一行内容，垫与不垫两种送法，读点位上的候选分必须落在容差内。"""
    rows, orders = [], []
    for qid, state, spec in QUESTIONS:
        r = from_systemone(state, spec, qid=qid)
        ids, order = bb.encode_prompt(r)
        rows.append(ids); orders.append(order)
    pad = bb.tokenizer.pad_token_id if bb.tokenizer.pad_token_id is not None else 0
    width = max(len(r) for r in rows)
    ids_b = torch.full((len(rows), width), int(pad), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids_b[i, :len(r)] = torch.tensor(r); mask[i, :len(r)] = 1
    lengths = [len(r) for r in rows]
    head = bb.model.get_output_embeddings().weight.float()
    hid_pad = bb.forward(ids_b, attn_mask=mask)
    sc_pad = readout(hid_pad, head, list(bb.letter_ids[:3]), lengths=lengths).scores
    single = []
    for i, r in enumerate(rows):
        hid = bb.forward(torch.tensor([r], dtype=torch.long))
        single.append(readout(hid, head, list(bb.letter_ids[:3]), lengths=[len(r)]).scores[0])
    sc_one = torch.stack(single)
    n = min(len(o) for o in orders)
    d = (sc_pad[:, :n] - sc_one[:, :n]).abs()
    args_pad = [LETTERS[int(v)] for v in sc_pad[:, :n].argmax(dim=1)]
    args_one = [LETTERS[int(v)] for v in sc_one[:, :n].argmax(dim=1)]
    return {"width": width, "lengths": lengths, "max_abs_diff": float(d.max()),
            "mean_abs_score": float(sc_one.abs().mean()), "argmax_pad": args_pad,
            "argmax_single": args_one, "same_argmax": args_pad == args_one}


def main() -> int:
    import transformers
    t0 = time.time()
    bb = load_backbone("qwen3.5-0.8b", device="cpu", dtype=torch.float16)
    load_seconds = time.time() - t0
    cfg = transformers.AutoConfig.from_pretrained(str(bb.snapshot.path)).get_text_config()
    full = LD.full_attention_layers(cfg)
    ver = [LD.verify_attention_layer(bb.snapshot.path, li, bb.body, text_config=cfg,
                                     seq=8, tol=1e-3, sample_rows=512) for li in full[:2]]
    rep = {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "load_seconds": round(load_seconds, 1), "peak_rss_mb_at_load": round(peak_mb(), 1),
        "model_param_bytes_mb": round(sum(x.numel() * x.element_size() for x in bb.model.parameters()) / 1e6, 1),
        "seam": bb.provenance_config(), "hidden_size": bb.hidden_size,
        "letter_ids": list(bb.letter_ids), "letter_rows_shape": list(bb.letter_rows.shape),
        "full_attention_layers": full,
        "layout": [{"layer": v["layer_idx"], "ok": v["ok"], "note": v["parity"].as_note(),
                    "row_match": v["row_match"]["ok"], "keys": v["keys"],
                    "layout": v["layout"], "mismatch": v["row_match"]["mismatch"]} for v in ver],
        "swap_brain": swap_brain(bb), "causal_invariance": causal_invariance(bb),
        "peak_rss_mb_end": round(peak_mb(), 1),
    }
    print(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    return 0 if all(v["ok"] for v in ver) and rep["causal_invariance"]["same_argmax"] else 1


if __name__ == "__main__":
    sys.exit(main())
