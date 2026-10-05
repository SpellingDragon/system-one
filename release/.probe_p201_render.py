"""p2-01 探针（临时）：Qwen3.5-0.8B tokenizer 与 P1 decision/render 的接缝实测。

只读引用 sys1/decision/，不改它。跑法：release/.venv/bin/python .probe_p201_render.py
"""
from __future__ import annotations

import importlib
import string
import sys

sys.path.insert(0, ".")

from transformers import AutoTokenizer  # noqa: E402

R = importlib.import_module("sys1.decision.render")

PATH = "bench/ms_models/models/Qwen--Qwen3.5-0.8B/snapshots/master"
IMG = "\u003c|image_pad|>\u003c|\u003e"


def enc(tok, text):
    out = tok.encode(text, add_special_tokens=False)
    return out["input_ids"] if isinstance(out, dict) else list(out)


def main():
    tok = AutoTokenizer.from_pretrained(PATH)
    print("tokenizer cls:", type(tok).__name__, "vocab", tok.vocab_size, "len", len(tok))
    print("RENDER_VERSION", R.RENDER_VERSION, "THINK_OFF", repr(R.THINK_OFF_SUFFIX))

    row = R.from_systemone(
        "CPU 0% for 3s",
        {"type": "choice", "instructions": "pick", "criteria": {"A": "sunny", "B": "rain"}},
    )
    msgs, order = R.render(row)
    for kw in ({}, {"enable_thinking": False}, {"enable_thinking": True}):
        try:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            print(kw, "| endswith THINK_OFF:", t.endswith(R.THINK_OFF_SUFFIX), "| tail", repr(t[-28:]))
        except Exception as exc:  # noqa: BLE001
            print(kw, "| ERR", type(exc).__name__, str(exc)[:180])

    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    base = enc(tok, t)
    print("prompt tokens:", len(base), "| last 3:", base[-3:])

    letters = {c: enc(tok, c) for c in string.ascii_uppercase}
    bad_single = {c: v for c, v in letters.items() if len(v) != 1}
    bad_bound = {}
    for c, v in letters.items():
        if len(v) != 1:
            continue
        after = enc(tok, t + c)
        if after != base + [v[0]]:
            bad_bound[c] = after[len(base):]
    print("letters not single-token:", bad_single)
    print("letters boundary-merge failures:", bad_bound)
    print("A id:", letters["A"][0], "Z id:", letters["Z"][0])

    specials = ["\u003c|image_pad|>", "\u003c|video_pad|>", "\u003c|vision_start|>", "\u003c|vision_end|>", "\u003cthink\u003e", "\u003c/think\u003e"]
    for s in specials:
        print("special", repr(s), "id", tok.convert_tokens_to_ids(s))


if __name__ == "__main__":
    main()
