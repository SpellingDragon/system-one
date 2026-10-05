"""p2-10 前置探针：0.6B 替身能不能被服务层按 decision 口径驱动，一次前向多慢。

口径：CPU、禁 MPS；loader=minimal（p2-05 确立的替身侧门，正式门为 0.8B 多模态设计）。
"""
from __future__ import annotations

import time

import torch

from production.sft import build_student
from sys1.decision import LETTERS, from_systemone, option_scores, render

STATE = "weather: clear sky at 07:00, humidity 40%"
SPEC = {"type": "choice", "instructions": "Which option fits the evidence?",
        "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"}}

def tok_ids(t, text):
    out = t.encode(text, add_special_tokens=False)
    if isinstance(out, dict):
        return list(out["input_ids"])
    if hasattr(out, "ids"):
        return list(out.ids)
    return list(out)


def main() -> None:
    t0 = time.time()
    st = build_student("0.6b", loader="minimal", device="cpu", dtype=torch.float32)
    print(f"[load] seconds={time.time() - t0:.1f} dtype={st.dtype} letters={len(st.letter_ids)} "
          f"pad_id={st.pad_id} head={tuple(st.head_weight.shape)} summary={st.summary()}")

    row = from_systemone(STATE, SPEC, qid="q1")
    messages, order = render(row)
    print(f"[render] order={order} messages_roles={[m['role'] for m in messages]}")
    try:
        text = st.tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        how = "enable_thinking kwarg ok"
    except TypeError as exc:
        text = st.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        how = f"enable_thinking rejected ({exc})"
    from sys1.decision.render import THINK_OFF_SUFFIX
    print(f"[template] {how}\n  tail={text[-40:]!r}\n  think_off_suffix={THINK_OFF_SUFFIX!r}\n"
          f"  endswith={text.endswith(THINK_OFF_SUFFIX)}")
    ids = tok_ids(st.tokenizer, text)
    print(f"[encode] tokens={len(ids)}")

    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        t1 = time.time()
        hidden = st.body(input_ids=inp, attention_mask=torch.ones_like(inp)).last_hidden_state
        f1 = (time.time() - t1) * 1000.0
        t2 = time.time()
        sc = option_scores(hidden, st.head_weight, list(st.letter_ids[:len(order)]), lengths=[len(ids)])
        f2 = (time.time() - t2) * 1000.0
    print(f"[forward] ms={f1:.1f} readout_ms={f2:.3f} shape={tuple(hidden.shape)} "
          f"scores={[round(float(v), 3) for v in sc[0]]} argmax={LETTERS[int(sc.argmax())]}")

    # 合批可行性：三条不同 k 的行右补空洞后一次前向，与逐条单行是否同值
    specs = [
        ("q1", {"type": "choice", "instructions": "Which option fits the evidence?",
                "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"}}),
        ("q2", {"type": "noul", "instructions": "Is it fine?", "criteria": {"false": "no", "true": "yes"}}),
        ("q3", {"type": "score", "instructions": "How good?",
                "criteria": {"1": "bad", "2": "mid", "3": "good"}}),
    ]
    rows = []
    for qid, spec in specs:
        r = from_systemone(STATE, spec, qid=qid)
        _, o = render(r)
        t = st.tokenizer.apply_chat_template(render(r)[0], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        rows.append((qid, r, o, tok_ids(st.tokenizer, t)))
    print("[multi-qid] " + ", ".join(f"{qid}:k={len(o)}/tok={len(i)}" for qid, _r, o, i in rows))
    width = max(len(i) for _q, _r, _o, i in rows)
    ids_b = torch.full((len(rows), width), st.pad_id, dtype=torch.long)
    mask_b = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (_q, _r, _o, ids) in enumerate(rows):
        ids_b[i, :len(ids)] = torch.tensor(ids)
        mask_b[i, :len(ids)] = 1
    with torch.no_grad():
        hb = st.body(input_ids=ids_b, attention_mask=mask_b).last_hidden_state
    for i, (qid, _r, o, ids) in enumerate(rows):
        sb = option_scores(hb[i:i + 1], st.head_weight, list(st.letter_ids[:len(o)]), lengths=[len(ids)])
        s1 = option_scores(
            st.body(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids)))
                .last_hidden_state,
            st.head_weight, list(st.letter_ids[:len(o)]), lengths=[len(ids)])
        d = float((sb - s1).abs().max())
        scale = max(float(s1.abs().mean()), 1e-6)
        print(f"[pad-invariance] {qid} max_abs={d:.3e} rel={d / scale:.3e} "
              f"argmax_same={bool(int(sb.argmax()) == int(s1.argmax()))}")


if __name__ == "__main__":
    main()
