"""p2-07 探针：Qwen3-0.6B（替身）在 CPU 上能不能做"前段缓存 + 增量前向"，三件事逐条取证。

复跑：cd release && .venv/bin/python .probe_p207_prefix.py
只看三件事：
① 渲染分段：只编"证据段"得到的编号，是不是整段 prompt 编号的**严格前缀**（否则缓存位对不上）；
② past_kv 复用：前段一次前向存下的 past_key_values，接上"只送问题段"能不能得到末位数字；
③ 数值漂移：复用路与整段重算路的候选分差多少、argmax 换不换人（B1c 阈值的定标凭据）。
"""
from __future__ import annotations

import time

import torch

from production.sft import build_student
from sys1.decision import from_systemone, option_scores, render

SPLIT_MARK = "\n\nQuestion: "


def enc(tok, text):
    out = tok.encode(text, add_special_tokens=False)
    return list(out["input_ids"]) if isinstance(out, dict) else list(getattr(out, "ids", out))


def common_prefix_len(a, b) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def _past_bytes(past) -> int:
    """数一遍 past_key_values 里所有 keys/values 的实占字节（壳与角度表不算）。"""
    total = 0
    layers = getattr(past, "layers", None)
    if layers is None:
        layers = past if isinstance(past, (list, tuple)) else []
    for lay in layers:
        for name in ("keys", "values"):
            v = lay[name] if isinstance(lay, (list, tuple)) else getattr(lay, name, None)
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total


def main() -> None:
    t0 = time.time()
    st = build_student("0.6b", loader="minimal", device="cpu", dtype=torch.bfloat16)
    print(f"build_student seconds={time.time() - t0:.1f} dtype={st.dtype} letters={len(st.letter_ids)}")
    tok, body, head = st.tokenizer, st.body, st.head_weight

    state = "weather: " + ", ".join(
        f"day{i:03d} clear sky at 07:00, humidity {30 + i % 50}%" for i in range(40))
    questions = [
        {"type": "choice", "instructions": "Which option fits the evidence?",
         "criteria": {"a": "sunny", "b": "rain", "c": "cloudy"}},
        {"type": "noul", "instructions": "Is the sky clear?", "criteria": {}},
        {"type": "choice", "instructions": "Pick the humidity band.",
         "criteria": {"low": "30%", "mid": "55%", "high": "80%"}},
    ]

    for qi, spec in enumerate(questions):
        row = from_systemone(state, spec, qid=f"q{qi}")
        messages, order = render(row)
        user = messages[1]["content"]
        prefix_user = user.split(SPLIT_MARK, 1)[0]
        sys_part = tok.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=False,
                                           enable_thinking=False)
        pre_text = tok.apply_chat_template([messages[0], {"role": "user", "content": prefix_user}],
                                           tokenize=False, add_generation_prompt=False,
                                           enable_thinking=False)
        full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
        # 只含 state 的那一截：system 壳 + Evidence 段正文 + 用户壳的收尾
        state_shell = pre_text[:len(sys_part) + len(prefix_user)]
        ids_full = enc(tok, full_text)
        ids_pre = enc(tok, pre_text)
        ids_state_shell = enc(tok, state_shell)
        n_align = common_prefix_len(ids_pre, ids_full)
        print(f"[q{qi}] full={len(ids_full)} pre={len(ids_pre)} common={n_align} "
              f"pre_is_prefix={n_align == len(ids_pre)} "
              f"state_shell_prefix_ok={common_prefix_len(ids_state_shell, ids_full) == len(ids_state_shell)} "
              f"shell_len={len(ids_state_shell)} suffix_len={len(ids_full) - len(ids_state_shell)}")
        if qi != 0:
            continue

        n_prefix = len(ids_state_shell)
        with torch.no_grad():
            t1 = time.time()
            outs = body(input_ids=torch.tensor([ids_full]), use_cache=True, return_dict=True)
            print(f"full_forward seconds={time.time() - t1:.2f} past_type={type(outs.past_key_values).__name__}")
            hid_full = outs.last_hidden_state

            # ② 前段单独前向存下的 past（这才是"state 段只算一次"）
            outs_pre = body(input_ids=torch.tensor([ids_full[:n_prefix]]), use_cache=True,
                            return_dict=True)
            past = outs_pre.past_key_values
            pb = _past_bytes(past)
            print(f"past seq_len={past.get_seq_length()} bytes={pb:,} ({pb / 2**20:.1f} MiB)")

            suffix_ids = ids_full[n_prefix:]
            pos = torch.arange(n_prefix, n_prefix + len(suffix_ids)).unsqueeze(0)
            t2 = time.time()
            incr = body(input_ids=torch.tensor([suffix_ids]), past_key_values=past,
                        position_ids=pos, use_cache=False, return_dict=True)
            print(f"incr_forward seconds={time.time() - t2:.2f} suffix_tokens={len(suffix_ids)}")

            k = len(order)
            s_full = option_scores(hid_full[:, -1:, :], head, st.letter_ids[:k])[0]
            s_incr = option_scores(incr.last_hidden_state[:, -1:, :], head, st.letter_ids[:k])[0]
            drift = (s_full.float() - s_incr.float()).abs()
            print(f"scores_full={[round(v, 4) for v in s_full.tolist()]}")
            print(f"scores_incr={[round(v, 4) for v in s_incr.tolist()]}")
            print(f"max_abs_drift={drift.max().item():.3e} "
                  f"score_scale={s_full.float().abs().mean().item():.3f} "
                  f"argmax_same={int(s_full.argmax()) == int(s_incr.argmax())}")
            print(f"reuse past is mutated? seq_len_after={past.get_seq_length()}")


if __name__ == "__main__":
    main()
