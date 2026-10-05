"""p2-05 探针②：最小 HF 载入口 + CPU 单步训练耗时（0.6B 替身，三种精度/批规模）。"""
import statistics
import sys
import time

import torch

sys.path.insert(0, ".")

from production.assets import fetch_snapshot  # noqa: E402
from production.teachers.text import check_letters  # noqa: E402
from sys1.decision.readout import option_scores  # noqa: E402
from sys1.decision.render import from_systemone, render as render_messages  # noqa: E402
from sys1.eval import run as evalrun  # noqa: E402

import transformers  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "qwen3-0.6b"
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
    sys.argv[2] if len(sys.argv) > 2 else "fp32"]
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 4
REPEAT = int(sys.argv[4]) if len(sys.argv) > 4 else 3

t0 = time.time()
snap = fetch_snapshot(NAME, local_only=True) if "local_only" in fetch_snapshot.__code__.co_varnames \
    else fetch_snapshot(NAME)
print("snapshot", snap.path, snap.source, "seconds", round(time.time() - t0, 1))
tok = transformers.AutoTokenizer.from_pretrained(str(snap.path))
print("pad", tok.pad_token_id, "eos", tok.eos_token_id, "padding_side", getattr(tok, "padding_side", None))
cfg = transformers.AutoConfig.from_pretrained(str(snap.path))
cls = getattr(transformers, (cfg.architectures or ["AutoModelForCausalLM"])[0])
t0 = time.time()
model = cls.from_pretrained(str(snap.path), dtype=DTYPE)
print("load seconds", round(time.time() - t0, 1), "class", cls.__name__)
model.eval()
letters = check_letters(tok)
head = model.get_output_embeddings().weight
body = getattr(model.model, "language_model", model.model)
print("body", type(body).__name__, "hidden", head.shape)

recs = evalrun.load_axis_records("quality")
items = []
for r in recs:
    if len(items) >= 64:
        break
    row = from_systemone(r["sample"]["state"], r["sample"]["questions"]["q"])
    messages, order = render_messages(row)
    if not 2 <= len(order) <= 4:
        continue
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    targets = r["sample"]["targets"]["q"]
    vec = [float(targets.get(k, 0.0)) for k in order]
    s = sum(vec) or 1.0
    items.append((ids[:256], len(order), [v / s for v in vec]))
print("items", len(items), "median tokens", statistics.median(len(i[0]) for i in items))

for p in model.parameters():
    p.requires_grad_(False)
target_leaf = None
for n, m in body.named_modules():
    if n.endswith("q_proj"):
        m.weight.requires_grad_(True)
        target_leaf = n
        break
print("trainable leaf", target_leaf)

seconds = []
for trial in range(REPEAT):
    batch = items[trial * BATCH:(trial + 1) * BATCH]
    if len(batch) < BATCH:
        break
    cap = int(__import__("os").environ.get("CAP", "512"))
    batch = [(t[:cap], k, v) for t, k, v in batch]
    width = max(len(b[0]) for b in batch)
    ids = torch.full((BATCH, width), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    tgt = torch.zeros((BATCH, max(b[1] for b in batch)))
    keep = torch.zeros_like(tgt, dtype=torch.bool)
    for j, (t, k, vec) in enumerate(batch):
        ids[j, :len(t)] = torch.tensor(t)
        mask[j, :len(t)] = 1
        tgt[j, :k] = torch.tensor(vec)
        keep[j, :k] = True
    lengths = [int(x) for x in mask.sum(1)]
    k = int(tgt.shape[1])
    one = torch.tensor(letters[:k])
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    t0 = time.time()
    out = body(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
    scores = option_scores(out, head.detach().float() if head.dtype != torch.float32 else head, one.tolist(), lengths)
    scores = scores.masked_fill(~keep, float("-inf"))
    logp = torch.log_softmax(scores.float(), -1).masked_fill(~keep, 0.0)   # nan fix
    loss = -(tgt * logp).sum(-1).mean()
    loss.backward()
    seconds.append(time.time() - t0)
    model.zero_grad(set_to_none=True)
    print(f"trial{trial} shape=({BATCH},{width}) seconds {seconds[-1]:.2f} loss {float(loss.detach()):.4f}")
print("MEAN step seconds", round(sum(seconds) / max(1, len(seconds)), 2), "dtype", DTYPE, "batch", BATCH)
