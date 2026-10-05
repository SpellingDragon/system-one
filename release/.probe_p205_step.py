"""p2-05 探针：数据装配（registry 口 + 教师缓存命中）与单步训练耗时实测。"""
import json
import sys
import time

import torch

sys.path.insert(0, ".")

from production.assets import load_backbone  # noqa: E402
from production.teachers.cache import DistCache, make_key  # noqa: E402
from production.teachers.verify_b3 import SCAFFOLD_MODEL_ID  # noqa: E402
from sys1.decision.render import (  # noqa: E402
    LETTERS, from_systemone, option_order, prompt_text, render as render_messages)
render = None
from sys1.decision.readout import option_scores  # noqa: E402
from sys1.eval import run as evalrun  # noqa: E402


recs = evalrun.load_axis_records("quality")
print("records", len(recs))

cache = DistCache(root="bench/teacher_cache/throughput_bench").load()
print("cache entries", len(cache.index))

hits = 0
rows = []
for r in recs[:400]:
    sample = r["sample"]
    row = from_systemone(sample["state"], sample["questions"]["q"])
    order = option_order(row)
    if not order or len(order) > len(LETTERS):
        continue
    prompt = prompt_text(row, order)
    keys = list(LETTERS[:len(order)])
    k = make_key(SCAFFOLD_MODEL_ID, prompt, keys)
    dist = cache.get(k)
    if dist:
        hits += 1
    rows.append((r["id"], order, dist is not None))
print("first400 scaffold-cache hits", hits)

bb = load_backbone("qwen3.5-0.8b", device="cpu", dtype=torch.float32)
tok = bb.tokenizer
t0 = time.time()
enc = []
for r in recs[:200]:
    sample = r["sample"]
    row = from_systemone(sample["state"], sample["questions"]["q"])
    messages, order = render_messages(row)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    enc.append(len(ids))
print("tokenize 200 rows seconds", round(time.time() - t0, 2), "median len", sorted(enc)[len(enc) // 2],
      "max", max(enc))

# ── 单步训练耗时（冻结底座 + 只训 4 个方阵的 LoRA 旁路，先手工量一量）
body = bb.body
for p in body.parameters():
    p.requires_grad_(False)
head = bb.model.get_output_embeddings().weight
letter_ids = list(bb.letter_ids[:4])
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

sample_ids = [tok(r["sample"]["state"] and "", add_special_tokens=False)["input_ids"] for r in recs[:1]]
tids = []
for r in recs[:8]:
    row = from_systemone(r["sample"]["state"], r["sample"]["questions"]["q"])
    messages, order = render_messages(row)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tids.append(tok(text, add_special_tokens=False)["input_ids"])
width = max(len(t) for t in tids)
ids = torch.full((8, width), pad, dtype=torch.long)
mask = torch.zeros_like(ids)
for i, t in enumerate(tids):
    ids[i, :len(t)] = torch.tensor(t)
    mask[i, :len(t)] = 1
print("batch shape", tuple(ids.shape))

lin = body.layers[3].self_attn.q_proj
lin.weight.requires_grad_(True)   # 计时用:让图可微(真实训练由 LoRA 旁路提供可微叶子)
lora_a = torch.nn.Parameter(torch.randn(16, lin.in_features) * 0.01)
lora_b = torch.nn.Parameter(torch.zeros(16, lin.out_features))
lora_a.requires_grad_(True)
lora_b.requires_grad_(True)

for tag, use_ckpt in (("no_checkpointing", False), ("checkpointing", True)):
    if use_ckpt:
        body.gradient_checkpointing_enable()
        body.enable_input_require_grads()
    t0 = time.time()
    with torch.autograd.grad_mode.set_grad_enabled(True):
        out = body(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        lengths = [int(x) for x in mask.sum(1)]
        scores = option_scores(out, head.detach(), letter_ids, lengths)
        tgt = torch.softmax(torch.randn(scores.shape), dim=-1)
        loss = -(tgt * torch.log_softmax(scores.float(), -1)).sum(-1).mean()
        loss.backward()
    print(f"step[{tag}] seconds", round(time.time() - t0, 2), "loss", round(float(loss), 4),
          "grad_norm", round(float(lora_a.grad.norm()), 4), "params_with_grad",
          sum(1 for p in body.parameters() if p.grad is not None))
    body.zero_grad(set_to_none=True)
    if use_ckpt:
        body.gradient_checkpointing_disable()
print(json.dumps({"hits_of_first400": hits}))
