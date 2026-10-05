"""p2-05 探针：分段记账开关在真 HF 底座（Qwen3-0.6B）上到底有没有按下。

复跑：cd release && .venv/bin/python .probe_p205_gc.py
只看三件事：① `enable_gradient_checkpointing()` 交回 True 时底座旗标真变了没有；
② 开着它能不能前向 + 反向下笔（HF 检查点要求首层输入留钩子，漏了会静默不接梯度）；
③ 多少枚垫片真收到了改动（一条都没有 = 记账链断了）。run 侧凭据见 runs/1005-p2-05-dev-gc-*。
"""
from __future__ import annotations

import time

import torch

from production import sft

cfg = sft.load_config("production/configs/tiny_cpu.yaml")
torch.set_num_threads(int(cfg["threads"]))
st = sft.build_student(cfg["backbone"], loader=cfg["loader"], device=cfg["device"],
                       dtype=cfg["dtype"], cache_dir=cfg.get("cache_dir"))
print("body 类:", type(st.body).__name__)
print("有 enable 钩子?", hasattr(st.body, "gradient_checkpointing_enable"),
      hasattr(st.body, "enable_input_require_grads"))
print("按下前旗标:", getattr(st.body, "gradient_checkpointing", "<无此属性>"))
print("交回:", sft.enable_gradient_checkpointing(st, True),
      " 按下后旗标:", getattr(st.body, "gradient_checkpointing", "<无此属性>"))

loras = sft.inject_lora(st.body, r=int(cfg["r"]), alpha=int(cfg["alpha"]),
                        dropout=float(cfg["dropout"]), backend=sft.BACKEND_KERNEL,
                        ascend_target=cfg["ascend_target"])
print("注入:", sft.freeze_all_but_lora(st.model, loras))
samples, _ = sft.assemble(sft.load_quality_records("quality"), st.tokenizer, cache=None,
                          limit=8, max_length=int(cfg["max_length"]))
idx = sft.bucket_batches(samples, max_tokens=512, shuffle=False, seed=7)[0][:2]
batch = sft.collate(samples, idx, pad_id=st.pad_id, letter_map=st.letter_ids)

started = time.time()
loss = sft.readout_ce_loss(sft.student_forward(st, batch), st.head_weight, batch.letter_ids,
                           batch.target, keep=batch.keep, lengths=batch.lengths)
loss.backward()
print(f"开着检查点前向+反向 loss={float(loss.detach()):.4f} 耗时 {time.time() - started:.2f}s")
got = sum(1 for m in loras.values() if m.lora_b.grad is not None)
print("收到改动的垫片数:", got, "/", len(loras))
assert got == len(loras), "开着检查点却有垫片没收到改动 —— 记账链断了"
print("PROBE OK")
