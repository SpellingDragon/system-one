"""p2-05 探针：0.8B 走 load_backbone 全接缝能否在 CPU 载入 + 注意力模块名清点。"""
import sys
import time

import torch

sys.path.insert(0, ".")

t0 = time.time()
from production.assets import load_backbone  # noqa: E402

try:
    bb = load_backbone("qwen3.5-0.8b", device="cpu", dtype=torch.float32)
    print("load_backbone OK seconds", round(time.time() - t0, 1))
except Exception as exc:  # noqa: BLE001
    print("load_backbone FAIL", type(exc).__name__, str(exc)[:300])
    sys.exit(1)

print("body type", type(bb.body).__name__)
print("hidden", bb.hidden_size, "head", tuple(bb.model.get_output_embeddings().weight.shape))
proj = {}
for name, mod in bb.body.named_modules():
    if isinstance(mod, torch.nn.Linear):
        leaf = name.split(".")[-1]
        proj.setdefault(leaf, []).append(name)
for leaf, names in sorted(proj.items()):
    print(f"linear leaf={leaf} count={len(names)} example={names[0]}")
cfg = bb.model.config.get_text_config()
print("layer_types", getattr(cfg, "layer_types", None))
print("num_layers", getattr(cfg, "num_hidden_layers", None), "heads", getattr(cfg, "num_attention_heads", None),
      "kv", getattr(cfg, "num_key_value_heads", None))
