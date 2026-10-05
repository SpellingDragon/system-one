"""p2-01 布局对拍探针（合成小模型 + 临时 safetensors，不碰 1.7G 检查点、不上 MPS）。

跑法：cd release && .venv/bin/python .probe_p201_layout.py
用途：先把"换算 + 一层前向对拍"这条数学在微型身板上验通，再上真检查点。
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from production.backbone import loader as LD  # noqa: E402


def build_case():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding

    cfg = Qwen3_5TextConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        rms_norm_eps=1e-6, attention_bias=False, partial_rotary_factor=0.25,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(7)
    attn = Qwen3_5Attention(cfg, 0).float().eval()
    return cfg, attn


class FakeStack(torch.nn.Module):
    def __init__(self, cfg, attn):
        super().__init__()
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
        lay = torch.nn.Module()
        lay.self_attn = attn
        self.layers = torch.nn.ModuleList([lay])
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg)
        self.config = cfg


def dump(attn, tmp: Path) -> None:
    pre = "model.layers.0.self_attn."
    tensors = {pre + k: v.detach().clone() for k, v in attn.state_dict().items()}
    shard = "model.safetensors"
    save_file(tensors, str(tmp / shard))
    wmap = {k: shard for k in tensors}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wmap, "metadata": {}}))
    (tmp / "config.json").write_text("{}")


def main() -> int:
    cfg, attn = build_case()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        dump(attn, tmp)
        rep = LD.verify_attention_layer(tmp, 0, FakeStack(cfg, attn), text_config=cfg, seq=12, tol=1e-5)
        print("layout:", rep["layout"], "keys:", rep["keys"])
        print("row_match ok:", rep["row_match"]["ok"], "mismatch:", rep["row_match"]["mismatch"][:3])
        print("note:", rep["note"])
        print("notes:", rep["notes"])
        print("VERIFY_OK" if rep["ok"] else "VERIFY_FAIL")
        return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
