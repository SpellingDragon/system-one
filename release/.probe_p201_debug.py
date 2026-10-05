"""逐步对比 HF 与自研栈两条路的中间量，定位布局换算的第一处偏差（合成微型身板，CPU）。"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from production.backbone import loader as LD, layout as L  # noqa: E402
spec = importlib.util.spec_from_file_location("pl", ".probe_p201_layout.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv  # noqa: E402


def rep(t, n):
    return t.repeat_interleave(n, dim=0)


def cmp(name, a, b):
    d = (a.float() - b.float()).abs().max().item()
    print(f"{name:22s} maxdiff={d:.3e} shapeA={tuple(a.shape)} shapeB={tuple(b.shape)}")


cfg, attn = mod.build_case()
lay = L.HeadLayout.from_config(cfg)
x = torch.randn(12, cfg.hidden_size, generator=torch.Generator().manual_seed(3))
pos = torch.arange(12).view(1, 1, -1).expand(3, 1, -1)
stack = mod.FakeStack(cfg, attn)
cos, sin = (t[0] for t in stack.rotary_emb(torch.zeros(1, 12, cfg.hidden_size), pos))

hd, H, KV = lay.head_dim, lay.heads, lay.kv_heads
T = 12
xin = x.unsqueeze(0)
qg, gate_h = torch.chunk(attn.q_proj(xin).view(1, T, -1, hd * 2), 2, dim=-1)
gate_h = gate_h.reshape(1, T, -1)
qh = attn.q_norm(qg.view(1, T, -1, hd)).transpose(1, 2)
kh = attn.k_norm(attn.k_proj(xin).view(1, T, -1, hd)).transpose(1, 2)
vh = attn.v_proj(xin).view(1, T, -1, hd).transpose(1, 2)
qr, kr = apply_rotary_pos_emb(qh, kh, cos.unsqueeze(0), sin.unsqueeze(0))

conv = LD.convert_attention_layer(".", 0, cfg) if False else None
raw = {"q_proj.weight": attn.q_proj.weight.detach(), "k_proj.weight": attn.k_proj.weight.detach(),
       "v_proj.weight": attn.v_proj.weight.detach(), "o_proj.weight": attn.o_proj.weight.detach(),
       "q_norm.weight": attn.q_norm.weight.detach(), "k_norm.weight": attn.k_norm.weight.detach()}
q_w, gate_w = L.split_gated_q(raw["q_proj.weight"], lay)
packed = L.pack_qkv(q_w, raw["k_proj.weight"], raw["v_proj.weight"], lay)
from sys1.kernels import gemm_kernel, rope_kernel, attn_sw_kernel  # noqa: E402
pk = gemm_kernel.forward(x, packed, None, "none", torch.float32).view(T, 3, H, hd).contiguous()
qn = L.permute_norm(raw["q_norm.weight"], lay); kn = L.permute_norm(raw["k_norm.weight"], lay)
pk[:, 0] = LD.rmsnorm_rows(pk[:, 0], qn, cfg.rms_norm_eps)
pk[:, 1] = LD.rmsnorm_rows(pk[:, 1], kn, cfg.rms_norm_eps)
ce, se = L.expand_rope_tables(cos, sin, lay)
pk = rope_kernel.forward(pk, ce, se)

perm = torch.tensor(L.rope_pair_permutation(lay), dtype=torch.long)
idx = perm.view(1, 1, hd).expand(H, T, hd)
qr_ref = torch.gather(qr[0], dim=-1, index=idx)
kr_ref = torch.gather(rep(kr[0], lay.group), dim=-1, index=idx)
rv = rep(vh[0], lay.group)
cmp("q_after_norm_rope", pk[:, 0].transpose(0, 1), qr_ref)
cmp("k_after_norm_rope", pk[:, 1].transpose(0, 1), kr_ref)
cmp("v_packed", pk[:, 2].transpose(0, 1), rv)


mask = torch.full((T, T), float("-inf")).triu(1).view(1, 1, T, T)
sc = (qr_ref.unsqueeze(0) * hd ** -0.5) @ rep(kr[0], lay.group).unsqueeze(0).transpose(-1, -2) + mask
w_ref = torch.softmax(sc.float(), dim=-1)
ref_ctx = (w_ref @ rv.unsqueeze(0)).squeeze(0).transpose(0, 1).reshape(T, H * hd)
o1 = attn_sw_kernel.forward(pk[:, 0].transpose(0, 1), pk[:, 1].transpose(0, 1), pk[:, 2].transpose(0, 1),
                            window=T, scale=hd ** -0.5, out_dtype=torch.float32).transpose(0, 1).reshape(T, H * hd)
cmp("attn_context", o1, ref_ctx)
g_ours = gemm_kernel.forward(x, gate_w, None, "none", torch.float32)
cmp("gate_values", g_ours, gate_h[0])
fin_ours = gemm_kernel.forward(o1 * torch.sigmoid(g_ours), raw["o_proj.weight"], None, "none", torch.float32)
fin_ref = attn.o_proj((ref_ctx * torch.sigmoid(gate_h[0])).unsqueeze(0))[0]
cmp("final_o_proj", fin_ours, fin_ref)

# --- 隔离：先看 norm 后、rope 前的置换对不对，再看 rope 后 ---
q_pre_hf = attn.q_norm(qg.view(1, T, -1, hd))[0]                  # (T,H,hd) HF 座次
q_pre_our = LD.rmsnorm_rows(gemm_kernel.forward(x, packed, None, "none", torch.float32)
                            .view(T, 3, H, hd)[:, 0], qn, cfg.rms_norm_eps)
cmp("q_pre_rope_perm", q_pre_our.transpose(0, 1), torch.gather(q_pre_hf.transpose(0, 1), -1, idx))
q_post_hf = qr[0]
c2, s2 = L.expand_rope_tables(cos, sin, lay)
solo = LD.rmsnorm_rows(gemm_kernel.forward(x, packed, None, "none", torch.float32).view(T, 3, H, hd).contiguous(),
                       torch.ones(32), cfg.rms_norm_eps)
print("cos len", cos.shape, "rd", lay.rotary_dim, "half", lay.half, "perm", L.rope_pair_permutation(lay)[:8], "...")

# --- 对照：不换座次（perm_qk=False）时，norm+投影是否与 HF 完全一致 ---
packed_plain = L.pack_qkv(q_w, raw["k_proj.weight"], raw["v_proj.weight"], lay, perm_qk=False)
pp = gemm_kernel.forward(x, packed_plain, None, "none", torch.float32).view(T, 3, H, hd)
norm_w = raw["q_norm.weight"]
pp0 = LD.rmsnorm_rows(pp[:, 0], norm_w, cfg.rms_norm_eps)
cmp("plain_vs_hf_norope", pp0.transpose(0, 1), q_pre_hf.transpose(0, 1))
cmp("qidx_check", torch.gather(q_pre_hf.transpose(0, 1), -1, idx), q_pre_our.transpose(0, 1))

q_raw_our = gemm_kernel.forward(x, q_w, None, "none", torch.float32).view(T, H, hd).transpose(0, 1)
cmp("q_proj_only", q_raw_our, qg[0].transpose(0, 1))
pp0v = pp[:, 0].transpose(0, 1)
d = (pp0v - torch.gather(q_pre_hf.transpose(0, 1), -1, torch.arange(32))).abs()
bad = (d.max(dim=-1).values > 1e-5).nonzero()
print("bad (head,tok) count:", bad.shape[0], "first:", bad[:4].tolist())
print("head0 row0 ours:", pp0v[0, 0, :6].tolist(), "hf:", q_pre_hf[0, 0, :6].tolist())
