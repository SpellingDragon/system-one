"""4B 教师可行性实测：载入内存 + 推理吞吐（伪标场景口径）。"""
import time, torch
from modelscope import snapshot_download
t0 = time.time()
path = snapshot_download("Qwen/Qwen3-4B", cache_dir="bench/ms_models")
print(f"downloaded {time.time()-t0:.0f}s", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).to("mps").eval()
tok = AutoTokenizer.from_pretrained(path)
print(f"载入 driver_mem={torch.mps.driver_allocated_memory()/1e9:.2f}GB "
      f"(recommended {(torch.mps.recommended_max_memory()/1e9):.0f}GB)", flush=True)
ids = tok("Evidence: the sensor reads 42 degrees. Question: is it hot? Answer with one letter.", return_tensors="pt").input_ids.to("mps")
with torch.no_grad():
    for _ in range(2): m(ids)                       # 预热
    torch.mps.synchronize(); t1 = time.perf_counter()
    for _ in range(5): m(ids)                        # 短上下文前向
    torch.mps.synchronize(); t_short = (time.perf_counter()-t1)/5
    big = torch.randint(100, 50000, (4, 1024), device="mps")
    for _ in range(2): m(big)
    torch.mps.synchronize(); t1 = time.perf_counter()
    for _ in range(3): m(big)                        # 伪标口径 4×1024
    torch.mps.synchronize(); t_big = (time.perf_counter()-t1)/3
    out = m.generate(ids, max_new_tokens=16, do_sample=False)   # 生成式兜底口径
print(f"短ctx(≈25tok) 前向 {t_short*1e3:.0f}ms | 4×1024 前向 {t_big:.2f}s ({4096/t_big:,.0f} tok/s) "
      f"| 生成16tok {(len(out[0])-ids.shape[1])}tok 成功 | 峰值 driver_mem={torch.mps.driver_allocated_memory()/1e9:.2f}GB", flush=True)
