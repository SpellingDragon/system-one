"""p2-05 探针③：peft 0.21 的 LoRA 键名/初始化/前向口径，作为自研注入件的参照尺。"""
import math

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn

torch.manual_seed(0)
R, ALPHA, DROP = 16, 32, 0.05
IN, OUT = 32, 24

base = nn.Linear(IN, OUT, bias=False)
ref = get_peft_model(base, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROP,
                                      target_modules=["q_proj"], bias="none", task_type=None)) \
    if False else None

# 包一层带名字的模型，peft 需要 target_modules 名字命中
class Holder(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.q_proj = lin

    def forward(self, x):
        return self.q_proj(x)


holder = Holder(nn.Linear(IN, OUT, bias=False))
peft_model = get_peft_model(holder, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROP,
                                               target_modules=["q_proj"], bias="none", task_type=None))
sd = {k: v for k, v in peft_model.state_dict().items() if "lora" in k}
print("peft lora keys:", {k: tuple(v.shape) for k, v in sd.items()})
print("B 全零？", all(bool((v == 0).all()) for v in sd.values() if "lora_B" in v))
A = [v for k, v in sd.items() if "lora_A" in k][0]
print("A 均值/标准差", round(float(A.mean()), 6), round(float(A.std()), 6),
      "kaiming_bound", round(1 / math.sqrt(IN), 6))

x = torch.randn(2, 5, IN)
scaling = ALPHA / R
lin = holder.q_proj  # peft 就地替换后 holder.q_proj 是 LoraLayer
peft_model.eval()
torch.manual_seed(7)
with torch.no_grad():
    peft_out = peft_model(x)
# 手工旁路（同一份 A/B，dropout 关闭）：base(x) + scaling*(x@A^T)@B^T
w = lin.base_layer.weight
b = lin.base_layer.bias
manual = F.linear(x, w, b) + ((x @ A.T) @ [v for k, v in sd.items() if "lora_B" in k][0].T) * scaling
print("peft vs 手工旁路 最大差", float((peft_out - manual).abs().max()))

# 合并式参照：peft merge 与 lora_asc.merge(scaling·B@A) 是否同值
merged_module = peft_model.merge_adapter() if hasattr(peft_model, "merge_adapter") else None
print("merge 前后权重差均值", float((lin.base_layer.weight - w).abs().mean()) if w is not None else "n/a")
B = [v for k, v in sd.items() if "lora_B" in k][0]
print("lora_asc.merge 口径 (scaling·B@A) 形状", tuple((B @ A * scaling).shape))

# dropout 语义：peft 是先 dropout 再进 A
torch.manual_seed(1)
peft_model.train()
out_train = peft_model(x)
torch.manual_seed(1)
xd = F.dropout(x, p=DROP, training=True)
manual_drop = F.linear(x, w) + ((xd @ A.T) @ B.T) * scaling
print("dropout 口径 peft vs 手工 最大差", float((out_train - manual_drop).abs().max()))
