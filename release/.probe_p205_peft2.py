"""p2-05 探针③b：peft 0.21 的 LoRA 键名/初始化/前向口径，作为自研注入件的参照尺。"""
import math

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn

torch.manual_seed(0)
R, ALPHA, DROP = 16, 32, 0.05
IN, OUT = 32, 24


class Holder(nn.Module):
    """给 Linear 一个能命中 target_modules 的名字（peft 按末段名匹配）。"""

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
A = [v for k, v in sd.items() if "lora_A" in k][0]
B = [v for k, v in sd.items() if "lora_B" in k][0]
print("B 全零？", bool((B == 0).all()), "A 标准差", round(float(A.std()), 6),
      "1/sqrt(IN)=", round(1 / math.sqrt(IN), 6))

x = torch.randn(2, 5, IN)
scaling = ALPHA / R
lin = holder.q_proj
w_before = lin.base_layer.weight.detach().clone()
peft_model.eval()
with torch.no_grad():
    peft_out = peft_model(x)
manual = F.linear(x, w_before) + ((x @ A.T) @ B.T) * scaling
print("eval 态 peft vs 手工旁路 最大差", float((peft_out - manual).abs().max()))

peft_model.merge_adapter()
print("merge 后权重变化均值", float((lin.base_layer.weight - w_before).abs().mean()),
      "对照 (scaling·B@A) 均值", float((B @ A * scaling).abs().mean()))
print("merge 逐元素最大差",
      float((lin.base_layer.weight - (w_before + B @ A * scaling)).abs().max()))

torch.manual_seed(1)
peft_model.train()
holder2 = Holder(nn.Linear(IN, OUT, bias=False))
holder2.q_proj.load_state_dict({"weight": w_before})
pm2 = get_peft_model(holder2, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROP,
                                         target_modules=["q_proj"], bias="none", task_type=None))
pm2.load_state_dict({k: v for k, v in sd.items()}, strict=False)
torch.manual_seed(1)
out_train = pm2(x)
torch.manual_seed(1)
xd = F.dropout(x, p=DROP, training=True)
manual_drop = F.linear(x, w_before) + ((xd @ A.T) @ B.T) * scaling
print("train 态 dropout 口径 peft vs 手工 最大差", float((out_train - manual_drop).abs().max()))
"""p2-05 探针③b：peft 0.21 的 LoRA 键名/初始化/前向口径，作为自研注入件的参照尺。"""
import math

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn

torch.manual_seed(0)
R, ALPHA, DROP = 16, 32, 0.05
IN, OUT = 32, 24


class Holder(nn.Module):
    """给 Linear 一个能命中 target_modules 的名字（peft 按末段名匹配）。"""

    def __init__(self, lin):
        super().__init__()
        self.q_proj = lin

    def forward(self, x):
        return self.q_proj(x)


holder = Holder(nn.Linear(IN, OUT, bias=False))
pm = get_peft_model(holder, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROP,
                                       target_modules=["q_proj"], bias="none", task_type=None))
sd = {k: v for k, v in pm.state_dict().items() if "lora" in k}
print("peft lora keys:", {k: tuple(v.shape) for k, v in sd.items()})
A = [v for k, v in sd.items() if "lora_A" in k][0]
B = [v for k, v in sd.items() if "lora_B" in k][0]
print("B 全零?", bool((B == 0).all()), "A std", round(float(A.std()), 6), "1/sqrt(IN)", round(1 / math.sqrt(IN), 6))

x = torch.randn(2, 5, IN)
scaling = ALPHA / R
lin = holder.q_proj
w0 = lin.base_layer.weight.detach().clone()
pm.eval()
with torch.no_grad():
    ref = pm(x)
manual = F.linear(x, w0) + ((x @ A.T) @ B.T) * scaling
print("eval 态 peft vs 手工旁路 最大差", float((ref - manual).abs().max()))

pm.merge_adapter()
print("merge 后与 (w0 + scaling*B@A) 最大差",
      float((lin.base_layer.weight - (w0 + B @ A * scaling)).abs().max()))

h2 = Holder(nn.Linear(IN, OUT, bias=False))
h2.q_proj.weight.data.copy_(w0)
pm2 = get_peft_model(h2, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROP,
                                    target_modules=["q_proj"], bias="none", task_type=None))
pm2.load_state_dict(sd, strict=False)
pm2.train()
torch.manual_seed(1)
out_t = pm2(x)
torch.manual_seed(1)
xd = F.dropout(x, p=DROP, training=True)
manual_d = F.linear(x, w0) + ((xd @ A.T) @ B.T) * scaling
print("train 态 dropout 口径 peft vs 手工 最大差", float((out_t - manual_d).abs().max()))
