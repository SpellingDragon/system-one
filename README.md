# deep-multimodal-laya (DML)

基于 **TileLang** 的类 laya 非自回归 System-1 决策引擎课程项目，分两部分（同一仓库、同一架构：因果 decoder + 选项字母读出 + 类型温度）：

| 部分 | 手册 | 一句话 |
|---|---|---|
| 一 · 学习版 | **[GUIDE.md](GUIDE.md)** | 从零（禁载任何预训练权重）造"小脑"+决策程序+校准，TileLang 内核跑通 CUDA/MPS/昇腾 |
| 二 · 正式版 | **[PRODUCTION.md](PRODUCTION.md)** | 载 0.8B 级开源 backbone，SFT→OPD→RL 三栈；**同 harness 胜 laya 与 StartLux-0.8B**（Jev 为参考上限）；1M/多模态/中文生产化；论文级技术报告收尾 |

## 上手顺序

```bash
bash refs/clone.sh          # 0. 拉齐参考仓（pin，§2.1/§3.1 文件级地图的路径基础）
pip install -e ".[dev]"     # 1. 本仓库
# 2. 之后照 GUIDE §8「起步五步」执行；二阶段 benchmark 拉取见 PRODUCTION §5.0
```

## 目录地图（每个目录内有细指引 README）

```
dmlaya/      ★ 两轨共享实现层（decision=两阶段同构桥梁；勿分叉）
  decision/  渲染 + 字母读出 + 类型温度 + 分组票选     ← 先读这里
  layers/    注意力/rope/indexpool（混合注意力机制层）
  kernels/   TileLang 三件套 _kernel/_cuda/_mps/_asc
  data/      样本 schema / 转写 / 装配
  eval/      唯一评测 harness + pin registry + 双基线  ← 一切数字出自这里
  lang/ vision/ testing/   （空位，到对应阶段按 README 认领）
learning/    一阶段 s0–s3 脚本（禁第三方权重）
production/  二阶段 assets/sft/opd/rl + teachers/
serving/     二阶段部署：/v1/systemone + FP8 + 跨后端
runs/        实验 lab notebook —— 论文证据链唯一真源（§9.1）
report/      技术报告 LaTeX + 互审记录（§9.2/§9.3）
examples/ tests/ docs/ configs/   共用轻目录（约定见 dmlaya 与两手册）
refs/        参考仓克隆地（脚本在此；产物被 .gitignore）
```

## 三条全局红线（细则见各手册）

1. **learning/ 轨禁止加载任何第三方预训练权重**（GUIDE §7-0）；production/ 轨允许。
2. **所有跑分走 `dmlaya/eval/`**；laya 与 StartLux-0.8B 双基线须同 harness 亲跑（PRODUCTION §6）。
3. **数字必须可溯源**：报告/答辩中每个数字对应 `runs/` 内一个 run；无 run-id 视为捏造（PRODUCTION §11-0）。
4. **一阶段注释即评分证据**：中文注释 + 文件头三件套 + `白话：` 段落（不用技术名词讲明白），`python tools/check_comments.py` 为 CI 门（GUIDE §6.1）。
