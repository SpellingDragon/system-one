# System-One Study

**这是一个学习项目**。它记录的不是"一个模型有多强"，而是**一个人如何从 0 到 1 学会造一个端侧决策引擎，并学会用受控实验证明它好在哪**——两阶段，一部教材，一份全程可审计的学习账。

## 学习叙事：一幕一题

### 第一幕 · 从零学会造（`scratch/`，P1 学习版，已完成归档）

题目苛刻得不讲情面：**禁止加载任何第三方预训练权重**。于是一切都得自己来——

- 自己训 BPE 词表（16k，中英双料）→ 自己写 ~40M 因果 decoder（RoPE/GQA/FFN 一手敲出来）→ 自己定义决策程序（末位字母读出 + 类型温度）→ 再亲手把 gemm/rope/attn/LN 写成 **TileLang 内核**并补上全套反向算子；
- 终点不靠感觉：四轴评测一键复现（`repro_p1.sh`，exit 0 即变更级 DoD 判定入口）。最终正式档：**决策质量 choice +31.7pp / noul +25pp 双双越过参考线，parity 20/20，P50 5.8ms**——从零造出的小脑，真的会做判断。

这一幕学到的比分数更值钱，三条刻在 runs 里：字母必须单 token（全计划最大单点，A–Z=32..57 实测钉死后才敢往下走）；CPU 全绿不代表设备路径可行（swap-brain 用例在 MPS 一跑就漏出跨设备泄漏）；同配置双跑会把吞吐腰斩（编排者自己踩的雷，写进了方法论）。

### 第二幕 · 学会造得更好（`release/`，P2 正式版，进行中）

题目换成了科研的口味：**基线与对手用同一个 Qwen3.5-0.8B 基座**（实测 StartLux-0.8B `model_type=qwen3_5`）——架构差异清零，胜负纯看训练配方。于是 P2 的全部设计都是"向最强处学，量化每一步值多少"：

- **三教师分治**：文本向对手家族的 StartLux-4B 蒸馏（本地）、视觉向 GLM-5.3-Flash API 离线伪标（320B 知识，几十元成本，全链端侧零在线）、学生 100% 基于 Apache 2.0 基座（版权边界 D11 白纸黑字）；
- **三栈消融**：SFT（教师 soft target）→ on-policy 蒸馏（per-token reverse KL，GKD 谱系）→ RLCD（log scoring rule + 分域 verifier 双通道，ICLR 2026 同构）——每一栈的增益（或归零）都上消融表，负结果照入库；
- **昇腾自研算子栈**：910B 云端训练推理，七类算子手写 TileLang 方言件（本地 CPU 对拍 20 绿、GDN 反向 partial 如实挂账）——吃自己狗粮的完全体；
- **边界实验**：1M 窗口三件套（滑窗封顶+线性记忆适配器+门控短路）——第一幕的天花板（choice 校准反化、零样本 32K 召回衰减到 0.25）恰是第二幕的靶子。

## 学习的方法论（第三层叙事）

这个项目同时是 **agent 驱动软件工程**的活教材：多级 OpenSpec 变更（一级司编排、二级司领域、三级司执行；规划审覆盖、执行审一致、验收审证据）、子 agent 五段战报与完成度四查、计划缺陷回写闭环（执行期暴露的每个缺口都改写回计划与 skill，R1–R18 反模式全部来自真实翻车）、`runs/` 证据文化（四件套入库、大权重按来源重建、**run 不删**——连三次空跑的曲线壳都留着，因为诚实比好看贵）。

## 你会在这里得到什么

| 想要 | 去读 |
|---|---|
| 从零造决策引擎的完整参考实现 | `scratch/sys1/` + `scratch/learning/`（中文注释三件套，白话可懂） |
| 每步为什么这么做、踩过什么坑 | `scratch/runs/` 与 `release/runs/` 的 notes（结论行制度） |
| 大项目如何拆给多个 agent 并行做 | `openspec/`（13 子域 72 孙任务全勾对账）+ `.agent/`（skill 与派发模板） |
| 端侧推理/蒸馏/RL 的诚实数字 | README 两幕数字 + `release/examples/a2_needle_curve.py` 复跑 |

## 快速上手

```bash
cd scratch && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
PYTHON=.venv/bin/python bash tools/ci.sh      # 四门禁
bash examples/repro_p1.sh                       # 第一幕一键复现（exit 0 = DoD）
cd ../release && .venv/bin/python -m pytest tests -q -m "not integration"   # 第二幕 296 绿
```

## 三条红线（学习项目的品格）

1. 第一幕 `learning/` 轨禁第三方权重；第二幕产出基座必须 Apache 2.0（StartLux 仅限蒸馏/verifier/对照评测）；
2. 一切跑分走 `sys1/eval/` 同 harness，双基线亲跑不引用卡面；
3. `sys1/decision/` 契约零改动——程序即宪法，三层守护。
