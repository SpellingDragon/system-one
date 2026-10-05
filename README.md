# System-One — 根级导航

**System-One：从 0 到 1 造一个端侧校准的 System-1 决策引擎**（基于 TileLang 的两阶段课程项目 · 教师参考答案仓）。**本根只做导航**，实体在四个目录：

| 目录 | 内容 | 状态 |
|---|---|---|
| [`scratch/`](scratch/) | **第一阶段（P1 学习版）**：完整主路径结构（sys1/learning/tests/…）+ 手册（GUIDE.md/PRODUCTION.md）+ 全部实验产物（runs/bench）+ P1 导览表 | ✅ 已完成归档（openspec `teacher-p1-scratch-mps`），冻结只读 |
| [`release/`](release/) | **第二阶段（P2 正式版）**：从 scratch 冻结的 `sys1/` 基线出发，SFT→OPD→RL 三栈 + 三扩展的实施主场 | ⏳ 待启动（openspec `teacher-p2-production-full` 活跃，启动门已满足） |
| [`openspec/`](openspec/) | 变更计划唯一真源（多级 specs：一级编排 + 二级子变更 + 三级孙任务） | P1 已 archive，P2 活跃 |
| [`.agent/`](.agent/) | 本项目沉淀的 agent 资产：openspec-multilevel-planning skill + sub-agent 派发模式 | 见 [AGENTS.md](AGENTS.md) |

**新会话/协作者请先读 [AGENTS.md](AGENTS.md)**（阶段工作规则、.agent 资产用法、CI/hook 说明）。

```bash
# P1 复现（scratch，只读使用）
cd scratch && PYTHON=.venv/bin/python bash tools/ci.sh        # 全门禁
bash examples/repro_p1.sh                                      # 一键 s0→s3 + 四轴评测
# P2 启动（release，届时自建 venv 后按 openspec P2 编排执行）
```
