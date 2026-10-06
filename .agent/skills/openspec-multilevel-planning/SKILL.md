---
name: openspec-multilevel-planning
description: Plans large implementation work as multi-level OpenSpec changes with hierarchical specs, then orchestrates multi-agent execution and acceptance review. Level-1 change orchestrates only (mermaid tracking graph, merge rules, DoD, startup gates); level-2 sub-changes are self-contained per decoupled domain (proposal/design/tasks/spec); level-3 leaf tasks carry verification commands. Covers goal-task coverage matrices and consumer-scenario enumeration during planning review, dispatching parallel apply agents with write whitelists, wave-by-wave integration verification, handling empty agent returns, cross-domain defect arbitration, plan-defect write-back during execution, and dual-track acceptance review (code review + runtime evidence audit). Use when asked to create change plans, break large features or course/teacher-reference implementations into sub-domains and granular tasks, orchestrate parallel execution waves, or execute/review such plans (依赖拆解、子领域、孙任务、多级 specs、编排波次、并发实施、验收评审).
---

# OpenSpec 多级变更规划（multi-level spec-driven planning）

将大型实现工作固化为三级结构：**一级变更司编排、二级子变更司领域、三级孙任务司执行**；规划审"覆盖"、执行审"一致"、验收审"证据"。本 skill 源自多轮真实规划的完整复盘（含执行期返工教训），坑均已踩过并化为规则。

## 何时使用

- 用户要求"制定变更计划并拆分任务/子领域/孙任务"
- 大特性、双阶段（如学习版+正式版）、课程教师参考答案等需要多 agent/多人并行的实现规划
- 要求"多级 specs 目录结构"、openspec 层级化、编排波次
- 执行/验收既有计划（apply 编排、双路评审）

## 核心结构（三级）

```
openspec/changes/<top-change>/            # 一级：CLI 可跟踪（4/4 artifacts）
├── proposal.md                           # Why/What/Capabilities 总览（注明 spec 已下沉）
├── design.md                             # 静态决策：D2 子域表/D4 合并规则/D5 三级验收
├── tasks.md                              # 只编排：mermaid 图 + 子变更清单 + 收尾 F 任务
├── specs/orchestration/spec.md           # 一级仅留编排契约（启动门/波次/契约守护/DoD）
└── changes/<NN-domain>/                  # 二级：CLI 不跟踪，一级 checkbox 跟踪
    ├── proposal.md                       # 含"边界与依赖"节（关键！）
    ├── design.md                         # 技术要点/文件清单/风险与回退
    ├── tasks.md                          # 三级孙任务
    └── specs/<capability>/spec.md        # 该域验收契约
```

要点：
- spec 下沉到二级；一级 `specs/` 只放 orchestration（否则 CLI 状态不完整）。
- 二级嵌套不被 CLI 识别是**有意设计**：二级可整体提升为顶层 change 而结构无损；进度靠一级 checkbox 人工更新。

## 工作流（六阶段）

### 阶段 1 澄清范围（AskUserQuestion，2–3 问封顶）

必问三项：**裁剪面**（砍/留/最小占位）、**规模档**（环境约束下的档位）、**规范遵守度**（项目自有质量门）。

### 阶段 2 一级变更（每阶段一个 change）

`openspec new change <name>` → 依次产出 proposal / design / specs / tasks（`openspec instructions <id> --change <name> --json` 取模板）。

design.md 必含决策（编号 D1..Dn，含被否方案）：
- **D2 子域表**：`# | 子域 | spec 能力 | 一句话边界 | 外部依赖`——"外部依赖"列即接口登记表
- **D3 波次表** + 澄清句：波次仅为汇报分组，解锁=依赖入 main
- **D4 合并规则** / **D5 三级验收**（孙任务 exit 0 / 子域 scenarios / 变更级 DoD 可勾选）

### 阶段 3 二级子变更（每域四件套）

proposal 必含**"边界与依赖"**节四要素：依赖（注明接口常数 vs 实现依赖）、被依赖方、接口面、禁止事项。

孙任务三要素：`- [ ] <编号> 动词开头产出 + 关键约束 —— 验证：<可复跑命令>`；训练类验证 = run-id+指标存在。

### 阶段 4 编排落文档（一级 tasks.md）

mermaid 跟踪图（关键路径红标）+ 头部三行（只编排 / 启动门 / 波次语义澄清）+ 清单行含孙任务级前置。

### 阶段 5 规划自审（最关键的防线）

执行 [review-checklist.md](review-checklist.md)。除结构一致性外，**五项覆盖性检查**（目标-任务矩阵、消费者枚举、档位覆盖、性能主张、真实路径）是执行期返工的解药——P1 实证：结构全绿的计划仍可漏掉"内核接入生产模型""反向传播""正式档实跑"三个整工作项，全部到执行期才由用户追问暴露。

### 阶段 6 执行编排（apply：多 agent 并发实施 + 验收评审）

完整 SOP 见 [apply-orchestration.md](apply-orchestration.md)。核心信条：**编排者司集成与裁决，agent 司域内实现；凡报必验、凡修必回归；子 agent 完成情况协议化核查（四查：勾选真实性/产物盘点/数字溯源/遗漏检测），未完成者归因三型（agent 未完成/做错/计划缺任务）后修计划重派；执行期发现的计划缺口必须回写计划与 skill。****派发前置：并发资源划拨审计与波次边界声明（九要素）；agent 写后校验三连是凭据可信的地基。**

## 核心原则（源真实项目实证）

| 原则 | 一句话 | 反面教训 |
|---|---|---|
| 风险前置 | 最不确定的探针放 W0 首孙任务，失败有回退且不阻主线 | 方言可用性未知却排在后段 |
| 参照系先行 | 双基线/对照物与开发解耦，W0 即产出 | 无参照则一切增益不可判 |
| 慢资源缓存先行 | 重型推理的离线缓存列为 W0 核心工作项 | 各消费方重复付慢推理成本 |
| 契约三层守护 | spec 条款 + 合并规则硬拦 + 收尾 diff 断言 | 仅口头约定必被顺手改掉 |
| 负结果入库 | 消融零增益/发散也是交付物，run 不删、notes 必有结论行 | 只留绿曲线 = 掩盖坑 |
| 粒度判据 | 单文件单机制单测试；"一个会话可完成 + 一条验证命令" | 三合一大任务无法独立验证 |
| **目标必有任务承载** | 变更目标的每个关键词（如"计算主体 TileLang 化"）都映射得到孙任务 | 计划审了任务一致性却漏了目标覆盖，内核与模型成两套平行实现 |
| **消费者先于实现** | 每个产出物先枚举消费者（推理/训练/评测/服务）再定验收 | 只想着推理用内核，训练要不要反向传播无人问 |

## 反模式（均真实出现并被修复）

规划期（R1–R5）：
- **R1 跨变更启动门缺失**：后置 change 依赖的"契约"其实现在前者 W1。→ orchestration spec 加启动门。
- **R2 波次内隐藏依赖**：同波两域有孙任务级产物引用。→ 孙任务就地标注前置。
- **R3 波次栅栏心智**：W 标签暗示整波等待。→ 头部澄清"解锁=依赖入 main"。
- **R4 偏大孙任务**：三合一。→ 拆为实现/对拍/组装。
- **R5 时间断言脆弱**：CI 断言时长被慢机误报。→ 退出码为唯一判定。

执行期（R6–R9、R15–R17，详见 apply-orchestration.md）：
- **R6 空返即重做**：agent 空返 ≠ 未完成，先验工作区，重派令改"复跑验证"。
- **R7 信汇报不复验**：数字/绿灯必须编排者 grep 校账 + 全量复跑。
- **R8 跨域缺陷无人认领**：编排者是唯一越域裁决与修复点（修必附回归用例）。
- **R9 集成只在最后**：每波间即做集成验证。
- **R15 编辑工具假成功**：写返回成功而文件实为占位/0 字节/晚到覆盖。→ 写后校验三连（wc+compile+锚 grep）。
- **R16 全局态脆断言**："X 不在模块表/环境为空"顺序依赖，单域绿全量红。→ 快照差集。
- **R17 并发共享资源互踩**：并发 pip、同设备双跑、账目共写、后台被信号停起。→ 派发前资源划拨审计+账目单点归属。
- **R18 锁解无人候**：资源释放（长跑完/窗口空）无人消费等待面，延后验项闲置到用户提醒。→ 锁必携等待表；释放=触发，当场消费。

规划覆盖性（R10–R14，P1 执行期返工所生）：
- **R10 目标无任务承载**：目标关键词（"TileLang 加速""三后端"）没有对应孙任务，执行期用户一问才暴露。→ review 加目标-任务覆盖矩阵。
- **R11 消费者漏枚举**：产出物（内核）的消费者只想到推理漏了训练——训练要 backward 整个工作项缺失。→ review 加消费者场景枚举。
- **R12 配置档位覆盖不全**：配置声明 main 档而孙任务只覆盖 tiny 档；正式档"配置存在≠实跑"。→ review 加档位全覆盖检查；正式档须有实跑任务+命令指引。
- **R13 性能主张无基准**："加速"类主张没有吞吐对比孙任务，直到执行期才实测发现是负优化。→ 主张挂基准任务，数据入 run。
- **R14 mock 冒充路径验证**：monkeypatch 通过 ≠ 远端/真实路径可用（corpus 远端直到执行期才首验）。→ review 加"真实路径验证"单列任务检查。

## 收尾（一级直管 F 任务）

```
- [ ] F1 契约零改动断言（git diff <基线>..HEAD -- <契约目录> 为空）
- [ ] F2 全量门禁（项目 CI 命令 exit 0）
- [ ] F3 DoD 逐项核验记入最终 run notes（用项目自己的 run 工具立收尾 run）
- [ ] F4 openspec archive + 导览文档（域 → 证据/踩坑记录链接，run-id 逐字符核对）
```

完成后汇总呈现：三级结构树、mermaid 编排图、粒度统计、已修复的自审发现。
