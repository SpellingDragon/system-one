# Tasks: p2-09-chinese-track [W2 · 依赖 p2-03+05；与 06/07/08/10 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/chinese-track/spec.md`。

### 工作项 A 转写与评测

- [ ] A1 CMMLU 子集 + CLUE(tnews/ocnli) 决策化转写（choice/noul）+ registry pin —— 验证：`python -m pytest tests/test_chinese.py -k transcribe -q`

### 工作项 B 配比消融与对照

- [ ] B1 中文配比消融：30%/50% 两档 tiny SFT 对照 —— 验证：两 run-id + 中文 acc 对照入 runs
- [ ] B2 G3 对照表：DML vs laya（亲跑，崩溃域如实记录 + 失败样本 3 例） —— 验证：对照表含 laya 中文行 + ms + 采样参数
