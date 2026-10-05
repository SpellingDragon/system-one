# Tasks: p2-06-opd-gkd [W2 · 依赖 p2-05+02；与 07/08/09/10 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/opd-gkd/spec.md`。

### 工作项 A 采样器

- [ ] A1 on-policy 采样器：学生 top-k（默认 8）选项分布，接口可被 RL 复用 —— 验证：`python -m pytest tests/test_opd.py -k sampler -q`

### 工作项 B 蒸馏损失

- [ ] B1 稠密蒸馏损失：reverse-KL（可切 JSD）+ 数值方向单测（KL(p_t‖p_s)） —— 验证：`python -m pytest tests/test_opd.py -k kl -q`
- [ ] B2 缓存命中训练单测（教师前向计数=0） —— 验证：`python -m pytest tests/test_opd.py -k cache -q`

### 工作项 C 训练与消融

- [ ] C1 tiny OPD run（载 p2-05 的 SFT 产物续训） —— 验证：runs/ 含 opd run-id 且 config.parent_run_id 指向 sft
- [ ] C2 消融对：on/off 同 seed 两 run，Δ（acc/ECE）结论入 notes（零/负增益如实） —— 验证：两 run-id + notes 含 `OPD 增益 Δ=` 行
