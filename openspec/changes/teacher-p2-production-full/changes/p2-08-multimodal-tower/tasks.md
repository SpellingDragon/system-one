# Tasks: p2-08-multimodal-tower [W2 · 依赖 p2-01+02+03；与 06/07/09/10 并发]

> 三级结构：`### 工作项` → 孙任务 checkbox。spec：`specs/multimodal-tower/spec.md`。

### 工作项 A 视觉塔与装配

- [ ] A1 ViT 载入（SigLIP/CLIP，ModelScope 优先）+ 冻结→解冻两段配置 —— 验证：`python -m pytest tests/test_mm.py -k vit -q`
- [ ] A2 `sys1/data/mm.py` 图文装配（image token 入 state 段，schema 兼容） —— 验证：`python -m pytest tests/test_mm.py -k assemble -q`

### 工作项 B 对照与评测

- [ ] B1 加图/去图对照点 run（同一样本两预测入 runs） —— 验证：runs/ 含 mm 对照 run-id
- [ ] B2 `sys1/eval/multimodal.py`：MMBench-CN 子集（≥200 题）+ OCRBench 抽样，含无图对照列 —— 验证：评测报告含 acc/样本数/run-id/无图列
- [ ] B3 视觉伪标缓存复用冒烟（二次冒烟教师前向=0） —— 验证：`python -m pytest tests/test_mm.py -k pseudo_cache -q`
