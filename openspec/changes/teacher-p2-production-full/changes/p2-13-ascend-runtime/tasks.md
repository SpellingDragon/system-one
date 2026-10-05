# p2-13 · tasks（孙任务）

- [x] R1 编写 `ascend/env_setup.sh` 与 `selfcheck.py`（版本 pin：CANN/torch_npu/tilelang/TileKernels；本地可跑 dry-run） —— 验证：`bash ascend/env_setup.sh --dry-run` exit 0
- [x] R2 【本地】七类算子方言件 target=cpu 语义对拍全绿（接口与 `sys1/kernels/` 同名） —— 验证：`python -m pytest tests/test_ascend_gradcheck.py -k cpu -q` exit 0
- [ ] R3 【C1 探针·云端】910B 上跑通 tilelang 方言编译+TileKernels 昂腾后端+torch_npu 前向参照，产吞吐初值与底座决策 —— 验证：run notes 含探针结论行（方言可用性/数值/吞吐三行）
- [ ] R4 【C2·云端】七算子 NPU 编译+梯度对拍绿（fp32 互锁） —— 验证：`pytest tests/test_ascend_gradcheck.py -q`（NPU 环境下）exit 0
- [ ] R5 【C2·云端】自研栈 vs torch_npu 基准表（单算子+SFT 步级） —— 验证：run notes 含基准表（tok/s 两列）
- [ ] R6 实现 `ascend/cost_ledger.py` 分段记账并接入 C1–C4 各段 —— 验证：ledger 表含各段时长/费用行
- [ ] R7 【C4】tiny 冒烟全链定档 + gate 成本报价（¥600 熔断检查） —— 验证：run notes 含报价行与熔断判定
