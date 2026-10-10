# p2-13 · tasks（甲路·2026-10-07 用户裁定：C5 暂停，GDN/方言移植为唯一主线）

- [x] R3 C1 实勘（run de15）· A1 路线A废（950 不可得）· B1 首轮五层剥四（run 7793）
- [x] **P0-1 编译链打通（离线主战场）**：第 5 层 bisheng 类型系统兼容——离线盘点 tilelang 模板对 950 类型代理（bfloat16_t 转换算子/vec_t/内置 builtin 变量）的全部依赖点，产出 `port910b_compat.h` 兼容层（910B `__bf16` builtin 等价映射）；判据=本地静态穷举清单完备 —— 验证：清单入 run notes，覆盖模板中全部 950-only 符号（grep 举证）
  - 完成凭据（1007）：清单 `release/ascend/port910b/INVENTORY.md` 349 个去重符号 = A35/B20/C250/D44；自证 `cd release/ascend/port910b && ./checklist.sh` → `AUDIT PASS: 穷举 354 个符号 100% 见于清单（unlisted=0）`；兼容层 `port910b_compat.h` 主机三配置 `SELFCHECK PASS`；run `1007-p2-13-p01-910b-inventory-885c`
- [x] **P0-2a 探针工装修复（离线·完成）**：source CANN env 或显式 -I $ASCEND_HOME/include / setsid 脱离 / 每探针 timeout 90 / P3 拆 grep 与编译两段（四条清单见 run p02-first-attempt）—— 验证：bash -n ✓；四修=CANN env source + -I $CANN_INC + timeout 90 全编译件 + find 限深超时（da4438e 随附）
- [x] **P0-2L 本地编译判决环境（2026-10-08 思路重定向·主战场本地化）**：Mac 原生 aarch64 Docker（Ubuntu）+ CANN 8.5.2 toolkit（aarch64 公开下载版）——bisheng 交叉编译不需设备，P2–P9 编译类探针与 vecadd/gemm 编译判决**全部本地零成本闭环** —— 验证：容器内 bisheng --version + 最小 .asc 编译出 .o + probe910b 本地版全批判决落 run
- [x] **P0-2 链通里程碑（上卡 2026-10-08 执行）**：真机 CANN 8.5.2 上 **E2E-CUBE-ONLY PASS**（bundle 锚点漂移现场修复后）；kernel 可装载可执行但 `rel_err=nan`——L1/L0 的 NZ/分形布局未实现，**数值正确性移交 P1-1 首要验证目标**；run be5c 后续档：`bash bundle_ondemand.sh` 一键（clone→overlay→inject→compile→run，本地 clean-tilelang 全真预演已过 [4] E2E-CUBE-ONLY PASS）；卡上唯余 [5] RUNTIME-PASS{json}（2048³ bf16 数值+tflops）—— 验证：bundle 输出行入 run
- [ ] P1-1 亦改本地迭代制：四件改道移植本地编译验证为主、上卡只验数值性能
- [ ] P1-1 readout/gemm/dW/add_ln 四件**改道移植**（向量面最薄可标量化；P0-1 修正：无纯 AIC 件）+对拍 —— 验证：gradcheck NPU 档
  - **P1-1a 编译面已收（2026-10-08 并发波）**：add_ln ✓ / readout ✓ / GDN-conv ✓ / gemm-cube 无回归 ✓（attempts/B、C 双代理，主仓 compat 合并真源 4/4 全绿，run p11-merge）；**P1-1b 续收（2026-10-09 双代理波）**：gemm 全链（含 L0C→GM 输出段 copy_matrix_cc_to_gm 12 参）✓、dW(l0tr 主案 5/5 形)✓、rope(表加载×3)✓、attn_sw(×2)✓——真源 compat 966 行六线集成回归全绿；GAP-D1/D1b 已并真源；余：bf16 标量 cast（生产 fp16 无碍）、G-D2 上游单（内部持有）、G-D3 不投、dW 出口朝向/rope 就地接口=生产接入波裁决
  - **P1-1c 数值面首战（2026-10-10 真机两窗）**：向量面三件绿（GDN-conv 6.35e-08 / rope 3.13e-08 / attn_sw 1.41e-07）——标量化+表加载+§11 数学件路线数值成立；**cube 面双崩**（gemm_l1 与 dW 同型 aicore exception 507015=DMA 非法访问）；gemm direct rel=nan（已知 UB 布局）
  - [ ] **P1-1d cube-V12 位段修复（关键路径）**：官方 mm 链 LoadData2DParams 构造点逐参对照，重打 §12 装填参数（startIndex/repeatTimes/srcStride/dstGap 单位与位序）—— 验证：单窗复验 gemm_l1 64³ + dW 双绿（≤8min）
  - [ ] P1-1e addln/readout 数值 case 补入 run_numerics（rig 修复）+ 下窗顺带收
- [x] P1-2 rope/attn_sw 移植（已随 P1-1b 完成：编译×5 + 数值双绿 1e-7 级，P1-1c 记档）
- [ ] **P1-3 GDN 件移植（本项目旗舰）**：短卷积 ✓ 真机数值绿（6.35e-08，P1-1c）→ 剩余聚焦 **delta rule 前后向**（u/w 门控+beta/s 公式+衰减递推；decay 的 log 缺件=需查 exp 负指数表载或 §11 扩展件）—— 验证：fp32 分步对拍（partial 可入账）+ 真机 rel
- [ ] P2 三栈联调（torch_npu+自研件混合栈，GDN 件替换Conv2D路径）—— 验证：0.8B 训练步跑通
- [ ] R6/R7 成本记账与 gate 报价（恢复 C5 的前置）—— 验证：报价行+熔断判定

## 挂起（甲路裁定）
- p2-05/06/11 的 C5 云端正式训练：**全部挂起**，解锁条件=P2 行（三栈联调）达成
