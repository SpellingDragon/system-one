# CI 基线归因台账 · release-gates 首跑 11 红（R-P1-4，2026-10-11）

上游凭据：ubuntu run `38052186621` = `11 failed, 290 passed, 28 skipped, 3 deselected`。
本文按 R23 三分（本变更引入 / 既有缺陷 / 环境伪影）逐条归因，**先证后改**；处置全部只落
`release/tests/*` 的 skip/门与注释，未触碰 `sys1/` 产品语义（design D1/D5、白名单纪律）。

## 0. 结论速览

| 组 | 条数 | 归因 | 处置 | 下一次 ubuntu run 预期 |
|---|---|---|---|---|
| test_backends 方言路 | 4 | 环境（设备条件用例缺门） | `@need_dialect_host`（skipif 无 MPS） | → SKIP ×4 |
| 资产缺件（chinese/longctx/mm/opd） | 6 | 环境（bench/ 按设计不入库） | skip-when-missing（复用仓内既有判据口径） | → SKIP ×6 |
| test_longctx sliding 逐位相等 | 1 | 环境伪影（KNOWN-ENV，数值平台敏感） | **不 skip**，注释挂账 + 本档 §3 | 仍红 ×1（待编排者裁决） |

11 红均非"本变更引入"、均非产品回归；28 个既有 SKIP 口径未动。

## 1. 资产缺件 ×6 → skip-when-missing（统一口径）

根因：`.gitignore:10` `bench/` 不入库（PRODUCTION §5.0 本地装配副本）。仓库此前两种口径并存：
一部分用例 skip（如 `test_registry.py:46-48`、`test_assets.py:40` `real_only`、本档涉及文件内
`test_chinese.py:144-147`），另一部分直接红。R-P1-4 把失败侧统一为既有 skip 纪律，不新造抽象。

| # | 用例 | 缺的资产 | 红签名（ubuntu） | 处置（判据来源） |
|---|---|---|---|---|
| 1 | `test_chinese.py::test_transcribe_derived_fetch_is_zero_traffic_but_keeps_evidence` | `bench/eval_data/assembled/cmmlu-subset/cmmlu-subset.jsonl`（题面原件） | `_fetch_derived` 报 FetchError | 文件存在性 skip，命令沿用本文件 `:198` 口径 `registry fetch --cn`（判据同 `:144-147`） |
| 2 | `test_longctx.py::test_needle_table_is_registry_skeleton_and_deterministic` | `bench/.../needle-synthetic/needle-synthetic.jsonl`（针位骨架） | resolve_table 落 `needle_plan` 回退，assert source=="skeleton" 红 | `@skeleton_only`（判据=既有常量 `longctx.SKELETON_REL`；命令沿用 `longctx.py:164` 报错原文 `fetch --needle`） |
| 3 | `test_longctx.py::test_needle_table_from_plan_mirrors_skeleton` | 同上 | `load_skeleton()` 抛 NeedleError | 同上 |
| 4 | `test_mm.py::test_real_processor_pad_expansion` | `bench/ms_models/models/*Qwen3.5-0.8B*/snapshots/master` | `glob(...)[0]` IndexError | glob 空则 skip（措辞同 `test_assets.py:40` real_only"真快照不在 bench/ms_models"） |
| 5 | `test_mm.py::test_vision_pack_replay_zero_online` | `bench/teacher_cache`（p2-02 B3 视觉伪标包） | `assert len(cache) >= 10` 红（实得 0） | `len<10` 则 skip（阈值原样保留为 skip 判据） |
| 6 | `test_opd.py::test_cache_scaffold_model_id_is_keyed_apart_from_real_teacher` | `bench/teacher_cache/p2_05_pseudo`（scaffold 伪标包） | `assert len(cache) > 0` 红（0>0） | `len==0` 则 skip；键分家断言逻辑一字未动 |

## 2. test_backends ×4 → 设备条件用例缺门（先证假设，再上门）

**假设**：这 4 条走 `get_compiled` 的"方言路"，其第一道闸 `tilelang_available()`
（`sys1/kernels/backends.py:146`）三段判据含"本机真有 MPS"（`backends.py:88`）。
ubuntu 无 MPS → `get_compiled` 在调 builder **之前**就返回 None → 计数 0（`:107-109` 红）、
builder 不执行（`:113-117` 红 `0==2`）、异常路根本没走到（`:134` DID NOT WARN）、
失败记忆无从登记（`:145-157` 红，`len(calls)==0≠1`，报错文案"被反复重试"此时有误导性）。

**证**（host 上伪造 `torch.backends.mps.is_available()→False` 跑改前原副本）：
```
4 failed, 11 passed        # 失败用例与签名和 ubuntu run 完全一致
```
即 ubuntu 红 = 探测口径伪影，**不是**回退逻辑回归（回退语义由同文件 CPU 用例
`test_all_ops_fallback_matches_torch_ref_on_cpu` 等 11 条在 ubuntu 照常守住）。

**处置**：模块级 `need_dialect_host = pytest.mark.skipif(not torch.backends.mps.is_available(), ...)`
挂在 4 条用例上——与 `test_assets.py:41-44` 的 MPS 纪律同族（skip 认"真没有 MPS"，
host 有 MPS 照跑不豁免）。不改 `backends.py`：给探测加旁路=动产品语义，越界。

**已知局限（如实记下）**：门只判"有无 MPS"。若某天在"有 MPS 但没装 tilelang"的 mac 上跑
release 全量，这 4 条仍会红（`tilelang_available()` 因 import 失败判否）。当前 CI 无该形态
job（mac-gates 只跑 `scratch/tools/ci.sh --fast`），暂不为此加抽象；真出现时把门条件收紧为
`mps && tilelang importable` 即可。

## 3. KNOWN-ENV ×1：`test_longctx.py::test_sliding_routes_agree_with_mask_reference`

**判定：纯计算、零资产依赖，不得 skip 掩盖。** `compare_routes(200,32,chunk=48)`
（`sys1/layers/attention.py:394-432`）固定 seed 合成张量，输入与磁盘无关。

红点唯一落在 `kernel_vs_masked_max_abs == 0.0` 的**逐位相等**断言：掩码路物化 200×200
分数矩阵、带状路按 79 宽带分块，两路 fp32 matmul/softmax 的**归约顺序**随 BLAS 实现与
指令集而变——host（arm64, Accelerate/vecLib 轮）位等成立绿；ubuntu（x86 CPU wheel,
OpenBLAS）位等被打破红。数值仍在 allclose 容差内（同报告里 `kernel_allclose=True`
的断言先于位等断言执行），故属"验收口径平台敏感"，非路由回归。

**处置**：不 skip、不放宽（放宽容差/改 xfail 都动 sys1 侧验收口径，越白名单）。用例上方挂
KNOWN-ENV 注释指回本档；在编排者裁决前，下一次 ubuntu run 预期保留这一条红
（`continue-on-error` 尚在，不阻塞合入）。裁决选项：A) 容差改 `<=1e-6` 并同步 spec 口径；
B) `@xfail(strict=True, reason=KNOWN-ENV)` 显式挂账；C) 上 mac runner 设备矩阵后自然消解。

## 4. 负例自证（R14：守卫必须真响）——可复跑

改前基线（host，资产在位，5 个白名单文件）：
```
cd release && .venv/bin/python -m pytest tests/test_backends.py tests/test_chinese.py \
  tests/test_longctx.py tests/test_mm.py tests/test_opd.py -q -m "not integration"
→ 130 passed, 1 skipped, 1 deselected          # 改前/改后同数字：门在 host 不误伤
```

**负例一（伪造 MPS 不可用）**——先存原副本 `cp tests/test_backends.py /tmp/s1_orig/`，
桩脚本（收集前打桩）：
```
torch.backends.mps.is_available = lambda: False; pytest.main([target, "-q", ...])
```
```
BEFORE（原副本）: 4 failed, 11 passed    # 精确复现 ubuntu 4 红
AFTER （现役）  : 11 passed, 4 skipped   # 门真响
host 正跑       : 15 passed              # 有 MPS 不豁免
```

**负例二（隐藏资产文件）**——把 `needle-synthetic.jsonl`、`cmmlu-subset.jsonl`、
`bench/teacher_cache`、`bench/ms_models/models/Qwen--Qwen3.5-0.8B`（须**移出 glob 视野**，
原名加后缀会保留 `*Qwen3.5-0.8B*` 命中）临时改名，跑：
```
BEFORE（原副本）: needle×2 failed；chinese×1 failed；mm pad IndexError；mm pack/opd assert 红
AFTER （现役）  : 全部 SKIPPED（-rs 出示补件命令），0 failed
复原校验        : 资产目录 ls 回原位 ✓（探针脚本带 trap 复原，本次实测已复原）
```

**负例边界（如实记录）**：只抽掉骨架**单文件**做全量时，白名单外的
`test_registry.py::test_extras_qtype_field_uniform` 会 FileNotFoundError——它的 skip 判据是
manifest 整体在场，单文件隐藏构造出 CI 永不出现的"部分装配"态；非本轮 11 红、越白名单，未动。

## 5. 门禁翻转条件（写于 `.github/workflows/ci.yml` release-gates 注释处）

保留 `continue-on-error: true`：本地全绿 ≠ ubuntu 全绿（§2/§3 的环境差在 CI 实况验证前
不撤销保险）。翻转由编排者执行：

1. 观察下一次 push 触发的 ubuntu run；
2. 满足其一：`0 failed`；或唯一红仅剩 §3 的 KNOWN-ENV 一条**且**已按 §3 选项 A/B 裁决；
3. 翻转操作 = 删 `continue-on-error: true` 那一行，其余一字不动（step 里 `set -o pipefail`
   已保 pytest 退出码透传，删行即生效为硬门）。

预期下一次 ubuntu run 稳态：`1 failed (仅 §3 KNOWN-ENV), 290 passed, 38 skipped, 3 deselected` —— 10 条红转 SKIP 只增 skipped（28→38），passed 数不变（红项本就未计入 passed）；host 全量为 `332 passed, 3 skipped, 3 deselected`（收集数与 ubuntu 差在可选依赖的收集期条件，属既有现象非本轮引入）。

## 6. 产物与验证命令索引

- 修改：`release/tests/test_backends.py`（门+注释）、`test_longctx.py`（门+KNOWN-ENV 注释）、
  `test_chinese.py`、`test_mm.py`、`test_opd.py`（skip-when-missing）；`.github/workflows/ci.yml`（仅注释，
  `yaml.safe_load` 复验过：jobs×3、continue-on-error=True、steps=8 不变）
- 新增：本档；`openspec/changes/audit-remediation-1010/tasks.md` R-P1-4 勾选
- 全量自证：`cd release && .venv/bin/python -m pytest tests -q -m "not integration"`
  → `332 passed, 3 skipped, 3 deselected`（host，0 failed）
