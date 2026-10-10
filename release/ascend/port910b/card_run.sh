#!/bin/bash
# card_run.sh — 910B 单窗统一验收（P1-1i 修订：判决件必须量到产品件）
# 幂等、自包含、setsid 后台跑；verdict 行可直接 grep。
#
# ── P1-1i 三处修法（对应 wave2 误判根因，逐条可复核）────────────────────────
# ① [3] 段 target 列表**先跑产品件**（linear_prod / dw_prod / readout_prod：经
#    `ascend/kernels/*_kernel.py` 入口件 + ascend_env 路由，即 train_step 真正用的那条路），
#    原 attempts 内联 case 改名 `*_intrinsic` 作**性能对照轨**保留（判的是 §12 intrinsic 路，
#    从来不是产品件——这正是本元缺陷的来源）。判决行前缀：产品件 `Y-<case>-PROD-*`，
#    对照件 `NUM-<case>-*`。
# ② [2] 段 overlay 落点：编译 `-I` 根是 `<site-packages>/tilelang/src`，而桩头里的引用形如
#    `#include "c_api/asc_simd.h"`（见 patches/common.h:11、debug.h:12、gemm.h:3）⇒ c_api/simt_api
#    必须落 `$PIP_SRC = tilelang/src` 下。旧写法 `PIP_SRC=$(dirname "$TPL")` = `tilelang/src/tl_templates`
#    ⇒ wave2 首轮六件全灭于 `'c_api/asc_simd.h' file not found`。
#    （同一形态在 bundle_ondemand.sh:37 用 `$(dirname "$TPL")/../$d` 是对的，此处按字面写清。）
# ③ [2] 段起跑前**断言**桩头已就位：缺 `$PIP_SRC/c_api/asc_simd.h` 或任一同源头 ⇒
#    `FATAL overlay-missing: <路径>` 并 `exit 1`（旧版 `cp ... || true` 会把缺件吞成静默失败，
#    再到 [3] 炸成一堆看不懂的 file-not-found）。
#
# ── 本机自证钩子（默认全部关闭 ⇒ 真机行为与旧版逐字节一致）──────────────────
#   SYS1_DIR=…           [1] 段 clone 的目标目录（默认 /tmp/sys1）
#   CARDRUN_SKIP_CLONE=1 跳过 git clone，用已就位的 SYS1_DIR（host 无网/禁 git 时自证用）
#   CARDRUN_STOP_AFTER=N 跑完第 N 步即 exit 0（host 自证 [2] 段用 N=2）
#   TLROOT_OVERRIDE=…    覆写 tilelang 根（host 自证时指到一个空沙盒，避免污染 .venv）
#   Y_SELFTEST_DROP_CAPI=1 overlay 完成后**故意删掉** c_api/asc_simd.h ⇒ ③ 的断言必须响
#   CARDRUN_SKIP_BISHENG=1 跳过 bisheng 选项注入（host 自证用，免得改写本机 .venv 里的 bisheng.py）
#
set -uo pipefail
export TILELANG_CACHE_DIR=$(mktemp -d)   # 缓存纪律：key 不含模板内容，必须全新；[3][4] 共享此目录=编译复用
for c in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-8.5.2 /usr/local/Ascend/cann-*; do
  [ -f "$c/set_env.sh" ] && { set +u; source "$c/set_env.sh"; set -u; break; }
done
CCEC=$(find /usr/local/Ascend -path "*ccec_compiler/bin/bisheng" 2>/dev/null | head -1)
[ -n "$CCEC" ] && export BISHENG_HOME=$(dirname "$(dirname "$CCEC")")
export ASCEND_NPU_ARCH=dav-2201

SYS1=${SYS1_DIR:-/tmp/sys1}
STOP_AFTER=${CARDRUN_STOP_AFTER:-}

echo "=== [0] env ==="
python3 -c "import torch; print('torch', torch.__version__, 'npu_avail', hasattr(torch,'npu') and torch.npu.is_available())"
python3 -c "import tilelang" 2>/dev/null || pip3 install -q -i https://pypi.tuna.tsinghua.edu.cn/simple tilelang 2>&1 | tail -1
python3 -c "import tilelang; print('tilelang', tilelang.__version__)"

echo "=== [1] repo @latest ==="
if [ "${CARDRUN_SKIP_CLONE:-0}" = "1" ]; then
  echo "SKIP-CLONE（自证钩子）：用已就位的 $SYS1"
else
  rm -rf "$SYS1"
  git clone -q --depth 1 https://github.com/SpellingDragon/system-one-study.git "$SYS1" || { echo "FATAL clone"; exit 1; }
fi
cd "$SYS1" && git log --oneline -1 || echo "[note] SKIP-CLONE 且无 .git ⇒ 无 log 行"

echo "=== [2] overlay 910B templates + bisheng inject ==="
cd "$SYS1/release/ascend/port910b"
# tilelang 包根：`-I` 根是 $TLROOT/src，桩头按 $TLROOT/src/<c_api|simt_api>/<name>.h 解析
TLROOT=${TLROOT_OVERRIDE:-$(python3 -c "import tilelang,os;print(os.path.dirname(tilelang.__file__))")}
TPL=$TLROOT/src/tl_templates/ascend
PIP_SRC=$TLROOT/src                       # ← P1-1i 修正点（旧：$(dirname "$TPL") = tl_templates，错一层）
echo "TLROOT=$TLROOT"
echo "PIP_SRC(桩头落点根)=$PIP_SRC   TPL(模板落点)=$TPL"
mkdir -p "$TPL" "$PIP_SRC/c_api" "$PIP_SRC/simt_api"
cp -f patches/port910b_compat.h patches/gemm.h patches/common.h patches/debug.h \
      patches/dcache_bypass.h patches/numeric_limits.h "$TPL/" || { echo "FATAL overlay-cp(templates)"; exit 1; }
cp -f patches/c_api/*.h "$PIP_SRC/c_api/"     || { echo "FATAL overlay-cp(c_api)"; exit 1; }
cp -f patches/simt_api/*.h "$PIP_SRC/simt_api/" || { echo "FATAL overlay-cp(simt_api)"; exit 1; }
if [ "${Y_SELFTEST_DROP_CAPI:-0}" = "1" ]; then
  rm -f "$PIP_SRC/c_api/asc_simd.h"
  echo "[selftest] 已按 Y_SELFTEST_DROP_CAPI=1 删掉 c_api/asc_simd.h ⇒ 下面的断言应当响"
fi
# ③ 存在性断言（防再犯）：cp 的返回码不够，必须按“编译期真正会找的那个路径”逐个验文件
for f in "$TPL/port910b_compat.h" "$TPL/common.h" \
         "$PIP_SRC/c_api/asc_simd.h" \
         "$PIP_SRC/simt_api/asc_simt.h" "$PIP_SRC/simt_api/asc_fp16.h" \
         "$PIP_SRC/simt_api/asc_bf16.h" "$PIP_SRC/simt_api/asc_fp8.h"; do
  [ -f "$f" ] || { echo "FATAL overlay-missing: $f  ← 桩头没落到 -I 根（-I 根=$TLROOT/src，引用形如 \"c_api/asc_simd.h\"）"; exit 1; }
done
ls -l "$PIP_SRC/c_api/asc_simd.h" || { echo "FATAL overlay-missing: $PIP_SRC/c_api/asc_simd.h"; exit 1; }
echo "OVERLAY-OK headers=$(ls "$PIP_SRC/c_api" "$PIP_SRC/simt_api" | grep -c '\.h$') tpl=$(ls "$TPL"/*.h | wc -l | tr -d ' ')"
echo "compat §12+fill: $(grep -c 'asc_fill_l1\|(uint8_t)sid, false' "$TPL/port910b_compat.h")"
if [ "${CARDRUN_SKIP_BISHENG:-0}" = "1" ]; then
  echo "[selftest] SKIP-BISHENG：不碰本机 tilelang/contrib/bisheng.py（host 自证专用，真机必须注入）"
else
  python3 patches/patch_bisheng.py && echo INJECT-OK || { echo "FATAL bisheng-inject"; exit 1; }
fi
if [ -n "$STOP_AFTER" ] && [ "$STOP_AFTER" -ge 2 ] 2>/dev/null; then echo "STOP-AFTER-2 OVERLAY-OK"; exit 0; fi

echo "=== [3] numerics（产品件判决 = 下窗要的东西；intrinsic = 性能对照轨）==="
cd "$SYS1/release/ascend/port910b"
# 产品件（入口件 → ascend_env 路由 → *_asc.py 生产正文）；判决行 Y-<case>-PROD-PASS / Y-<case>-ASC-COMPILE-PASS
for t in linear_prod dw_prod readout_prod; do
  echo "CASE-PROD-$t"
  # 预算 900s：真机首次编译+发射的余量（容器冷编译实测 4–7s，卡上留足）
  timeout 900 env DMLAYA_ASCEND_TARGET=ascend python3 run_numerics.py --only "$t" --track card 2>&1 \
    | grep -E "Y-$t|FAIL|Error:|error:|aicore" | head -6
done
# 对照轨（attempts/§12 intrinsic 正文，非产品件）：仅供与上一波数字对比，不作产品结论
for t in gemm_l1_intrinsic dw_intrinsic readout_intrinsic addln_intrinsic gdn; do
  echo "CASE-PERF-$t"
  timeout 500 python3 run_numerics.py --only "$t" 2>&1 \
    | grep -E "NUM-$t|FAIL|Error:|error:|aicore" | head -4
done
echo "CASE-delta"
timeout 500 python3 run_delta_card.py 2>&1 | grep -E "NUM-delta_fwd|FAIL|Error" | head -3
if [ -n "$STOP_AFTER" ] && [ "$STOP_AFTER" -ge 3 ] 2>/dev/null; then echo "STOP-AFTER-3"; exit 0; fi

echo "=== [4] mixed-stack train step on npu (P2 首步) ==="
cd "$SYS1/release"
timeout 900 env DMLAYA_ASCEND_TARGET=ascend python3 ascend/train_step.py --target ascend 2>&1 | grep -E "P2-STEP|FAIL|Error|error:|aicore" | head -6

echo "=== [5] verdict summary ==="
echo "RUNNER-DONE"
