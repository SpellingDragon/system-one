#!/bin/bash
# ============================================================================
# bundle_ondemand.sh — P0-2 上机一键（自包含·幂等·单文件，窗口预算 <=10min）
# 用法（910B 实例上）： bash bundle_ondemand.sh
# 输出： COMPILE-VERDICT / RUNTIME-PASS{json} / RUNTIME-FAIL 行 + VERDICT-SUMMARY
# ============================================================================
set -uo pipefail
echo "=== [0] env ==="
SRC="https://github.com/SpellingDragon/system-one-study.git"
SUB=/tmp/sys1 && rm -rf $SUB
git clone -q --depth 1 "$SRC" $SUB || { echo "FATAL: clone failed"; exit 1; }
P910B=$SUB/release/ascend/port910b/patches   # 模板真源（compat/debug/dcache/common/numeric_limits + stubs）
[ -d "$P910B" ] || { echo "FATAL: $P910B missing"; exit 1; }

# CANN env 自适应（官方 set_env.sh 引用未定义变量，source 期间放宽 -u）
set +u
for c in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann-*; do
  [ -f "$c/set_env.sh" ] && { source "$c/set_env.sh"; break; }
done
set -u
CCEC=$(find /usr/local/Ascend -path "*ccec_compiler/bin/bisheng" 2>/dev/null | head -1)
[ -n "$CCEC" ] && export BISHENG_HOME=$(dirname $(dirname "$CCEC"))
ASCEND_NPU_ARCH=dav-2201; export ASCEND_NPU_ARCH
echo "BISHENG_HOME=${BISHENG_HOME:-<default>}  arch=$ASCEND_NPU_ARCH"

echo "=== [1] tilelang ==="
python3 -c "import tilelang" 2>/dev/null || pip3 install -q -i https://pypi.tuna.tsinghua.edu.cn/simple tilelang
python3 -c "import tilelang; print('tilelang', tilelang.__version__)"

echo "=== [2] overlay 910B templates onto pip tree ==="
TPL=$(python3 -c "import tilelang,os;print(os.path.join(os.path.dirname(tilelang.__file__),'src','tl_templates','ascend'))")
cp -f "$P910B"/*.h "$TPL"/ 2>/dev/null
for d in c_api simt_api; do mkdir -p "$(dirname "$TPL")/../$d" 2>/dev/null && cp -f "$P910B"/$d/*.h "$(dirname "$TPL")/../$d"/ 2>/dev/null || true; done
grep -c TL_ASCEND_SIMT "$TPL/common.h" || { echo "FATAL: common.h not patched"; exit 1; }

echo "=== [3] bisheng options injection (NATIVE_TYPES + asc include + ccec path) ==="
python3 - <<'EOF'
import pathlib
p = pathlib.Path(__import__("tilelang").__file__).parent / "contrib" / "bisheng.py"
s = p.read_text()
if "TL_PORT910B_NATIVE_TYPES" not in s:
    old = 'result = ["-O2", "-fPIC", "-std=c++20"'
    assert old in s, "anchor drift"
    import glob, os
    inc = [d for d in glob.glob("/usr/local/Ascend/*/aarch64-linux") + glob.glob("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux")]
    extra = '", "-DTL_PORT910B_NATIVE_TYPES"' + "".join(f', "-I{d}/asc/impl", "-I{d}/asc/include"' for d in inc[:1]) + ']'
    s = s.replace(old + ']', old + extra, 1)
    p.write_text(s); print("patched bisheng.py")
else:
    print("bisheng.py already patched")
EOF

echo "=== [4] compile verdict (local-verified shape) ==="
cd /tmp && timeout 300 python3 "$P910B/../e2e_cube.py" 2>&1 | grep -E "E2E-CUBE|HAS_MIX" | head -2

echo "=== [5] runtime verdict (2048^3 bf16 numeric + tflops) ==="
timeout 480 python3 "$SUB/release/ascend/port910b/run_kernel.py" 2>&1 | tail -3

echo "=== VERDICT-SUMMARY (copy the lines above into run notes) ==="
