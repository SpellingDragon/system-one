# 本地 910B 编译判决环境（P0-2L）

一次性构建：`bash local_env.sh`（colima aarch64 + ubuntu:22.04 via DaoCloud + CANN 8.5.0 toolkit，安装包放 cann_cache/ 已 ignore）。

日常循环（零成本）：
```bash
docker exec cann910b bash -c 'source /usr/local/Ascend/cann-8.5.0/set_env.sh; \
  PIP=/usr/local/lib/python3.10/dist-packages/tilelang/src/tl_templates/ascend; \
  cp /tilelang/src/tl_templates/ascend/{port910b_compat.h,debug.h,dcache_bypass.h,numeric_limits.h,common.h} $PIP/ 2>/dev/null; \
  cd /tmp && BISHENG_HOME=$ASCEND_HOME_PATH/aarch64-linux/ccec_compiler python3 /tmp/e2e_verdict.py'
```
要点：
- 必须用 **内层 ccec_compiler/bin/bisheng**（BISHENG_HOME 覆盖）+ `.asc` 自动语言（外层 wrapper 硬传 -x cce 会撞 mad feature-gate）；
- pip bisheng.py 需注入 `-DTL_PORT910B_NATIVE_TYPES`（一次性，见 run 1008-a545）；
- 主仓模板树是补丁真源，容器 pip 树只是运行副本。
