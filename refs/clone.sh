#!/usr/bin/env bash
# DML 课程参考仓一键拉取 —— GUIDE §2.1 / PRODUCTION §3.1 的"文件级参考地图"依赖这些仓。
# 用法: bash refs/clone.sh   （克隆到仓库根下 refs/；§2.1 中 <仓名>/... 路径按 refs/<仓名>/... 解析）
# 幂等: 已存在的仓跳过; 课程发布点以 commit 锁定(pinned), 保证地图路径不断裂。
set -euo pipefail

REFS="$(cd "$(dirname "$0")/.." && pwd)/refs"
mkdir -p "$REFS" && cd "$REFS"

clone() {  # clone <目录名> <url> [rev]
  local name="$1" url="$2" rev="${3:-HEAD}"
  if [ -d "$name/.git" ]; then
    echo "SKIP  $name (已存在)"
    return
  fi
  git clone -q "$url" "$name"
  if [ "$rev" != "HEAD" ]; then (cd "$name" && git checkout -q "$rev"); fi
  echo "OK    $name @ $(git -C "$name" rev-parse --short HEAD)  <- $url"
}

# —— 教学主线（一阶段） ——
clone llms-from-scratch-cn   https://github.com/datawhalechina/llms-from-scratch-cn.git
# —— 内核与 DSL（M1） ——
clone tilelang               https://github.com/tile-ai/tilelang.git
clone TileKernels            https://github.com/deepseek-ai/TileKernels.git
# —— 基线与同构参照 ——
clone laya                   https://github.com/NandhaKishorM/laya.git
# StartLux-Decision: 未核实到公开 GitHub 仓(代码 Apache-2.0, 权重 CC BY-NC 不随克隆), 由课程方提供本地包/镜像地址:
# clone StartLux-Decision    <课程提供的URL>
# —— 长上下文机制参照（M2/二阶段） ——
clone Naive-N0.5-Flash       https://github.com/NaiveAI-Labs/Naive-N0.5-Flash.git
# —— 评测工程与决策方法论（二阶段为主；⚠ CC BY-NC-SA，只重写成文、禁拷贝代码入本仓 MIT 树） ——
clone jev-cookbook           https://github.com/datawhalechina/jev-cookbook.git

echo
echo "提醒: benchmark 数据不属于本脚本, 见 PRODUCTION.md §5.0 —— Intern-Decision @ 2f81580 / jevbench @ 7ce310c7 / typed-decisions @ f7a2487e (pin 到 bench/)"
