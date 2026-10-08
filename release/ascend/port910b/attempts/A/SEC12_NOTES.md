# A 案（shared.l1 gemm 数值路）侦察卷宗 — 2026-10-09

## 已定的完整事实链（全部实测/原文证据）
1. **触发**：A/B scope `shared.l1` + `transpose_B=True` → codegen `_lower_l1()`（gemm_mad.py:161-166）发 `ascend_gemm_l1<M,K,N,tile,trans,dtype>` 模板路（`__cube__` 单工 kernel，无 mix——发射面 9 符号，见 a2_probe EMISSION SURFACE）。
2. **建材真名**（官方 Cal 层解包所得，a29c/a30c）：
   - GM→L1：native `copy_gm_to_cbuf(dst,src,sid,blockCount,blockLen,srcStride,dstStride)`（32B 单位；§12 已用）
   - L1→L0A/B：`load_cbuf_to_ca(dst,src,startIndex,repeatTimes,srcStride,...)` / `load_cbuf_to_cb`（**bisheng 自带头无通用形声明**——是 `CCE_SCALAR()` 的 clang builtin，需 extern 声明或找到其声明头）；另有 **`load_gm_to_ca/cb`（GM→L0 直载）** 可走捷径
   - cube：官方形 `mad(c,a,b,m,k,n,unitFlag,kDirectionAlign,cmatrixSource,cmatrixInitVal)`（10 参，注意 ≠ §10 自拼形——多 kDirAlign 位）
   - L0C→UB：Fixpipe Cal 的 native 件待最后一层（a30 grep 未中——在 load_cc family 或 `fix_matrix_cc_to_cbufubuf_dualout`）
   - Params 结构：LoadData2DParams{startIndex,repeatTimes,srcStride,sid,dstGap,ifTranspose,addrMode}；MmadParams(m,n,k,unitFlag,cmatrixSource,cmatrixInitVal)；FixpipeParamsV220(nSize,mSize,srcStride,dstStride,reluEn)
3. **框架 include 的组装难题**（本轮所停）：
   - 单拎 `dav_c220/*_impl.h` → 缺 LocalTensor（interface 前置）
   - `kernel_operator.h` 全链 → `kernel_operator_dump_tensor_impl.h:605` illegal initializer（`ASCENDC_DUMP` 被 kernel_macros.h:118 默认定义=1，dump 模板体解析出错——官方 cmake 有自己的 -D/-U 矩阵未穷尽）
   - 破解方向（下轮）：**绕框架，直接 extern 声明 `__builtin_cce_*` 系列**（官方 Cal 即其薄包装，签名可从 builtin 定义反查 `CCE_SCALAR` 宏展开=`__builtin_cce_##name`）

## 主线守护态
§12 已加 `TL910B_SEC12_ENABLE` 门（默认关）；B/C/e2e_cube 回归全绿后才入账。

## 上卡验证点（V1/V2/V3 已标 §12 注释）
NZ 布局单位、L12L0 映射方向、l0c2gm 步距——数值真后 gemm rel_err 判据转绿。
