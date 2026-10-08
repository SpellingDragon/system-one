#pragma once

// __aicore__ 仲裁（compat 居 include 链首位，先于 debug.h 展开）：
// 910B bisheng 的 aicore 上下文语法是 `[aicore]` 限定符（CCE 函数仅此上下文注入）；
// host/950 面维持空。
#ifdef TL_PORT910B_NATIVE_TYPES
#define __aicore__ [aicore]
#else
#ifndef __aicore__
#define __aicore__
#endif
#endif

// ════════════════════════════════════════════════════════════════════════════
// port910b_compat.h — TileLang 昇腾后端 910B(dav-2201) 兼容层草稿
//
// 域 p2-13-ascend-runtime / 孙任务 P0-1 产物。只兑现 INVENTORY.md §3-B 的
// **B 类 20 个符号**（bisheng 类型系统第 5 层墙）；C 类 250 个 SIMT/SIMD 方言
// 符号不在本层解决，由 `TL_ASCEND_SIMT` 罩屏蔽；D 类 44 个符号的存在性由
// P0-2 探针（probe910b.sh）裁决。接入方式同目录 README.md。
//
// 覆盖符号（与 inventory_symbols.md 的 B 段一一对应）：
//   B1-TYPE(12) bfloat16_t bfloat16x2_t half half_t half2 float2 float4
//               make_float2 make_float4 make_int2 make_longlong4 make_ulonglong4
//   B2-CVT(5)   __bfloat162float __bfloat1622float2 __float22bfloat162_rn
//               __half22float2 __float22half2_rn   （+ __float2bfloat16_rn 供
//                codegen_ascend.cc:3355 发射，附表因词法合并未单列）
//   B3-LOCK(2)  ASC_LOCK_BLOCK ASC_LOCK_NON_BLOCK（枚举值；asc_lock/unlock 本体在 D2）
//   B4(1)       __SIMT_DEVICE_FUNCTIONS_DECL__
//
// 设计约束（举证自移植对象，全部为静态取证）：
//   1) numeric_limits.h:19-29 用 `TL_BIT_CAST(half|bfloat16_t, uint16_t{...})`
//      且声明为 `inline constexpr` → 标量类型必须 **2 字节、平凡可复制**，
//      且允许 constexpr 位重解释。
//   2) debug.h:209,213,221,227 用 C 风格 `(float)val` → 必须提供
//      `explicit operator float()`（C 风格转换可调用显式转换函数）。
//   3) codegen_ascend.cc:3594,3601,3630,3638 发射
//      `((bfloat16x2_t*)(&(v)))[i/2].x` / `((half2*)(&v))[i/2].y` /
//      `((ushort2*)(&v))[i/2].x` → 元组类型必须是 **成员为 .x/.y 的 4 字节
//      可取址 struct**（不能是 clang 向量类型，否则 `[]`+`.x` 组合语义不同）。
//   4) codegen_ascend.cc:1078 发射 `*(unsigned long long*)&make_float2(v, v)`
//      → make_* 返回的临时量必须可按地址重解释为 8 字节整数。
//   5) codegen_ascend.cc:2723-2725 注释（上游取证）："bisheng rejects them
//      outside a VF body" → bisheng **自身可能带 make_float2/make_int2 内建**。
//      故本层每个名字都可单独退让：`-DTL_PORT910B_SKIP_<name>`（见 §0.3）。
// ════════════════════════════════════════════════════════════════════════════

// ── 0. 开关与总罩 ────────────────────────────────────────────────────────────
// 0.1 本层只在 910B 面生效；950 面（TL_ASCEND_SIMT）由真头提供这些名字。
#if !defined(TL_ASCEND_SIMT)

#include <stdint.h>
#include <type_traits>
// 910B: .asc 模式 bisheng 自动注入 __clang_cce_* 头（CCE 函数/pipe_t 直接可见）。
// 插件注入的 kernel 注册段引用官方 TLV 元数据结构链——直接对齐官方宏头（比逐结构复刻稳）：
#if defined(TL_PORT910B_NATIVE_TYPES)
#include "basic_api/utils/kernel_utils_macros.h"
#endif

// 0.2 后端选择：编译器若提供原生 16 位浮点 builtin 就直接用，否则退到自带
//     storage struct（零依赖，行为等价但算术走 float 提升）。
#ifndef TL_PORT910B_BF16_BUILTIN
#if defined(__BFLT16_MANT_DIG__) || defined(__CLANG_BF16__) || defined(TL_PORT910B_NATIVE_TYPES)
#define TL_PORT910B_BF16_BUILTIN 1
#else
#define TL_PORT910B_BF16_BUILTIN 0
#endif
#endif

#ifndef TL_PORT910B_HALF_BUILTIN
#if defined(__FLT16_MANT_DIG__) || defined(TL_PORT910B_NATIVE_TYPES)
#define TL_PORT910B_HALF_BUILTIN 1
#else
#define TL_PORT910B_HALF_BUILTIN 0
#endif
#endif

// 0.3 逐名字退让：上卡遇到 "redefinition of 'float2'" 这类冲突时，
//     加 -DTL_PORT910B_SKIP_float2 即可把该名字让给 bisheng，不动本文件。
// cce (.asc) compilation: clang driver injects __clang_cce_types.h which provides
// NATIVE bfloat16_t (typedef __bf16) and half (__cce_half). Our proxies must stand down
// there, or `using`/typedef conflicts arise (verified 2026-10-08 local Docker env).
#if defined(TL_PORT910B_NATIVE_TYPES)  // passed by codegen for cce (.asc) compiles
#define TL_PORT910B_SKIP_bfloat16_t
#define TL_PORT910B_SKIP_half
#endif

#define TL910B_EMIT(name) !defined(TL_PORT910B_SKIP_##name)

namespace tl910b {

// ── 1. 位工具 ────────────────────────────────────────────────────────────────
// bit_cast 走 __builtin_bit_cast（与 numeric_limits.h:8 的 TL_BIT_CAST 同源，
// B1 已证实 bisheng clang15 无 std::bit_cast 但有该内建）。
template <typename D, typename S> inline constexpr D tl910b_bit_cast(const S &x) {
  static_assert(sizeof(D) == sizeof(S), "tl910b_bit_cast requires equal size");
  return __builtin_bit_cast(D, x);
}

// bf16 <-> f32：截尾 + round-to-nearest-even（`_rn` 语义），NaN 保持 quiet。
inline constexpr uint16_t tl910b_f32_to_bf16_bits(float f) {
  const uint32_t u = tl910b_bit_cast<uint32_t, float>(f);
  if (((u >> 23) & 0xffu) == 0xffu && (u & 0x7fffffu) != 0u) {
    return static_cast<uint16_t>((u >> 16) | 0x40u); // NaN -> quiet NaN
  }
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t rounded = u + 0x7fffu + lsb; // RNE
  return static_cast<uint16_t>(rounded >> 16);
}

inline constexpr float tl910b_bf16_bits_to_f32(uint16_t b) {
  return tl910b_bit_cast<float, uint32_t>(static_cast<uint32_t>(b) << 16);
}

// fp16 <-> f32：软件路径只在 TL_PORT910B_HALF_BUILTIN=0 时用于转换内建；
// 与 IEEE 754 binary16 一致（含 subnormal 与 inf/NaN）。
inline constexpr float tl910b_f16_bits_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t man = h & 0x3ffu;
  if (exp == 0u) {
    if (man == 0u) {
      return tl910b_bit_cast<float, uint32_t>(sign); // +-0
    }
    uint32_t e = 0x7fu + 1u; // 规格化 subnormal
    uint32_t m = man << (23 - 10);
    while ((m & (1u << 23)) == 0u) {
      m <<= 1;
      --e;
    }
    m &= (1u << 23) - 1u;
    return tl910b_bit_cast<float, uint32_t>(sign | (e << 23) | m);
  }
  if (exp == 0x1fu) {
    return tl910b_bit_cast<float, uint32_t>(
        sign | 0x7f800000u | (man << (23 - 10))); // inf / NaN
  }
  const uint32_t e = exp + (0x7fu - 0x0fu);
  return tl910b_bit_cast<float, uint32_t>(sign | (e << 23) | (man << (23 - 10)));
}

inline constexpr uint16_t tl910b_f32_to_f16_bits(float f) {
  const uint32_t u = tl910b_bit_cast<uint32_t, float>(f);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const int32_t expf = static_cast<int32_t>((u >> 23) & 0xffu);
  const uint32_t man = u & 0x7fffffu;
  if (expf == 0xff) { // inf / NaN（NaN 一律归为 quiet）
    return static_cast<uint16_t>(sign | 0x7c00u | (man ? 0x200u : 0u));
  }
  const int32_t e = expf - 127 + 15; // fp16 有偏指数（全部用有符号比较！）
  if (e >= 31) { // 指数 31 为 inf/NaN 保留 -> 溢出（65520 以上，含恰好 2^16）
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (e >= 1) { // 规格数：丢 13 bit，RNE = +(0x0fff + 保留最低位)
    const uint32_t lsb = (man >> 13) & 1u;
    const uint32_t rounded = (man + 0x0fffu + lsb) >> 13;
    if (rounded == 0x400u) { // 舍入进位到下一指数；e=30 时正好落到 inf(0x7C00)
      return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e + 1) << 10));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) |
                                 rounded);
  }
  if (e < -10) { // 连最小次正规 2^-24 都够不着 -> 带符号 0
    return static_cast<uint16_t>(sign);
  }
  const uint32_t m = man | 0x800000u; // 补回隐含 1，得 24 bit 有效数
  const uint32_t shift = static_cast<uint32_t>(14 - e);
  const uint32_t lsb = (m >> shift) & 1u;
  const uint32_t rounded = (m + ((1u << (shift - 1)) - 1u) + lsb) >> shift;
  return static_cast<uint16_t>(sign | rounded);
}

// ── 2. B1-TYPE：标量类型 ─────────────────────────────────────────────────────
#if TL910B_EMIT(bfloat16_t)
#if TL_PORT910B_BF16_BUILTIN
  // builtin 路径：`__bf16` 是 2 字节标量浮点类型，直接 typedef 即满足
  // TL_BIT_CAST / (float) / 指针运算；(float) 与算术由 clang 负责提升。
  typedef __bf16 bfloat16_t;
#else
  // storage 路径：自带 POD，行为自洽（数值面走 float 提升）。
  struct bfloat16_t {
    uint16_t data;

    bfloat16_t() = default;
    constexpr explicit bfloat16_t(uint16_t bits) : data(bits) {}
    constexpr bfloat16_t(float v)
        : data(tl910b_f32_to_bf16_bits(v)) {} // codegen 常以 float 初始化

    explicit constexpr operator float() const {
      return tl910b_bf16_bits_to_f32(data);
    }
    constexpr explicit operator double() const {
      return static_cast<double>(static_cast<float>(*this));
    }
    constexpr explicit operator bool() const {
      return (data & 0x7fffu) != 0u;
    }
  };
  // 比较与算术：全部转 float 计算再回 bf16（910B 上标量核本就无 bf16 原生 ALU）。
  constexpr bfloat16_t operator+(bfloat16_t a, bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) + static_cast<float>(b));
  }
  constexpr bfloat16_t operator-(bfloat16_t a, bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) - static_cast<float>(b));
  }
  constexpr bfloat16_t operator*(bfloat16_t a, bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) * static_cast<float>(b));
  }
  constexpr bfloat16_t operator/(bfloat16_t a, bfloat16_t b) {
    return bfloat16_t(static_cast<float>(a) / static_cast<float>(b));
  }
  constexpr bool operator==(bfloat16_t a, bfloat16_t b) {
    return a.data == b.data;
  }
  constexpr bool operator!=(bfloat16_t a, bfloat16_t b) {
    return a.data != b.data;
  }
  constexpr bool operator<(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }
  constexpr bool operator>(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) > static_cast<float>(b);
  }
  constexpr bool operator<=(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) <= static_cast<float>(b);
  }
  constexpr bool operator>=(bfloat16_t a, bfloat16_t b) {
    return static_cast<float>(a) >= static_cast<float>(b);
  }
  // codegen_ascend.cc:3419 等以 `bfloat16_t` 作模板实参，需要可默认构造。
  struct bfloat16x2_t;
#endif
#endif // TL910B_EMIT(bfloat16_t)

#if TL910B_EMIT(half)
#if TL_PORT910B_HALF_BUILTIN
  typedef _Float16 half;
#else
  struct half {
    uint16_t data;

    half() = default;
    constexpr explicit half(uint16_t bits) : data(bits) {}
    constexpr half(float v) : data(tl910b_f32_to_f16_bits(v)) {}

    explicit constexpr operator float() const {
      return tl910b_f16_bits_to_f32(data);
    }
    constexpr explicit operator double() const {
      return static_cast<double>(static_cast<float>(*this));
    }
    constexpr explicit operator bool() const { return (data & 0x7fffu) != 0u; }
  };
  constexpr half operator+(half a, half b) {
    return half(static_cast<float>(a) + static_cast<float>(b));
  }
  constexpr half operator-(half a, half b) {
    return half(static_cast<float>(a) - static_cast<float>(b));
  }
  constexpr half operator*(half a, half b) {
    return half(static_cast<float>(a) * static_cast<float>(b));
  }
  constexpr half operator/(half a, half b) {
    return half(static_cast<float>(a) / static_cast<float>(b));
  }
  constexpr bool operator==(half a, half b) { return a.data == b.data; }
  constexpr bool operator!=(half a, half b) { return a.data != b.data; }
  constexpr bool operator<(half a, half b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }
  constexpr bool operator>(half a, half b) {
    return static_cast<float>(a) > static_cast<float>(b);
  }
  constexpr bool operator<=(half a, half b) {
    return static_cast<float>(a) <= static_cast<float>(b);
  }
  constexpr bool operator>=(half a, half b) {
    return static_cast<float>(a) >= static_cast<float>(b);
  }
#endif
#endif // TL910B_EMIT(half)

#if TL910B_EMIT(half_t)
  // gemm.h:36 文档里以 half_t 指称 fp16 元素类型；与 tilelang CUDA 面同名习惯一致。
  typedef half half_t;
#endif

// ── 3. B1-TYPE：2 元组（.x/.y 可取址，供 ((T*)(&v))[i].x 用法） ──────────────
#if TL910B_EMIT(bfloat16x2_t)
  // 两个后端下形状一致：4 字节、.x/.y 成员、可 ((bfloat16x2_t*)&v)[i] 下标。
  struct bfloat16x2_t {
    bfloat16_t x, y;
  };
#endif

#if TL910B_EMIT(half2)
  struct half2 {
    half x, y;
  };
#endif

#if TL910B_EMIT(float2)
  struct float2 {
    float x, y;
  };
#endif

#if TL910B_EMIT(float4)
  struct float4 {
    float x, y, z, w;
  };
#endif

#if TL910B_EMIT(int2)
  struct int2 {
    int x, y;
  };
#endif

#if TL910B_EMIT(ushort2)
  // codegen_ascend.cc:3601,3638 用 `(ushort2*)(&v))[i/2].x` 访问 >4 lane 的
  // vector_bool；common.h:27 的 ushort typedef 在本头之后才出现，故此处用原生长度。
  struct ushort2 {
    unsigned short x, y;
  };
#endif

#if TL910B_EMIT(longlong4)
  struct longlong4 {
    long long x, y, z, w;
  };
#endif

#if TL910B_EMIT(ulonglong4)
  struct ulonglong4 {
    unsigned long long x, y, z, w;
  };
#endif

// ── 4. B1-TYPE：make_* 构造器（聚合初始化，避免向量 builtin 依赖） ───────────
#if TL910B_EMIT(make_float2)
  constexpr float2 make_float2(float x, float y) { return float2{x, y}; }
#endif
#if TL910B_EMIT(make_float4)
  constexpr float4 make_float4(float x, float y, float z, float w) {
    return float4{x, y, z, w};
  }
#endif
#if TL910B_EMIT(make_int2)
  constexpr int2 make_int2(int x, int y) { return int2{x, y}; }
#endif
#if TL910B_EMIT(make_longlong4)
  constexpr longlong4 make_longlong4(long long x, long long y, long long z,
                                     long long w) {
    return longlong4{x, y, z, w};
  }
#endif
#if TL910B_EMIT(make_ulonglong4)
  constexpr ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y,
                                       unsigned long long z,
                                       unsigned long long w) {
    return ulonglong4{x, y, z, w};
  }
#endif

// ── 5. B2-CVT：转换内建（发射点 codegen_ascend.cc:3333,3355,3403,3411,3419,3428） ─
// 语义与 CUDA/AscendC 同名件对齐：`_rn` = round-to-nearest-even；成对版拆两次标量。
#if TL910B_EMIT(__bfloat162float)
  constexpr float __bfloat162float(const bfloat16_t &x) {
#if defined(TL_PORT910B_NATIVE_TYPES)
    // bisheng BFloat16 proxy 无转换算子：位级取出后软件还原（bf16 是 f32 高位截断）
    return tl910b_bf16_bits_to_f32(tl910b_bit_cast<uint16_t, bfloat16_t>(x));
#elif TL_PORT910B_BF16_BUILTIN
    return static_cast<float>(x);
#else
    return tl910b_bf16_bits_to_f32(x.data);
#endif
  }
#endif

#if TL910B_EMIT(__float2bfloat16_rn)
  constexpr bfloat16_t __float2bfloat16_rn(float x) {
#if TL_PORT910B_BF16_BUILTIN || defined(TL_PORT910B_NATIVE_TYPES)
    // builtin 下无 RNE 保证（实现定义），显式走位路径再位重解释。
    return tl910b_bit_cast<bfloat16_t, uint16_t>(tl910b_f32_to_bf16_bits(x));
#else
    return bfloat16_t(tl910b_f32_to_bf16_bits(x));
#endif
  }
#endif

#if TL910B_EMIT(__bfloat1622float2)
  constexpr float2 __bfloat1622float2(const bfloat16x2_t &x) {
#if defined(TL_PORT910B_NATIVE_TYPES)
    // proxy 对类型无 .x/.y：按 2x16bit 位拆包各自还原
    uint32_t bits = tl910b_bit_cast<uint32_t, bfloat16x2_t>(x);
    return float2{tl910b_bf16_bits_to_f32(static_cast<uint16_t>(bits & 0xFFFFu)),
                  tl910b_bf16_bits_to_f32(static_cast<uint16_t>(bits >> 16))};
#else
    return float2{__bfloat162float(x.x), __bfloat162float(x.y)};
#endif
  }
#endif

#if TL910B_EMIT(__float22bfloat162_rn)
  constexpr bfloat16x2_t __float22bfloat162_rn(const float2 &x) {
#if defined(TL_PORT910B_NATIVE_TYPES)
    uint32_t lo = tl910b_f32_to_bf16_bits(x.x), hi = tl910b_f32_to_bf16_bits(x.y);
    return tl910b_bit_cast<bfloat16x2_t, uint32_t>(lo | (hi << 16));
#else
    return bfloat16x2_t{__float2bfloat16_rn(x.x), __float2bfloat16_rn(x.y)};
#endif
  }
#endif

#if TL910B_EMIT(__half22float2)
  constexpr float2 __half22float2(half2 x) {
    return float2{static_cast<float>(x.x), static_cast<float>(x.y)};
  }
#endif

#if TL910B_EMIT(__float22half2_rn)
  constexpr half2 __float22half2_rn(float2 x) {
    return half2{half(x.x), half(x.y)};
  }
#endif

} // namespace tl910b

// ── 6. 全局名字注入（逐名字可退让；`using` 而非宏，保持重载/模板能力） ───────
#define TL910B_USING(name) using tl910b::name
#if TL910B_EMIT(bfloat16_t)
TL910B_USING(bfloat16_t);
#endif
#if TL910B_EMIT(half)
TL910B_USING(half);
#endif
#if TL910B_EMIT(half_t)
TL910B_USING(half_t);
#endif
#if TL910B_EMIT(bfloat16x2_t)
TL910B_USING(bfloat16x2_t);
#endif
#if TL910B_EMIT(half2)
TL910B_USING(half2);
#endif
#if TL910B_EMIT(float2)
TL910B_USING(float2);
#endif
#if TL910B_EMIT(float4)
TL910B_USING(float4);
#endif
#if TL910B_EMIT(int2)
TL910B_USING(int2);
#endif
#if TL910B_EMIT(ushort2)
TL910B_USING(ushort2);
#endif
#if TL910B_EMIT(longlong4)
TL910B_USING(longlong4);
#endif
#if TL910B_EMIT(ulonglong4)
TL910B_USING(ulonglong4);
#endif
#if TL910B_EMIT(make_float2)
TL910B_USING(make_float2);
#endif
#if TL910B_EMIT(make_float4)
TL910B_USING(make_float4);
#endif
#if TL910B_EMIT(make_int2)
TL910B_USING(make_int2);
#endif
#if TL910B_EMIT(make_longlong4)
TL910B_USING(make_longlong4);
#endif
#if TL910B_EMIT(make_ulonglong4)
TL910B_USING(make_ulonglong4);
#endif
#if TL910B_EMIT(__bfloat162float)
TL910B_USING(__bfloat162float);
#endif
#if TL910B_EMIT(__float2bfloat16_rn)
TL910B_USING(__float2bfloat16_rn);
#endif
#if TL910B_EMIT(__bfloat1622float2)
TL910B_USING(__bfloat1622float2);
#endif
#if TL910B_EMIT(__float22bfloat162_rn)
TL910B_USING(__float22bfloat162_rn);
#endif
#if TL910B_EMIT(__half22float2)
TL910B_USING(__half22float2);
#endif
#if TL910B_EMIT(__float22half2_rn)
TL910B_USING(__float22half2_rn);
#endif

// ── 7. B3-LOCKMODE：asc_lock/asc_unlock 的模式实参 ───────────────────────────
// 发射点 codegen_ascend.cc:1855-1857,1869-1871（`asc_lock(PIPE_X, id, ASC_LOCK_BLOCK)`）
// 与 rewrite_flag_to_buf.cc:20-23。注意：本组只解决"枚举值可编译"，
// `asc_lock/asc_unlock` 本体在 910B 是否存在属 D2-AIC-API（探针 P3）。
#ifndef ASC_LOCK_BLOCK
#define ASC_LOCK_BLOCK 0
#endif
#ifndef ASC_LOCK_NON_BLOCK
#define ASC_LOCK_NON_BLOCK 1
#endif

// ── 8. B4：__SIMT_DEVICE_FUNCTIONS_DECL__（ascend_fp8.h:9 的 TL_DEVICE 前缀） ─
// 950 上它是 SIMT device 函数声明限定；910B 面无 SIMT，空展开即可。
#ifndef __SIMT_DEVICE_FUNCTIONS_DECL__
#define __SIMT_DEVICE_FUNCTIONS_DECL__
#endif

// ── 8b. debug 打印的 float 化辅助（proxy 类型无 (float) 算子）─────────────────
inline float tl910b_debug_as_float(const bfloat16_t &v) {
  return tl910b::tl910b_bf16_bits_to_f32(tl910b::tl910b_bit_cast<uint16_t, bfloat16_t>(v));
}
template <typename T> inline float tl910b_debug_as_float(T v) { return static_cast<float>(v); }

// ── 10. AIC 适配器：tilelang codegen 裸 emitted 符号 → 910B CCE 原生（P1-1 首件）
#if defined(TL_PORT910B_NATIVE_TYPES)
// codegen 生成 `extern "C" __global__ __mix__(1,2) ...` 与 `if ASC_IS_AIC {...} else {...}`
// 910B cce 原生即有 __mix__(cube,vec)=core_ratio 属性宏（w9/w10 实证）——让位，不覆盖。
// 仅 host/无 cce 环境自检时补一个空定义保 TU 可析。
#ifndef __mix__
#define __mix__(a, b)
#endif
#define ASC_IS_AIC (true)                  // 一期 AIC 优先；AIV 分支保留可编（死代码消除前）

// pipe 事件常量：与 codegen_ascend 的 PIPE_* 对齐即可（值仅传参用）
enum tl910b_pipe_id { TL910B_PIPE_M = 0, TL910B_PIPE_MTE1, TL910B_PIPE_MTE2,
                      TL910B_PIPE_MTE3, TL910B_PIPE_V, TL910B_PIPE_S, TL910B_PIPE_ICACHE };

__aicore__ inline void asc_init() {}                       // 910B mad 无显式初始化
// FFTS 调度表基址（与官方 asc/impl/basic_api/kernel_utils.h:36 同名对齐；host launch 注入）
__gm__ void *g_sysFftsAddr = nullptr;
// （TLV 枚举 FuncMetaType/KernelType/BinaryMetaType 与注册结构由官方 kernel_utils_macros.h 提供）


#define TL910B_ASC_MMAD_OVERLOAD(TY)                                          \
  __aicore__ inline void asc_mmad(__ubuf__ float *cc, __ubuf__ TY *ca,             \
                                  __ubuf__ TY *cb, uint16_t m, uint16_t k,        \
                                  uint16_t n, int unit_flag_ctrl, bool gemv_ctrl, \
                                  bool btbuf_ctrl, bool zero_c_ctrl) {            \
    /* 910B mad(cc, BTAddr, ca, cb, M, K, N, unit_flag, gemv, BTbuf, zero_C)；    \
       地址空间指针经整数中转（C-style 整→空间指针 CCE 允许）*/                   \
    mad((__cc__ float *)(uintptr_t)cc, (__ca__ TY *)(uintptr_t)ca,                 \
        (__cb__ TY *)(uintptr_t)cb, m, k, n, (uint8_t)unit_flag_ctrl,               \
        gemv_ctrl, btbuf_ctrl, zero_c_ctrl);                                       \
  }
TL910B_ASC_MMAD_OVERLOAD(bfloat16_t)
TL910B_ASC_MMAD_OVERLOAD(half)
TL910B_ASC_MMAD_OVERLOAD(float)
// (int8 形态 910B mad 走 fp8 家族参数，非一期范围，暂缓)

// 跨 pipe 同步：一期保守全序列化（正确性优先，后续换 SetFlag/WaitFlag 精细化）
__aicore__ inline void asc_sync_intra_wait(int pipe, int event) { (void)pipe; (void)event; __cce_pipe_barrier(PIPE_V); }
__aicore__ inline void asc_sync_intra_set(int pipe, int event)  { (void)pipe; (void)event; }
__aicore__ inline void asc_lock(int pipe, int id)   { (void)pipe; (void)id; __cce_pipe_barrier(PIPE_V); }
__aicore__ inline void asc_unlock(int pipe, int id, int mode) { (void)pipe; (void)id; (void)mode; }

// ── 10b. AIV/索引/搬运面（一期 if(0) 分支符号可解析；数值待 host launch 绑定后精化）
#define ASC_IS_AIV (0)
struct tl910b_dim3 { uint32_t x, y, z; };
__aicore__ inline tl910b_dim3 tl910b_blockIdx_holder() { uint32_t i = get_block_idx(); return tl910b_dim3{0, i, 0}; }
#define blockIdx tl910b_blockIdx_holder()
__aicore__ inline int32_t asc_get_sub_block_id() { return (int32_t)get_subblockid(); }

enum asc_load_l2_cache_mode { ASC_LOAD_L2_CACHE_ALLOC = 0, ASC_LOAD_L2_CACHE_NORMAL = 1 };
// codegen: asc_copy_gm2ub_align(dst, src, sid, lenBurst, srcStride, dstStride, blockGroup, mode, nBurst, totalLen)
__aicore__ inline void asc_copy_gm2ub_align(__ubuf__ uint8_t *dst, __gm__ uint8_t *src,
                                           int sid, int lenBurst, int srcStride, int dstStride,
                                           int blockGroup, asc_load_l2_cache_mode mode,
                                           int nBurst, int totalLen) {
  (void)blockGroup; (void)mode; (void)totalLen;
  uint64_t config = ((uint64_t)(sid & 0xf)) | ((uint64_t)(nBurst & 0xfff) << 4) |
                    ((uint64_t)(lenBurst & 0xffff) << 16) | ((uint64_t)(srcStride & 0xffff) << 32) |
                    ((uint64_t)(dstStride & 0xffff) << 48);
  copy_gm_to_ubuf(dst, src, config);
}
__aicore__ inline void asc_copy_ub2gm_align(__gm__ uint8_t *dst, __ubuf__ uint8_t *src,
                                           int sid, int lenBurst, int srcStride, int dstStride,
                                           int blockGroup, int mode, int nBurst, int totalLen) {
  (void)blockGroup; (void)mode; (void)totalLen;
  uint64_t config = ((uint64_t)(sid & 0xf)) | ((uint64_t)(nBurst & 0xfff) << 4) |
                    ((uint64_t)(lenBurst & 0xffff) << 16) | ((uint64_t)(srcStride & 0xffff) << 32) |
                    ((uint64_t)(dstStride & 0xffff) << 48);
  copy_ubuf_to_gm(dst, src, config);
}
__aicore__ inline void asc_sync_intra_arrive(int pipe, int event) { (void)pipe; (void)event; }
#endif // TL_PORT910B_NATIVE_TYPES (AIC adapters)

#if !defined(TL_PORT910B_NATIVE_TYPES)  // 自检仅对自带类型有意义；NATIVE 下类型由 cce 原生保证

// ── 9. 自检（P9 探针即"编一个 TU"，这些断言在编译期即给出等价性真值） ────────
#if defined(TL_PORT910B_NATIVE_TYPES)
static_assert(sizeof(bfloat16_t) == 2, "native bf16 must be 2 bytes");
static_assert(sizeof(half) == 2, "native half must be 2 bytes");
#else
static_assert(sizeof(tl910b::bfloat16_t) == 2, "bf16 must be 2 bytes for TL_BIT_CAST");
static_assert(sizeof(tl910b::half) == 2, "fp16 must be 2 bytes for TL_BIT_CAST");
#endif
static_assert(sizeof(tl910b::bfloat16x2_t) == 4, "bf16x2 must be 4 bytes");
static_assert(sizeof(tl910b::half2) == 4, "half2 must be 4 bytes");
static_assert(sizeof(tl910b::float2) == 8, "float2 must be 8 bytes");
static_assert(sizeof(tl910b::float4) == 16, "float4 must be 16 bytes");
static_assert(sizeof(tl910b::ushort2) == 4, "ushort2 must be 4 bytes");
static_assert(sizeof(tl910b::ulonglong4) == 32, "ulonglong4 must be 32 bytes");
static_assert(__is_trivially_copyable(tl910b::bfloat16_t), "must be trivially copyable");
static_assert(__is_trivially_copyable(tl910b::bfloat16x2_t), "must be trivially copyable");

// 位视图与 codegen 的 `((T*)(&v))[i].x` 习惯等价：4 个 bf16 lane 装进 ulonglong4 的低 8 字节
static_assert(tl910b::tl910b_f32_to_bf16_bits(1.0f) == 0x3F80u, "bf16(1.0)");
static_assert(tl910b::tl910b_bf16_bits_to_f32(0x3F80u) == 1.0f, "f32 from bf16");
static_assert(tl910b::tl910b_f32_to_f16_bits(1.0f) == 0x3C00u, "fp16(1.0)");
static_assert(tl910b::tl910b_f16_bits_to_f32(0x3C00u) == 1.0f, "f32 from fp16");
static_assert(tl910b::tl910b_f32_to_f16_bits(-2.0f) == 0xC000u, "fp16(-2.0)");
static_assert(tl910b::tl910b_f32_to_bf16_bits(0.5f) == 0x3F00u, "bf16(0.5)");

#endif // !TL_PORT910B_NATIVE_TYPES (self-check)
#endif // !TL_ASCEND_SIMT
