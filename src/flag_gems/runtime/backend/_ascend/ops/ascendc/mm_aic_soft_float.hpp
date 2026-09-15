// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>
#ifndef SF_INLINE
#define SF_INLINE inline
#endif
namespace SoftFloat {
SF_INLINE uint64_t RoundShift(uint64_t x, int s) {
  if (s <= 0) return x << (-s);
  if (s >= 64) return 0;
  uint64_t q = x >> s, rem = x & ((uint64_t(1) << s) - 1), half = uint64_t(1) << (s - 1);
  return q + (rem > half || (rem == half && (q & 1)));
}
SF_INLINE uint32_t FromInt(int32_t x) {
  if (x == 0) return 0;
  uint32_t sign = x < 0 ? 0x80000000u : 0, a = x < 0 ? 0u - uint32_t(x) : uint32_t(x);
  int e = 31 - __builtin_clz(a);
  uint32_t m = e <= 23 ? a << (23 - e) : uint32_t(RoundShift(a, e - 23));
  if (m == 0x1000000u) {
    m >>= 1;
    ++e;
  }
  return sign | uint32_t(e + 127) << 23 | (m & 0x7fffffu);
}
SF_INLINE uint32_t Mul(uint32_t a, uint32_t b) {
  uint32_t sign = (a ^ b) & 0x80000000u, ma = a & 0x7fffffu, mb = b & 0x7fffffu;
  int ea = (a >> 23) & 255, eb = (b >> 23) & 255;
  if (ea == 255 || eb == 255) {
    if ((ea == 255 && ma) || (eb == 255 && mb) || ((a & 0x7fffffffu) == 0) || ((b & 0x7fffffffu) == 0))
      return 0x7fc00000u;
    return sign | 0x7f800000u;
  }
  if ((a & 0x7fffffffu) == 0 || (b & 0x7fffffffu) == 0) return sign;
  if (ea) {
    ma |= 0x800000u;
    ea -= 127;
  } else {
    int s = __builtin_clz(ma) - 8;
    ma <<= s;
    ea = -126 - s;
  }
  if (eb) {
    mb |= 0x800000u;
    eb -= 127;
  } else {
    int s = __builtin_clz(mb) - 8;
    mb <<= s;
    eb = -126 - s;
  }
  uint64_t prod = uint64_t(ma) * mb;
  int high = (prod >> 47) != 0, e = ea + eb + high, shift = 23 + high;
  if (e > 127) return sign | 0x7f800000u;
  if (e < -126) return sign | uint32_t(RoundShift(prod, shift - 126 - e));
  uint32_t m = uint32_t(RoundShift(prod, shift));
  if (m == 0x1000000u) {
    m >>= 1;
    ++e;
  }
  if (e > 127) return sign | 0x7f800000u;
  return sign | uint32_t(e + 127) << 23 | (m & 0x7fffffu);
}
SF_INLINE uint16_t ToBfloat(uint32_t x) {
  if ((x & 0x7f800000u) == 0x7f800000u && (x & 0x7fffffu)) return uint16_t(x >> 16) | 0x40;
  return uint16_t((x + 0x7fffu + ((x >> 16) & 1)) >> 16);
}
SF_INLINE uint16_t ToHalf(uint32_t x) {
  uint16_t sign = (x >> 16) & 0x8000u;
  int e = (x >> 23) & 255;
  uint32_t m = x & 0x7fffffu;
  if (e == 255) return sign | (m ? 0x7e00u : 0x7c00u);
  if (e == 0) return sign;
  e -= 127;
  m |= 0x800000u;
  if (e > 15) return sign | 0x7c00u;
  if (e < -14) return sign | uint16_t(RoundShift(m, 13 - 14 - e));
  uint32_t q = uint32_t(RoundShift(m, 13));
  if (q == 2048) {
    q >>= 1;
    ++e;
  }
  if (e > 15) return sign | 0x7c00u;
  return sign | uint16_t((e + 15) << 10) | (q & 1023u);
}
}  // namespace SoftFloat
