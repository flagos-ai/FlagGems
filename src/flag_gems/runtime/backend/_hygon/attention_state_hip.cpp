// Copyright 2026, The FlagOS Contributors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Exact STATE BF16 QKV specialization, gfx936 only. QK/PV use MMAC with
// FP32 accumulators. A Cauchy bound selects a BF16-normal fixed-shift
// softmax; flagged tiles are recomputed by the Triton stable kernel.
// Reuse K/V LDS for the epilogue to coalesce output into 128-bit stores.
#include <hip/hip_runtime.h>
#include <cstdint>

using u2 = uint32_t __attribute__((ext_vector_type(2)));
using u4 = uint32_t __attribute__((ext_vector_type(4)));
using f4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ f4 dot(u2 a, u2 b, f4 c) {
  asm("v_mmac_f32_16x16x16_bf16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
  return c;
}
__device__ __forceinline__ float bf(uint16_t v) {
  return __uint_as_float(uint32_t(v) << 16);
}
__device__ __forceinline__ uint16_t rb(float x) {
  uint32_t z = __float_as_uint(x);
  return (z + 0x7fff + ((z >> 16) & 1)) >> 16;
}
__device__ __forceinline__ u2 ld(const uint16_t *p) {
  return *reinterpret_cast<const u2 *>(p);
}
__device__ __forceinline__ u2 pk(uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
  return {uint32_t(a) | (uint32_t(b) << 16), uint32_t(c) | (uint32_t(d) << 16)};
}

__device__ __forceinline__ u2 read64(const uint16_t *p) {
  uint32_t addr = uint32_t(uintptr_t(p));
  u2 v;
  asm volatile("ds_read_b64 %0, %1" : "=v"(v) : "v"(addr) : "memory");
  return v;
}
__device__ __forceinline__ void hold_scores(f4 &s, u2 a, u2 b, u2 c, u2 d, u2 e, u2 f, u2 g, u2 h) {
  asm volatile("" : "+v"(s) : "v"(a), "v"(b), "v"(c), "v"(d), "v"(e), "v"(f), "v"(g), "v"(h) : "memory");
}
__device__ __forceinline__ void wait_values(u2 &a, u2 &b, u2 &c, u2 &d, u2 &e, u2 &f, u2 &g, u2 &h) {
  asm volatile("s_waitcnt lgkmcnt(0)"
               : "+v"(a), "+v"(b), "+v"(c), "+v"(d), "+v"(e), "+v"(f), "+v"(g), "+v"(h)
               :
               : "memory");
}

__device__ __forceinline__ void wait4(u2 &a, u2 &b, u2 &c, u2 &d) {
  asm volatile("s_waitcnt lgkmcnt(0)" : "+v"(a), "+v"(b), "+v"(c), "+v"(d) : : "memory");
}
__device__ __forceinline__ u4 ld4(const uint16_t *p) {
  return *reinterpret_cast<const u4 *>(p);
}

template <int WARPS, int QT, int BN, int MODE, int KC, int STORE>
__global__ __launch_bounds__(WARPS * 64) void flash_vpackedlds(const uint16_t *__restrict__ Q,
                                                               const uint16_t *__restrict__ K,
                                                               const uint16_t *__restrict__ V,
                                                               const float *__restrict__ T,
                                                               uint8_t *__restrict__ F,
                                                               uint16_t *__restrict__ O,
                                                               float *__restrict__ L) {
  constexpr int M = 2049, D = 128, H = 16, BM = WARPS * QT * 16, KT = BN / 16;
  const int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, row = lane % 16, group = lane / 16;
  const int pb = blockIdx.z, ph = blockIdx.y, qb = blockIdx.x * BM + wave * (QT * 16);
  const int64_t base = int64_t(pb) * 12589056 + ph * 128;
  __shared__ uint16_t smem[(STORE == 2 ? WARPS * 16 * 136 : BN * (128 + 130))];
  __shared__ float bounds[WARPS];
  auto sk = smem;
  auto sv = smem + BN * D;
  float kmax = 0;
  for (int i = 0; i < 17; ++i) kmax = fmaxf(kmax, T[(pb * H + ph) * 17 + i]);
  u2 q[QT][8];
  float shift[QT], denom[QT] = {};
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    float norm = 0;
    int qr = qb + qt * 16 + row;
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      q[qt][dt] = qr < M ? ld(Q + base + int64_t(qr) * 6144 + dt * 16 + group * 4) : u2 {0, 0};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float v = bf(uint16_t(q[qt][dt][j / 2] >> (16 * (j % 2))));
        norm += v * v;
      }
    }
    norm += __shfl_xor(norm, 16, 64);
    norm += __shfl_xor(norm, 32, 64);
    shift[qt] = sqrtf(norm * kmax) * 0.12751743082455375f * 1.00001f;
  }
  float bound = 0;
#pragma unroll
  for (int qi = 0; qi < QT; ++qi) {
    // fmaxf discards NaNs. In particular, zero Q times an overflowing K
    // norm yields a NaN shift and must select stable recomputation.
    // Inspect exponent bits because this file is built with -ffast-math.
    bool finite = (__float_as_uint(shift[qi]) & 0x7f800000u) != 0x7f800000u;
    bound = fmaxf(bound, finite ? shift[qi] : __uint_as_float(0x7f800000u));
  }
#pragma unroll
  for (int mask = 32; mask; mask /= 2) bound = fmaxf(bound, __shfl_xor(bound, mask, 64));
  if (lane == 0) bounds[wave] = bound;
  __syncthreads();
  float block_bound = 0;
#pragma unroll
  for (int wi = 0; wi < WARPS; ++wi) block_bound = fmaxf(block_bound, bounds[wi]);
  bool good = block_bound <= 55.f;
  if (tid == 0) F[(pb * H + ph) * ((M + BM - 1) / BM) + blockIdx.x] = good;
  if (!good) return;
  f4 acc[QT][8];
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
    for (int dv = 0; dv < 8; ++dv) acc[qt][dv] = {0, 0, 0, 0};
  }
#pragma unroll
  for (int j = 0; j < BN * 128 / (WARPS * 64 * 8); ++j) {
    int idx = tid * 8 + j * (WARPS * 64) * 8, kr = idx / 128, dd = idx % 128;
    u4 kv = kr < M ? ld4(K + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
    u4 vv = kr < M ? ld4(V + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
    for (int hp = 0; hp < 2; ++hp)
      *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 + ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
          u2 {kv[hp * 2], kv[hp * 2 + 1]};
#pragma unroll
    for (int jp = 0; jp < 8; ++jp) {
      int vr = idx / 128, vc = dd + jp;
      sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 + (vr % 16) / 4] =
          uint16_t(vv[jp / 2] >> (16 * (jp % 2)));
    }
  }
  __syncthreads();
  for (int nb = 0; nb < 2048; nb += BN) {
    constexpr int LOADS = BN * 128 / (WARPS * 64 * 8);
    u4 next_k[LOADS], next_v[LOADS];
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * (WARPS * 64) * 8, kr = nb + BN + idx / 128, dd = idx % 128;
      if (nb + BN < 2048) {
        next_k[j] = ld4(K + base + int64_t(kr) * 6144 + dd);
        next_v[j] = ld4(V + base + int64_t(kr) * 6144 + dd);
      } else {
        next_k[j] = kr < M ? ld4(K + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
        next_v[j] = kr < M ? ld4(V + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
      }
    }
    if (qb < M) {
      f4 scores[QT][KT];
#pragma unroll
      for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) scores[qt][nk] = {0, 0, 0, 0};
      }
#pragma unroll
      for (int db = 0; db < 8; db += KC) {
        u2 keys[KC][KT];
#pragma unroll
        for (int dk = 0; dk < KC; ++dk) {
#pragma unroll
          for (int nk = 0; nk < KT; ++nk)
            keys[dk][nk] =
                read64(sk + (nk * 16 + row) * 128 + (((db + dk) * 16 + group * 4) ^ ((row % 16) * 4)));
        }
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) {
          if (KC == 8)
            wait_values(keys[0][nk],
                        keys[1][nk],
                        keys[2][nk],
                        keys[3][nk],
                        keys[4][nk],
                        keys[5][nk],
                        keys[6][nk],
                        keys[7][nk]);
          else
            wait4(keys[0][nk], keys[1][nk], keys[2][nk], keys[3][nk]);
        }
#pragma unroll
        for (int dk = 0; dk < KC; ++dk) {
#pragma unroll
          for (int nk = 0; nk < KT; ++nk) {
#pragma unroll
            for (int qt = 0; qt < QT; ++qt)
              scores[qt][nk] = dot(q[qt][db + dk], keys[dk][nk], scores[qt][nk]);
          }
        }
      }
#pragma unroll
      for (int nk = 0; nk < KT; ++nk) {
        u2 values[8];
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) values[dv] = read64(sv + (nk * 8 * 65 + dv * 65 + lane) * 4);
        if (MODE) {
#pragma unroll
          for (int qt = 0; qt < QT; ++qt)
            hold_scores(scores[qt][nk],
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7]);
        }
        u2 pp[QT];
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
          uint32_t halves[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float p = __builtin_amdgcn_exp2f(scores[qt][nk][j] * 0.12751743082455375f -
                                             (MODE >= 3 ? 0.f : shift[qt]));
            denom[qt] += p;
            uint32_t z = __float_as_uint(p);
            halves[j] = (MODE & 1) ? z + 0x7fff + ((z >> 16) & 1) : z + 0x8000;
          }
          pp[qt] = {__builtin_amdgcn_perm(halves[1], halves[0], 0x07060302),
                    __builtin_amdgcn_perm(halves[3], halves[2], 0x07060302)};
        }
        wait_values(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) {
          u2 vv = values[dv];
#pragma unroll
          for (int qt = 0; qt < QT; ++qt) acc[qt][dv] = dot(pp[qt], vv, acc[qt][dv]);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * (WARPS * 64) * 8, dd = idx % 128;
      for (int hp = 0; hp < 2; ++hp)
        *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 + ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
            u2 {next_k[j][hp * 2], next_k[j][hp * 2 + 1]};
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        int vr = idx / 128, vc = dd + jp;
        sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 + (vr % 16) / 4] =
            uint16_t(next_v[j][jp / 2] >> (16 * (jp % 2)));
      }
    }
    __syncthreads();
  }
  {
    constexpr int nb = 2048;
    constexpr int LOADS = BN * 128 / (WARPS * 64 * 8);
    u4 next_k[LOADS], next_v[LOADS];
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * (WARPS * 64) * 8, kr = nb + BN + idx / 128, dd = idx % 128;
      next_k[j] = u4 {0, 0, 0, 0};
      next_v[j] = u4 {0, 0, 0, 0};
    }
    if (qb < M) {
      f4 scores[QT][KT];
#pragma unroll
      for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) scores[qt][nk] = {0, 0, 0, 0};
      }
#pragma unroll
      for (int db = 0; db < 8; db += KC) {
        u2 keys[KC][KT];
#pragma unroll
        for (int dk = 0; dk < KC; ++dk) {
#pragma unroll
          for (int nk = 0; nk < KT; ++nk)
            keys[dk][nk] =
                read64(sk + (nk * 16 + row) * 128 + (((db + dk) * 16 + group * 4) ^ ((row % 16) * 4)));
        }
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) {
          if (KC == 8)
            wait_values(keys[0][nk],
                        keys[1][nk],
                        keys[2][nk],
                        keys[3][nk],
                        keys[4][nk],
                        keys[5][nk],
                        keys[6][nk],
                        keys[7][nk]);
          else
            wait4(keys[0][nk], keys[1][nk], keys[2][nk], keys[3][nk]);
        }
#pragma unroll
        for (int dk = 0; dk < KC; ++dk) {
#pragma unroll
          for (int nk = 0; nk < KT; ++nk) {
#pragma unroll
            for (int qt = 0; qt < QT; ++qt)
              scores[qt][nk] = dot(q[qt][db + dk], keys[dk][nk], scores[qt][nk]);
          }
        }
      }
#pragma unroll
      for (int nk = 0; nk < KT; ++nk) {
        u2 values[8];
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) values[dv] = read64(sv + (nk * 8 * 65 + dv * 65 + lane) * 4);
        if (MODE) {
#pragma unroll
          for (int qt = 0; qt < QT; ++qt)
            hold_scores(scores[qt][nk],
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                        values[4],
                        values[5],
                        values[6],
                        values[7]);
        }
        u2 pp[QT];
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
          uint32_t halves[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float p = nb + nk * 16 + group + j * 4 < M
                          ? __builtin_amdgcn_exp2f(scores[qt][nk][j] * 0.12751743082455375f -
                                                   (MODE >= 3 ? 0.f : shift[qt]))
                          : 0.f;
            denom[qt] += p;
            uint32_t z = __float_as_uint(p);
            halves[j] = (MODE & 1) ? z + 0x7fff + ((z >> 16) & 1) : z + 0x8000;
          }
          pp[qt] = {__builtin_amdgcn_perm(halves[1], halves[0], 0x07060302),
                    __builtin_amdgcn_perm(halves[3], halves[2], 0x07060302)};
        }
        wait_values(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) {
          u2 vv = values[dv];
#pragma unroll
          for (int qt = 0; qt < QT; ++qt) acc[qt][dv] = dot(pp[qt], vv, acc[qt][dv]);
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * (WARPS * 64) * 8, dd = idx % 128;
      for (int hp = 0; hp < 2; ++hp)
        *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 + ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
            u2 {next_k[j][hp * 2], next_k[j][hp * 2 + 1]};
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        int vr = idx / 128, vc = dd + jp;
        sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 + (vr % 16) / 4] =
            uint16_t(next_v[j][jp / 2] >> (16 * (jp % 2)));
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    float den = denom[qt];
    den += __shfl_xor(den, 16, 64);
    den += __shfl_xor(den, 32, 64);
    int qr = qb + qt * 16 + row;
    if (qr < M) {
#pragma unroll
      for (int dv = 0; dv < 8; ++dv) {
        if (STORE == 1) {
          uint32_t p0 = uint32_t(rb(acc[qt][dv][0] / den)) | (uint32_t(rb(acc[qt][dv][1] / den)) << 16);
          uint32_t p1 = uint32_t(rb(acc[qt][dv][2] / den)) | (uint32_t(rb(acc[qt][dv][3] / den)) << 16);
          uint32_t o0 = __shfl_xor(p0, 16, 64), o1 = __shfl_xor(p1, 16, 64);
          uint32_t r0 = (group & 1) ? ((o0 >> 16) | (p0 & 0xffff0000u)) : ((p0 & 0xffffu) | (o0 << 16));
          uint32_t r1 = (group & 1) ? ((o1 >> 16) | (p1 & 0xffff0000u)) : ((p1 & 0xffffu) | (o1 << 16));
          uint32_t s0 = __shfl_xor(r0, 32, 64), s1 = __shfl_xor(r1, 32, 64);
          u2 vec = (group & 2) ? u2 {s1, r1} : u2 {r0, s0};
          *reinterpret_cast<u2 *>(O + ((pb * H + ph) * M + qr) * D + dv * 16 + group * 4) = vec;
        } else {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            uint16_t val = rb(acc[qt][dv][j] / den);
            if (STORE == 2)
              smem[(wave * 16 + row) * 136 + dv * 16 + group + j * 4] = val;
            else
              O[((pb * H + ph) * M + qr) * D + dv * 16 + group + j * 4] = val;
          }
        }
      }
      if (group == 0)
        L[(pb * H + ph) * M + qr] = ((MODE >= 3 ? 0.f : shift[qt]) + log2f(den)) * 0.6931471805599453f;
    }
    if (STORE == 2) {
      __syncthreads();
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        int idx = tid * 8 + j * WARPS * 64 * 8;
        if (idx < WARPS * 16 * 128) {
          int rr = idx / 128, cc = idx % 128;
          int oq = blockIdx.x * BM + (rr / 16) * (QT * 16) + qt * 16 + rr % 16;
          if (oq < M)
            *reinterpret_cast<u4 *>(O + ((pb * H + ph) * M + oq) * 128 + cc) = ld4(smem + rr * 136 + cc);
        }
      }
      __syncthreads();
    }
  }
}

extern "C" int launch_state_flash(const void *q,
                                  const void *k,
                                  const void *v,
                                  const float *t,
                                  uint8_t *f,
                                  void *o,
                                  float *l,
                                  hipStream_t stream) {
  hipLaunchKernelGGL((flash_vpackedlds<4, 3, 16, 1, 8, 2>),
                     dim3(11, 16, 8),
                     dim3(256),
                     0,
                     stream,
                     static_cast<const uint16_t *>(q),
                     static_cast<const uint16_t *>(k),
                     static_cast<const uint16_t *>(v),
                     t,
                     f,
                     static_cast<uint16_t *>(o),
                     l);
  return hipGetLastError();
}

__device__ __forceinline__ uint16_t round_fp16(float x) {
  return __builtin_bit_cast(uint16_t, (_Float16)x);
}
__device__ __forceinline__ float fp16_value(uint16_t x) {
  return (float)__builtin_bit_cast(_Float16, x);
}
__device__ __forceinline__ f4 dot_fp16(u2 a, u2 b, f4 c) {
  asm("v_mmac_f32_16x16x16_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
  return c;
}
__device__ __forceinline__ u2 convert_v_fp16(u2 a) {
  u2 b;
  float x0 = bf(uint16_t(a[0])), x1 = bf(uint16_t(a[0] >> 16)), x2 = bf(uint16_t(a[1])),
        x3 = bf(uint16_t(a[1] >> 16));
  asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(b[0]) : "v"(x0), "v"(x1));
  asm("v_cvt_pkrtz_f16_f32 %0, %1, %2" : "=v"(b[1]) : "v"(x2), "v"(x3));
  return b;
}
// The fast path keeps the FP32 denominator and measures the FP16 probability
// rounding error. Its block bound includes the original BF16 V magnitude,
// conversion error, and a numerical margin. Rejected blocks use the stable path.
template <int W, int QT, int BN>
__device__ __forceinline__ bool try_short_fp16(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O, float *L, uint8_t *F) {
  constexpr int N = 362, H = 16, D = 128, BM = W * QT * 16, KT = BN / 16, LOADS = BN * 128 / (W * 64 * 8);
  int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, row = lane % 16, group = lane / 16;
  int head = blockIdx.y, qb = blockIdx.x * BM + wave * QT * 16;
  int64_t base = int64_t(head / 16) * (N * 6144) + (head % 16) * 128;
  __shared__ uint16_t smem[BN * (128 + 130)];
  __shared__ uint32_t bounds[W];
  __shared__ float errors[W];
  uint32_t vmaxbits = 0;
  uint32_t unsafe = 0;
  auto sk = smem;
  auto sv = smem + BN * 128;
  u2 q[QT][8];
  f4 acc[QT][8];
  float denom[QT] = {}, mx[QT], err[QT] = {};
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    mx[qt] = -__uint_as_float(0x7f800000u);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      q[qt][j] = qb + qt * 16 + row < N
                     ? ld(Q + base + int64_t(qb + qt * 16 + row) * 6144 + j * 16 + group * 4)
                     : u2 {0, 0};
      acc[qt][j] = {0, 0, 0, 0};
    }
  }
#pragma unroll
  for (int j = 0; j < LOADS; ++j) {
    int idx = tid * 8 + j * W * 64 * 8, kr = idx / 128, dd = idx % 128;
    u4 kv = kr < N ? ld4(K + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
    u4 vv = kr < N ? ld4(V + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};

#pragma unroll
    for (int z = 0; z < 4; ++z) {
      vmaxbits = max(vmaxbits, vv[z] & 0x7fffu);
      vmaxbits = max(vmaxbits, (vv[z] >> 16) & 0x7fffu);
    }
    u2 va = convert_v_fp16(u2 {vv[0], vv[1]}), vb = convert_v_fp16(u2 {vv[2], vv[3]});
    vv = u4 {va[0], va[1], vb[0], vb[1]};
#pragma unroll
    for (int hp = 0; hp < 2; ++hp)
      *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 + ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
          u2 {kv[hp * 2], kv[hp * 2 + 1]};
#pragma unroll
    for (int jp = 0; jp < 8; ++jp) {
      int vr = idx / 128, vc = dd + jp;
      sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 + (vr % 16) / 4] =
          uint16_t(vv[jp / 2] >> (16 * (jp % 2)));
    }
  }
  __syncthreads();
  for (int nb = 0; nb < N; nb += BN) {
    u4 next_k[LOADS], next_v[LOADS];
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * W * 64 * 8, kr = nb + BN + idx / 128, dd = idx % 128;
      next_k[j] = kr < N ? ld4(K + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
      next_v[j] = kr < N ? ld4(V + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
    }

    if (qb < N) {
      f4 scores[QT][KT];
#pragma unroll
      for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) scores[qt][nk] = {0, 0, 0, 0};
      }
      u2 keys[8][KT];
#pragma unroll
      for (int db = 0; db < 8; ++db) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk)
          keys[db][nk] = read64(sk + (nk * 16 + row) * 128 + ((db * 16 + group * 4) ^ ((row % 16) * 4)));
      }
#pragma unroll
      for (int nk = 0; nk < KT; ++nk)
        wait_values(keys[0][nk],
                    keys[1][nk],
                    keys[2][nk],
                    keys[3][nk],
                    keys[4][nk],
                    keys[5][nk],
                    keys[6][nk],
                    keys[7][nk]);
#pragma unroll
      for (int db = 0; db < 8; ++db) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) {
#pragma unroll
          for (int qt = 0; qt < QT; ++qt) scores[qt][nk] = dot(q[qt][db], keys[db][nk], scores[qt][nk]);
        }
      }
#pragma unroll
      for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float s = scores[qt][nk][j] * 0.12751743082455375f;
            scores[qt][nk][j] = nb + nk * 16 + group + j * 4 < N ? s : -__uint_as_float(0x7f800000u);
          }
        }
        mx[qt] = 0.f;
      }
#pragma unroll
      for (int nk = 0; nk < KT; ++nk) {
        u2 values[8];
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) values[dv] = read64(sv + (nk * 8 * 65 + dv * 65 + lane) * 4);
        u2 hi[QT];
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
          uint16_t hh[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float p = __builtin_amdgcn_exp2f(scores[qt][nk][j] - mx[qt]);
            uint32_t z = __float_as_uint(p);
            unsafe |= (nb + nk * 16 + group + j * 4 < N) && ((z - 0x38800000u) > (0x46800000u - 0x38800000u));
            denom[qt] += p;
            hh[j] = round_fp16(p);
            err[qt] += fabsf(p - fp16_value(hh[j]));
          }
          hi[qt] = pk(hh[0], hh[1], hh[2], hh[3]);
        }
        wait_values(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);

#pragma unroll
        for (int dv = 0; dv < 8; ++dv) {
#pragma unroll
          for (int qt = 0; qt < QT; ++qt) {
            acc[qt][dv] = dot_fp16(hi[qt], values[dv], acc[qt][dv]);
          }
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * W * 64 * 8, dd = idx % 128;
      // Keep V checks/conversion after the current tile's MMACs. Performing
      // them at prefetch time forces an early wait for the next global load.
      asm volatile("" : "+v"(next_v[j]) : "v"(acc[QT - 1][7]) : "memory");
#pragma unroll
      for (int z = 0; z < 4; ++z) {
        vmaxbits = max(vmaxbits, next_v[j][z] & 0x7fffu);
        vmaxbits = max(vmaxbits, (next_v[j][z] >> 16) & 0x7fffu);
      }
      u2 va = convert_v_fp16(u2 {next_v[j][0], next_v[j][1]}),
         vb = convert_v_fp16(u2 {next_v[j][2], next_v[j][3]});
      next_v[j] = u4 {va[0], va[1], vb[0], vb[1]};
#pragma unroll
      for (int hp = 0; hp < 2; ++hp)
        *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 + ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
            u2 {next_k[j][hp * 2], next_k[j][hp * 2 + 1]};
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        int vr = idx / 128, vc = dd + jp;
        sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 + (vr % 16) / 4] =
            uint16_t(next_v[j][jp / 2] >> (16 * (jp % 2)));
      }
    }
    __syncthreads();
  }
  if (unsafe) vmaxbits = 0x7f80u;
  float max_relative = 0;
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    float den = denom[qt], e = err[qt];
    den += __shfl_xor(den, 16, 64);
    den += __shfl_xor(den, 32, 64);
    e += __shfl_xor(e, 16, 64);
    e += __shfl_xor(e, 32, 64);
    denom[qt] = den;
    if (qb + qt * 16 + row < N) max_relative = fmaxf(max_relative, e / den + 0.00005f);
  }
  for (int mask = 32; mask; mask /= 2) {
    vmaxbits = max(vmaxbits, __shfl_xor(vmaxbits, mask, 64));
    max_relative = fmaxf(max_relative, __shfl_xor(max_relative, mask, 64));
  }
  if (lane == 0) {
    bounds[wave] = vmaxbits;
    errors[wave] = max_relative;
  }
  __syncthreads();
  uint32_t block_vmax = 0;
  float block_relative = 0;
#pragma unroll
  for (int wi = 0; wi < W; ++wi) {
    block_vmax = max(block_vmax, bounds[wi]);
    block_relative = fmaxf(block_relative, errors[wi]);
  }
  bool good = block_vmax <= 0x4700u && block_relative * bf(uint16_t(block_vmax)) + 0.0000001f <= 0.0015f;
  if (tid == 0) F[head * ((N + BM - 1) / BM) + blockIdx.x] = good;
  if (!good) return false;
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    float den = denom[qt];
    int qr = qb + qt * 16 + row;
    if (qr < N) {
#pragma unroll
      for (int dv = 0; dv < 8; ++dv) {
        uint32_t p0 = uint32_t(rb(acc[qt][dv][0] / den)) | (uint32_t(rb(acc[qt][dv][1] / den)) << 16);
        uint32_t p1 = uint32_t(rb(acc[qt][dv][2] / den)) | (uint32_t(rb(acc[qt][dv][3] / den)) << 16);
        uint32_t o0 = __shfl_xor(p0, 16, 64), o1 = __shfl_xor(p1, 16, 64);
        uint32_t r0 = (group & 1) ? ((o0 >> 16) | (p0 & 0xffff0000u)) : ((p0 & 0xffffu) | (o0 << 16));
        uint32_t r1 = (group & 1) ? ((o1 >> 16) | (p1 & 0xffff0000u)) : ((p1 & 0xffffu) | (o1 << 16));
        uint32_t s0 = __shfl_xor(r0, 32, 64), s1 = __shfl_xor(r1, 32, 64);
        u2 vec = (group & 2) ? u2 {s1, r1} : u2 {r0, s0};
        *reinterpret_cast<u2 *>(O + (head * N + qr) * 128 + dv * 16 + group * 4) = vec;
      }
      if (group == 0) L[head * N + qr] = (mx[qt] + log2f(den)) * 0.6931471805599453f;
    }
  }
  return true;
}
// Outline the conditional fallback to keep the fast path's register allocation
// independent. Three BF16 probability components retain approximately FP32
// precision for large V and cancellation near zero.
template <int W, int QT, int BN>
__device__ __attribute__((noinline)) void short_stable(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O, float *L) {
  constexpr int N = 362, H = 16, D = 128, BM = W * QT * 16, KT = BN / 16, LOADS = BN * 128 / (W * 64 * 8);
  int tid = threadIdx.x, lane = tid % 64, wave = tid / 64, row = lane % 16, group = lane / 16;
  int head = blockIdx.y, qb = blockIdx.x * BM + wave * QT * 16;
  int64_t base = int64_t(head / 16) * (N * 6144) + (head % 16) * 128;
  __shared__ uint16_t smem[BN * (128 + 130)];
  auto sk = smem;
  auto sv = smem + BN * 128;
  u2 q[QT][8];
  f4 acc[QT][8];
  float denom[QT] = {}, mx[QT];
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    mx[qt] = -__uint_as_float(0x7f800000u);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      q[qt][j] = qb + qt * 16 + row < N
                     ? ld(Q + base + int64_t(qb + qt * 16 + row) * 6144 + j * 16 + group * 4)
                     : u2 {0, 0};
      acc[qt][j] = {0, 0, 0, 0};
    }
  }
  for (int nb = 0; nb < N; nb += BN) {
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
      int idx = tid * 8 + j * W * 64 * 8, kr = nb + idx / 128, dd = idx % 128;
      u4 kv = kr < N ? ld4(K + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
      u4 vv = kr < N ? ld4(V + base + int64_t(kr) * 6144 + dd) : u4 {0, 0, 0, 0};
#pragma unroll
      for (int hp = 0; hp < 2; ++hp)
        *reinterpret_cast<u2 *>(sk + (idx / 128) * 128 + ((dd + hp * 4) ^ ((idx / 128 % 16) * 4))) =
            u2 {kv[hp * 2], kv[hp * 2 + 1]};
#pragma unroll
      for (int jp = 0; jp < 8; ++jp) {
        int vr = idx / 128, vc = dd + jp;
        sv[((vr / 16) * 8 * 65 + (vc / 16) * 65 + (vr % 4) * 16 + vc % 16) * 4 + (vr % 16) / 4] =
            uint16_t(vv[jp / 2] >> (16 * (jp % 2)));
      }
    }
    __syncthreads();
    if (qb < N) {
      f4 scores[QT][KT];
#pragma unroll
      for (int qt = 0; qt < QT; ++qt) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) scores[qt][nk] = {0, 0, 0, 0};
      }
      u2 keys[8][KT];
#pragma unroll
      for (int db = 0; db < 8; ++db) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk)
          keys[db][nk] = read64(sk + (nk * 16 + row) * 128 + ((db * 16 + group * 4) ^ ((row % 16) * 4)));
      }
#pragma unroll
      for (int nk = 0; nk < KT; ++nk)
        wait_values(keys[0][nk],
                    keys[1][nk],
                    keys[2][nk],
                    keys[3][nk],
                    keys[4][nk],
                    keys[5][nk],
                    keys[6][nk],
                    keys[7][nk]);
#pragma unroll
      for (int db = 0; db < 8; ++db) {
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) {
#pragma unroll
          for (int qt = 0; qt < QT; ++qt) scores[qt][nk] = dot(q[qt][db], keys[db][nk], scores[qt][nk]);
        }
      }
#pragma unroll
      for (int qt = 0; qt < QT; ++qt) {
        float mm = -__uint_as_float(0x7f800000u);
#pragma unroll
        for (int nk = 0; nk < KT; ++nk) {
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            scores[qt][nk][j] = nb + nk * 16 + group + j * 4 < N ? scores[qt][nk][j] * 0.12751743082455375f
                                                                 : -__uint_as_float(0x7f800000u);
            mm = fmaxf(mm, scores[qt][nk][j]);
          }
        }
        mm = fmaxf(mm, __shfl_xor(mm, 16, 64));
        mm = fmaxf(mm, __shfl_xor(mm, 32, 64));
        mm = fmaxf(mm, mx[qt]);
        float a = __builtin_amdgcn_exp2f(mx[qt] - mm);
        denom[qt] *= a;
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) acc[qt][dv] *= a;
        mx[qt] = mm;
      }
#pragma unroll
      for (int nk = 0; nk < KT; ++nk) {
        u2 values[8];
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) values[dv] = read64(sv + (nk * 8 * 65 + dv * 65 + lane) * 4);
        u2 hi[QT], lo[QT], tail[QT];
#pragma unroll
        for (int qt = 0; qt < QT; ++qt) {
          uint16_t hh[4], ll[4], tt[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            float p = __builtin_amdgcn_exp2f(scores[qt][nk][j] - mx[qt]);
            denom[qt] += p;
            hh[j] = rb(p);
            float rem = p - bf(hh[j]);
            ll[j] = rb(rem);
            tt[j] = rb(rem - bf(ll[j]));
          }
          hi[qt] = pk(hh[0], hh[1], hh[2], hh[3]);
          lo[qt] = pk(ll[0], ll[1], ll[2], ll[3]);
          tail[qt] = pk(tt[0], tt[1], tt[2], tt[3]);
        }
        wait_values(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
#pragma unroll
        for (int dv = 0; dv < 8; ++dv) {
#pragma unroll
          for (int qt = 0; qt < QT; ++qt) {
            acc[qt][dv] = dot(hi[qt], values[dv], acc[qt][dv]);
            acc[qt][dv] = dot(lo[qt], values[dv], acc[qt][dv]);
            acc[qt][dv] = dot(tail[qt], values[dv], acc[qt][dv]);
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int qt = 0; qt < QT; ++qt) {
    float den = denom[qt];
    den += __shfl_xor(den, 16, 64);
    den += __shfl_xor(den, 32, 64);
    int qr = qb + qt * 16 + row;
    if (qr < N) {
#pragma unroll
      for (int dv = 0; dv < 8; ++dv) {
        uint32_t p0 = uint32_t(rb(acc[qt][dv][0] / den)) | (uint32_t(rb(acc[qt][dv][1] / den)) << 16);
        uint32_t p1 = uint32_t(rb(acc[qt][dv][2] / den)) | (uint32_t(rb(acc[qt][dv][3] / den)) << 16);
        uint32_t o0 = __shfl_xor(p0, 16, 64), o1 = __shfl_xor(p1, 16, 64);
        uint32_t r0 = (group & 1) ? ((o0 >> 16) | (p0 & 0xffff0000u)) : ((p0 & 0xffffu) | (o0 << 16));
        uint32_t r1 = (group & 1) ? ((o1 >> 16) | (p1 & 0xffff0000u)) : ((p1 & 0xffffu) | (o1 << 16));
        uint32_t s0 = __shfl_xor(r0, 32, 64), s1 = __shfl_xor(r1, 32, 64);
        u2 vec = (group & 2) ? u2 {s1, r1} : u2 {r0, s0};
        *reinterpret_cast<u2 *>(O + (head * N + qr) * 128 + dv * 16 + group * 4) = vec;
      }
      if (group == 0) L[head * N + qr] = (mx[qt] + log2f(den)) * 0.6931471805599453f;
    }
  }
}

template <int W, int QT, int BN>
__global__ __launch_bounds__(W * 64) void short_state_flash(
    const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *O, float *L, uint8_t *F) {
  if (try_short_fp16<W, QT, BN>(Q, K, V, O, L, F)) return;
  short_stable<W, QT, BN>(Q, K, V, O, L);
}

extern "C" int launch_short_state_flash(
    const void *q, const void *k, const void *v, uint8_t *flags, void *out, float *lse, hipStream_t stream) {
  hipLaunchKernelGGL((short_state_flash<4, 3, 16>),
                     dim3(2, 32),
                     dim3(256),
                     0,
                     stream,
                     static_cast<const uint16_t *>(q),
                     static_cast<const uint16_t *>(k),
                     static_cast<const uint16_t *>(v),
                     static_cast<uint16_t *>(out),
                     lse,
                     flags);
  return hipGetLastError();
}
