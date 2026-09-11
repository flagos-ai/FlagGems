// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef CV_INPUT_NZ
#define CV_INPUT_NZ 0
#endif
#ifndef CV_BATCH_ROWS
#define CV_BATCH_ROWS 0
#endif
#ifndef CV_PREFETCH_INPUT
#define CV_PREFETCH_INPUT 0
#endif
#define SF_INLINE __aicore__ inline
/**
 * AIC-only CommonIR epilogue for W8A8 matmul on Ascend 910B.
 *
 * TLE leaves the INT32 matmul tile in L0C.  This fragment performs the full
 * epilogue without an AIV kernel:
 *
 *   1. VDEQF16 FixPipe: INT32 L0C -> FP16 L1, applying the N-channel scale.
 *   2. 16x16 diagonal Cube matmuls: apply the per-row scale in AIC.
 *   3. F322BF16 FixPipe: FP32 L0C -> final BF16 GM output.
 *
 * The host supplies row scales as packed 16x16 FP16 diagonal blocks.  The
 * N-channel scale already includes a common row-scale reference; diagonal
 * entries contain row_scale / reference.  This keeps the intermediate FP16
 * magnitude close to the final result.
 */

#include "kernel_operator.h"
#include "mm_aic_soft_float.hpp"

namespace {
constexpr uint32_t kDeqC1Addr = (CV_AM + CV_BN) * CV_K;
constexpr uint32_t kTmpC1Addr = (CV_AM + CV_BN) * CV_K + CV_N * 8;
constexpr uint32_t kDiagC1Addr = (CV_AM + CV_BN) * CV_K + CV_N * 8 + CV_BM * CV_BN * 2;
constexpr int32_t kMaxFullDeqN = CV_N;
constexpr int32_t kCube = 16;

__aicore__ inline int32_t Align16(int32_t x) {
  return ((x + 15) / 16) * 16;
}

__aicore__ inline int32_t Align32(int32_t x) {
  return ((x + 31) / 32) * 32;
}

__aicore__ inline void WrapPos(AscendC::TBuffAddr &addr,
                               uint32_t bytes,
                               uint32_t offset,
                               AscendC::TPosition pos) {
  addr.dataLen = bytes;
  addr.bufferAddr = offset;
  addr.bufferHandle = nullptr;
  addr.logicPos = static_cast<uint8_t>(pos);
}

__aicore__ inline void CopyDeqToC1(__gm__ uint64_t *deqGmPtr, uint32_t dstOff, uint32_t count) {
  AscendC::LocalTensor<uint64_t> dst;
  AscendC::TBuffAddr dstAddr;
  WrapPos(dstAddr, count * sizeof(uint64_t), dstOff, AscendC::TPosition::C1);
  dst.SetAddr(dstAddr);
  AscendC::GlobalTensor<uint64_t> src;
  src.SetGlobalBuffer(deqGmPtr);
  AscendC::DataCopy(dst, src, count);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID2);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID2);
}

__aicore__ inline void CopyDiagToC1(__gm__ half *diagGmPtr, uint32_t count) {
  AscendC::LocalTensor<half> dst;
  AscendC::TBuffAddr dstAddr;
  WrapPos(dstAddr, count * sizeof(half), kDiagC1Addr, AscendC::TPosition::C1);
  dst.SetAddr(dstAddr);
  AscendC::GlobalTensor<half> src;
  src.SetGlobalBuffer(diagGmPtr);
  AscendC::DataCopy(dst, src, count);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
}

__aicore__ inline void RunFixpipeFull(int64_t acc,
                                      __gm__ uint64_t *deq,
                                      __gm__ half *diag,
                                      __gm__ uint16_t *c,
                                      int32_t offM,
                                      int32_t offN,
                                      int32_t tileM,
                                      int32_t tileN,
                                      int32_t accStride,
                                      int32_t ldc,
                                      int32_t loadDeq,
                                      int32_t loadDiag,
                                      int32_t outBf16) {
  if ASCEND_IS_AIV {
    return;
  }
  if (tileM <= 0 || tileN <= 0 || offM < 0 || offN < 0) {
    return;
  }

  const int32_t mAlign = Align16(tileM);
  const int32_t nAlign = Align32(tileN);
  const bool fullDeq = (ldc > 0 && ldc <= kMaxFullDeqN);
  if (loadDeq == 2 && fullDeq) {
    CopyDeqToC1(deq, kDeqC1Addr, static_cast<uint32_t>(Align32(ldc)));
  } else if (loadDeq != 0) {
    CopyDeqToC1(deq + offN, kDeqC1Addr, static_cast<uint32_t>(nAlign));
  }

  const uint32_t deqOff = (fullDeq && (loadDeq == 2 || loadDeq == 0))
                              ? kDeqC1Addr + static_cast<uint32_t>(offN) * sizeof(uint64_t)
                              : kDeqC1Addr;
  AscendC::LocalTensor<uint64_t> deqLocal;
  AscendC::TBuffAddr deqAddr;
  WrapPos(deqAddr, static_cast<uint32_t>(nAlign) * sizeof(uint64_t), deqOff, AscendC::TPosition::C1);
  deqLocal.SetAddr(deqAddr);

  uint32_t accOff = 0;
  if (acc >= 0 && acc < (256 * 1024)) {
    accOff = static_cast<uint32_t>(acc);
  }
  AscendC::LocalTensor<int32_t> accLocal;
  AscendC::TBuffAddr accAddr;
  WrapPos(accAddr,
          static_cast<uint32_t>(accStride) * static_cast<uint32_t>(nAlign) * sizeof(int32_t),
          accOff,
          AscendC::TPosition::CO1);
  accLocal.SetAddr(accAddr);

  AscendC::LocalTensor<half> tmpC1;
  AscendC::TBuffAddr tmpAddr;
  WrapPos(tmpAddr,
          static_cast<uint32_t>(accStride) * static_cast<uint32_t>(nAlign) * sizeof(half),
          kTmpC1Addr,
          AscendC::TPosition::C1);
  tmpC1.SetAddr(tmpAddr);

  AscendC::FixpipeParamsV220 deqParams;
  deqParams.nSize = static_cast<uint16_t>(tileN);
  // The padded A rows are zero.  Drain a complete M0 fractal so the
  // transposed L1->L0B load never reads uninitialized lanes on M tails.
  deqParams.mSize = static_cast<uint16_t>(accStride);
  deqParams.srcStride = static_cast<uint16_t>(accStride);
  // NZ L1 layout: one N0 fractal contains mAlign rows, in 32-byte units.
  deqParams.dstStride = static_cast<uint32_t>(accStride);
  deqParams.quantPre = QuantMode_t::VDEQF16;
  AscendC::Fixpipe<half, int32_t, AscendC::CFG_NZ>(tmpC1, accLocal, deqLocal, deqParams);
  AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(EVENT_ID0);

  AscendC::LocalTensor<half> diagC1;
  AscendC::TBuffAddr diagC1Addr;
  WrapPos(diagC1Addr,
          static_cast<uint32_t>((accStride / 16 + 1) * 256) * sizeof(half),
          kDiagC1Addr,
          AscendC::TPosition::C1);
  diagC1.SetAddr(diagC1Addr);

  AscendC::LocalTensor<half> a2;
  AscendC::TBuffAddr a2Addr;
  WrapPos(a2Addr,
          static_cast<uint32_t>(accStride) * static_cast<uint32_t>(accStride) * sizeof(half),
          0,
          AscendC::TPosition::A2);
  a2.SetAddr(a2Addr);

  AscendC::LocalTensor<half> b2;
  AscendC::TBuffAddr b2Addr;
  WrapPos(b2Addr,
          static_cast<uint32_t>(accStride) * static_cast<uint32_t>(nAlign) * sizeof(half),
          0,
          AscendC::TPosition::B2);
  b2.SetAddr(b2Addr);

  // The original INT32 tile has been drained, so L0C can be reused by the
  // row-scale Cube matmul.
  AscendC::LocalTensor<float> scaledAcc;
  AscendC::TBuffAddr scaledAccAddr;
  WrapPos(scaledAccAddr,
          static_cast<uint32_t>(accStride) * static_cast<uint32_t>(nAlign) * sizeof(float),
          accOff,
          AscendC::TPosition::CO1);
  scaledAcc.SetAddr(scaledAccAddr);

  const int32_t mBlocks = accStride / kCube;
  const int32_t nBlocks = nAlign / kCube;
  const int64_t diagBase = static_cast<int64_t>(offM / accStride) * (accStride / 16 + 1) * 256;
  if (loadDiag == 1) {
    CopyDiagToC1(diag + diagBase, static_cast<uint32_t>((accStride / 16 + 1) * 256));
  } else if (loadDiag == 2) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
  }

  const int64_t cOffset = static_cast<int64_t>(offM) * ldc + offN;
  AscendC::FixpipeParamsV220 outParams;
  outParams.nSize = static_cast<uint16_t>(tileN);
  outParams.dstStride = static_cast<uint32_t>(ldc);

  // Process independent 16-row diagonal blocks, retaining their FP32
  // results in compact L0C matrices. One batched Fixpipe drains all blocks.
  // CANN A2: source ND stride is in 1024B units; destination in elements.
  if constexpr (CV_BATCH_ROWS) {
    static_assert(!CV_BATCH_ROWS || (CV_M % 16 == 0 && CV_N <= 4095));
    const int blocks = CV_M % CV_BM == 0 ? mBlocks : tileM / 16;
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    for (int mb = 0; mb < blocks; ++mb) {
      int bank = mb % 2;
      auto event = bank ? EVENT_ID1 : EVENT_ID0;
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event);
      AscendC::LoadData2DParams la;
      la.repeatTimes = 1;
      la.srcStride = 1;
      AscendC::LoadData(a2[bank * 256], diagC1[(mb + 1) * 256], la);
      AscendC::LoadData2DParams lb;
      lb.repeatTimes = static_cast<uint8_t>(nBlocks);
      lb.srcStride = static_cast<uint16_t>(mBlocks);
      lb.ifTranspose = true;
      AscendC::LoadData(b2[bank * 16 * nAlign], tmpC1[mb * 256], lb);
      AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event);
      AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event);
      AscendC::MmadParams mm;
      mm.m = 16;
      mm.n = static_cast<uint16_t>(tileN);
      mm.k = 16;
      mm.cmatrixInitVal = true;
      AscendC::Mmad(scaledAcc[mb * 16 * nAlign], a2[bank * 256], b2[bank * 16 * nAlign], mm);
      AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event);
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
    AscendC::GlobalTensor<bfloat16_t> cGm;
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(c) + cOffset);
    outParams.mSize = 16;
    outParams.srcStride = 16;
    outParams.ndNum = static_cast<uint16_t>(blocks);
    outParams.srcNdStride = static_cast<uint16_t>(16 * nAlign * sizeof(float) / 1024);
    outParams.dstNdStride = static_cast<uint16_t>(16 * ldc);
    outParams.quantPre = QuantMode_t::F322BF16;
    AscendC::Fixpipe(cGm, scaledAcc, outParams);
    return;
  }

  // Apply the packed block-diagonal row-scale matrix in one Cube Mmad.
  if (mBlocks > 1) {
    AscendC::LoadData2DParams zero;
    zero.repeatTimes = static_cast<uint8_t>(mBlocks * mBlocks);
    zero.srcStride = 0;
    AscendC::LoadData(a2, diagC1, zero);
    AscendC::PipeBarrier<PIPE_ALL>();
  }
  for (int32_t mb = 0; mb < mBlocks; ++mb) {
    AscendC::LoadData2DParams loadA;
    loadA.repeatTimes = 1;
    loadA.srcStride = 1;
    AscendC::LoadData(a2[(mb * mBlocks + mb) * kCube * kCube], diagC1[(mb + 1) * kCube * kCube], loadA);
  }
  for (int32_t kb = 0; kb < mBlocks; ++kb) {
    AscendC::LoadData2DParams loadB;
    loadB.repeatTimes = static_cast<uint8_t>(nBlocks);
    loadB.srcStride = static_cast<uint16_t>(mBlocks);
    loadB.ifTranspose = true;
    AscendC::LoadData(b2[kb * nBlocks * kCube * kCube], tmpC1[kb * kCube * kCube], loadB);
  }
  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
  AscendC::MmadParams mmParams;
  mmParams.m = static_cast<uint16_t>(accStride);
  mmParams.n = static_cast<uint16_t>(tileN);
  mmParams.k = static_cast<uint16_t>(accStride);
  mmParams.cmatrixInitVal = true;
  AscendC::Mmad(scaledAcc, a2, b2, mmParams);
  AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

  outParams.mSize = static_cast<uint16_t>(tileM);
  outParams.srcStride = static_cast<uint16_t>(accStride);
  if (outBf16 != 0) {
    AscendC::GlobalTensor<bfloat16_t> cGm;
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(c) + cOffset);
    outParams.quantPre = QuantMode_t::F322BF16;
    AscendC::Fixpipe(cGm, scaledAcc, outParams);
  } else {
    AscendC::GlobalTensor<half> cGm;
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c) + cOffset);
    outParams.quantPre = QuantMode_t::F322F16;
    AscendC::Fixpipe(cGm, scaledAcc, outParams);
  }
  // The surrounding TLE loop owns the final FIX->M dependency for the next
  // accumulator use.  Do not duplicate that wait here; it serializes the
  // output FixPipe with the next tile's GM->L1 work.
}
}  // namespace

#include "kernel_operator.h"
using namespace AscendC;
template <class T>
__aicore__ inline LocalTensor<T> Local(uint32_t off, uint32_t count, TPosition pos) {
  TBuffAddr a {};
  a.dataLen = count * sizeof(T);
  a.bufferAddr = off;
  a.logicPos = static_cast<uint8_t>(pos);
  LocalTensor<T> t;
  t.SetAddr(a);
  return t;
}
constexpr int TILES = (CV_MP / CV_BM) * (CV_NP / CV_BN), Q = TILES / CV_CORES, REM = TILES % CV_CORES;
__aicore__ inline int Start(int pid) {
  return pid < REM ? pid * (Q + 1) : REM * (Q + 1) + (pid - REM) * Q;
}
__aicore__ inline int Count(int pid) {
  return Q + (pid < REM);
}

#if defined(__DAV_C220_CUBE__)
extern "C"[aicore] __attribute__((always_inline)) void CV_CUBE_ENTRY(int64_t ap,
                                                                     int64_t bp,
                                                                     int64_t op,
                                                                     int64_t deqp,
                                                                     int64_t diagp,
                                                                     int64_t flagp,
                                                                     int64_t sap,
                                                                     int64_t sbp,
                                                                     int64_t wsp,
                                                                     int32_t pid) {
  PipeBarrier<PIPE_ALL>();
  GlobalTensor<uint32_t> fg;
  fg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(flagp));
  DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(fg);
  bool useFast = false;
  auto a1 = Local<int8_t>(0, CV_AM * CV_K, TPosition::A1);
  auto b1 = Local<int8_t>(CV_AM * CV_K, CV_BN * CV_K, TPosition::B1);
  constexpr int BANKS = (2 * CV_BM * CV_BK <= 65536 && 2 * CV_BN * CV_BK <= 65536) ? 2 : 1;
  auto a2 = Local<int8_t>(0, BANKS * CV_BM * CV_BK, TPosition::A2);
  auto b2 = Local<int8_t>(0, BANKS * CV_BN * CV_BK, TPosition::B2);
  SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
  SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
  SetFlag<HardEvent::FIX_M>(EVENT_ID0);
  SetFlag<HardEvent::FIX_M>(EVENT_ID1);
  int start = Start(pid), count = Count(pid);
  int lastRow = -1, lastCol = -1;
  for (int i = 0; i < count; ++i) {
    int tile = start + i, row = (tile % (CV_MP / CV_BM)) * CV_BM, col = (tile / (CV_MP / CV_BM)) * CV_BN;
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    GlobalTensor<int8_t> ag, bg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(ap) +
                       (CV_INPUT_NZ ? (row / CV_BM) * CV_AM * CV_K : row * CV_K));
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(bp) + col * CV_K);
    constexpr bool SPLIT = CV_K >= 2048 && CV_BK <= CV_K / 2;
    if (!CV_PREFETCH_INPUT || i == 0) {
      if constexpr (CV_INPUT_NZ) {
        if constexpr (SPLIT) {
          if (row != lastRow) DataCopy(a1, ag, CV_AM * CV_K / 2);
          if (col != lastCol) DataCopy(b1, bg, CV_BN * CV_K / 2);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
          if (row != lastRow) DataCopy(a1[CV_AM * (CV_K / 2)], ag[CV_AM * CV_K / 2], CV_AM * CV_K / 2);
          if (col != lastCol) DataCopy(b1[CV_BN * (CV_K / 2)], bg[CV_BN * CV_K / 2], CV_BN * CV_K / 2);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
        } else {
          if (row != lastRow) DataCopy(a1, ag, CV_AM * CV_K);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
          if (col != lastCol) DataCopy(b1, bg, CV_BN * CV_K);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
        }
      } else {
        Nd2NzParams pa;
        pa.ndNum = 1;
        pa.nValue = CV_BM;
        pa.dValue = CV_K;
        pa.srcNdMatrixStride = 0;
        pa.srcDValue = CV_K;
        pa.dstNzC0Stride = CV_AM;
        pa.dstNzNStride = 1;
        pa.dstNzMatrixStride = 0;
        Nd2NzParams pb = pa;
        pb.nValue = CV_BN;
        pb.dstNzC0Stride = CV_BN;
        if constexpr (SPLIT) {
          pa.dValue = CV_K / 2;
          pb.dValue = CV_K / 2;
          if (row != lastRow) DataCopy(a1, ag, pa);
          if (col != lastCol) DataCopy(b1, bg, pb);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
          if (row != lastRow) DataCopy(a1[CV_AM * (CV_K / 2)], ag[CV_K / 2], pa);
          if (col != lastCol) DataCopy(b1[CV_BN * (CV_K / 2)], bg[CV_K / 2], pb);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
        } else {
          if (row != lastRow) DataCopy(a1, ag, pa);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
          if (col != lastCol) DataCopy(b1, bg, pb);
          SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
        }
      }
    }
    if (i == 0) {
      asm volatile("" ::: "memory");
      useFast = fg.GetValue(0) != 0;
    }
    if (useFast && i == 0) CopyDeqToC1(reinterpret_cast<__gm__ uint64_t *>(deqp), kDeqC1Addr, CV_N);
    if (useFast && row != lastRow) {
      constexpr int DW = (CV_BM / 16 + 1) * 256;
      auto dg = Local<half>(kDiagC1Addr, DW, TPosition::C1);
      GlobalTensor<half> dgg;
      dgg.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(diagp) + (row / CV_BM) * DW);
      DataCopy(dg, dgg, DW);
      SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2);
    }

    constexpr int CBANKS = (CV_BM * CV_BN * 4 <= 65536) ? 2 : 1;
    int cb = i % CBANKS;
    auto ce = cb ? EVENT_ID1 : EVENT_ID0;
    auto c0 = Local<int32_t>(cb * CV_BM * CV_BN * 4, CV_BM * CV_BN, TPosition::CO1);
    WaitFlag<HardEvent::FIX_M>(ce);
    for (int ki = 0; ki < CV_K / CV_BK; ++ki) {
      int bank = ki % BANKS;
      auto e = bank ? EVENT_ID1 : EVENT_ID0;
      WaitFlag<HardEvent::M_MTE1>(e);
      if (ki == 0) WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
      if constexpr (SPLIT) {
        if (ki == CV_K / (2 * CV_BK)) WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
      }
      for (int mi = 0; mi < CV_BM / 16; ++mi) {
        LoadData2DParams ld;
        ld.repeatTimes = CV_BK / 32;
        ld.srcStride = CV_AM / 16;
        LoadData(a2[bank * CV_BM * CV_BK + mi * CV_BK * 16], a1[ki * CV_BK * CV_AM + mi * 512], ld);
      }
      if constexpr (!SPLIT) {
        if (ki == 0) WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
      }
      LoadData2DParams lb;
      lb.repeatTimes = (CV_BK / 32) * (CV_BN / 16);
      lb.srcStride = 1;
      LoadData(b2[bank * CV_BN * CV_BK], b1[ki * CV_BK * CV_BN], lb);
      SetFlag<HardEvent::MTE1_M>(e);
      WaitFlag<HardEvent::MTE1_M>(e);
      MmadParams mm;
      mm.m = CV_BM;
      mm.n = CV_BN;
      mm.k = CV_BK;
      mm.cmatrixInitVal = ki == 0;
      Mmad(c0, a2[bank * CV_BM * CV_BK], b2[bank * CV_BN * CV_BK], mm);
      SetFlag<HardEvent::M_MTE1>(e);
    }
    if constexpr (CV_PREFETCH_INPUT) {
      static_assert(!CV_PREFETCH_INPUT || (CV_INPUT_NZ && CV_K == 512));
      if (i + 1 < count) {
        // Last L1->L0 loads have consumed the current A/B panels.
        // Refill those C1 regions while the current tile's epilogue runs.
        int next = start + i + 1;
        int nr = (next % (CV_MP / CV_BM)) * CV_BM;
        int nc = (next / (CV_MP / CV_BM)) * CV_BN;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        GlobalTensor<int8_t> nag, nbg;
        nag.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(ap) + (nr / CV_BM) * CV_AM * CV_K);
        nbg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(bp) + nc * CV_K);
        if (nr != row) DataCopy(a1, nag, CV_AM * CV_K);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
        if (nc != col) DataCopy(b1, nbg, CV_BN * CV_K);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
      }
    }
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    if (useFast) {
      RunFixpipeFull(cb * CV_BM * CV_BN * 4,
                     reinterpret_cast<__gm__ uint64_t *>(deqp),
                     reinterpret_cast<__gm__ half *>(diagp),
                     reinterpret_cast<__gm__ uint16_t *>(op),
                     row,
                     col,
                     (row + CV_BM <= CV_M ? CV_BM : CV_M - row),
                     (col + CV_BN <= CV_N ? CV_BN : CV_N - col),
                     CV_BM,
                     CV_N,
                     0,
                     row != lastRow ? 2 : 0,
                     1);
    } else {
      GlobalTensor<int32_t> wg;
      wg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(wsp) + pid * CV_BM * CV_BN);
      FixpipeParamsV220 f;
      f.nSize = CV_BN;
      f.mSize = CV_BM;
      f.srcStride = CV_BM;
      f.dstStride = CV_BN;
      f.quantPre = QuantMode_t::NoQuant;
      Fixpipe(wg, c0, f);
      PipeBarrier<PIPE_ALL>();
      GlobalTensor<uint32_t> sg, tg, og;
      sg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(sap));
      tg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(sbp));
      og.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(op));
      DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(og);
      int mr = (row + CV_BM <= CV_M ? CV_BM : CV_M - row);
      for (int ri = 0; ri < mr; ++ri) {
        uint32_t sa = sg.GetValue(row + ri);
        for (int ci = 0; ci < (col + CV_BN <= CV_N ? CV_BN : CV_N - col); ci += 2) {
          uint32_t x = SoftFloat::Mul(SoftFloat::Mul(SoftFloat::FromInt(wg.GetValue(ri * CV_BN + ci)), sa),
                                      tg.GetValue(col + ci));
          uint32_t y =
              SoftFloat::Mul(SoftFloat::Mul(SoftFloat::FromInt(wg.GetValue(ri * CV_BN + ci + 1)), sa),
                             tg.GetValue(col + ci + 1));
          og.SetValue(((row + ri) * CV_N + col + ci) / 2,
                      uint32_t(SoftFloat::ToBfloat(x)) | (uint32_t(SoftFloat::ToBfloat(y)) << 16));
        }
      }
      DataCacheCleanAndInvalid<uint32_t, CacheLine::ENTIRE_DATA_CACHE>(og);
      PipeBarrier<PIPE_ALL>();
    }
    SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
    SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
    SetFlag<HardEvent::FIX_M>(ce);
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    lastRow = row;
    lastCol = col;
  }
  WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
  WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
  WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
  WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
  PipeBarrier<PIPE_ALL>();
}
#endif
