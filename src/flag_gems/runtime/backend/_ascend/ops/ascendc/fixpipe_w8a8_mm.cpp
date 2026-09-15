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

namespace {
constexpr uint32_t kDeqC1Addr = 384 * 1024;
constexpr uint32_t kTmpC1Addr = 320 * 1024;
constexpr uint32_t kDiagC1Addr = 448 * 1024;
constexpr int32_t kMaxFullDeqN = 8192;
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
          static_cast<uint32_t>(accStride) * static_cast<uint32_t>(accStride) * sizeof(half),
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
  const int64_t diagBase = static_cast<int64_t>(offM / accStride) * accStride * accStride;
  if (loadDiag != 0) {
    CopyDiagToC1(diag + diagBase, static_cast<uint32_t>(accStride * accStride));
  }

  const int64_t cOffset = static_cast<int64_t>(offM) * ldc + offN;
  AscendC::FixpipeParamsV220 outParams;
  outParams.nSize = static_cast<uint16_t>(tileN);
  outParams.dstStride = static_cast<uint32_t>(ldc);

  // Apply the packed block-diagonal row-scale matrix in one Cube Mmad.
  for (int32_t mb = 0; mb < mBlocks; ++mb) {
    AscendC::LoadData2DParams loadA;
    loadA.repeatTimes = static_cast<uint8_t>(mBlocks);
    loadA.srcStride = static_cast<uint16_t>(mBlocks);
    AscendC::LoadData(a2[mb * mBlocks * kCube * kCube], diagC1[mb * kCube * kCube], loadA);
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

#if defined(__DAV_C220_CUBE__)
extern "C"[aicore] __attribute__((always_inline)) void _mlir_ciface_fixpipe_vdeqf16(int64_t acc,
                                                                                    int64_t deq,
                                                                                    int64_t diag,
                                                                                    int64_t c,
                                                                                    int32_t offM,
                                                                                    int32_t offN,
                                                                                    int32_t tileM,
                                                                                    int32_t tileN,
                                                                                    int32_t accStride,
                                                                                    int32_t ldc,
                                                                                    int32_t loadDeq,
                                                                                    int32_t loadDiag,
                                                                                    int32_t outBf16) {
  RunFixpipeFull(acc,
                 reinterpret_cast<__gm__ uint64_t *>(deq),
                 reinterpret_cast<__gm__ half *>(diag),
                 reinterpret_cast<__gm__ uint16_t *>(c),
                 offM,
                 offN,
                 tileM,
                 tileN,
                 accStride,
                 ldc,
                 loadDeq,
                 loadDiag,
                 outBf16);
}
#endif
