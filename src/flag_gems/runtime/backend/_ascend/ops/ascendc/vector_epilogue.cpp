// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
namespace {
template <class T>
__aicore__ inline AscendC::LocalTensor<T> Local(uint32_t addr, uint32_t count) {
  AscendC::TBuffAddr b {};
  b.dataLen = count * sizeof(T);
  b.bufferAddr = addr;
  b.bufferHandle = nullptr;
  b.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
  AscendC::LocalTensor<T> t;
  t.SetAddr(b);
  return t;
}

}  // namespace
#if defined(__DAV_C220_VEC__)
extern "C"[aicore] __attribute__((always_inline)) void _mlir_ciface_epilogue_brcb_ub(int64_t cp,
                                                                                     int64_t sap,
                                                                                     int64_t sbp,
                                                                                     int64_t brp,
                                                                                     int32_t row,
                                                                                     int32_t r,
                                                                                     int32_t x,
                                                                                     int32_t cachedA,
                                                                                     int64_t actual_out) {
  auto src = Local<int32_t>(static_cast<uint32_t>(cp), r * x);
  auto value = Local<float>(static_cast<uint32_t>(actual_out), r * x);
  auto sa = Local<float>(static_cast<uint32_t>(sap) + (cachedA ? row * 4 : 0), r);
  auto sb = Local<float>(static_cast<uint32_t>(sbp), x);
  auto rows = Local<float>(static_cast<uint32_t>(brp), r * 8);
  AscendC::BrcbRepeatParams br;
  br.dstBlkStride = 1;
  br.dstRepStride = 8;
  AscendC::BinaryRepeatParams rm;
  rm.dstBlkStride = x / 8;
  rm.src0BlkStride = x / 8;
  rm.src1BlkStride = 1;
  rm.dstRepStride = 1;
  rm.src0RepStride = 1;
  rm.src1RepStride = 0;
  AscendC::BinaryRepeatParams cm;
  cm.dstBlkStride = 1;
  cm.src0BlkStride = 1;
  cm.src1BlkStride = 1;
  cm.dstRepStride = x / 8;
  cm.src0RepStride = x / 8;
  cm.src1RepStride = 0;
  AscendC::Cast(value, src, AscendC::RoundMode::CAST_RINT, r * x);
  AscendC::Brcb(rows, sa, static_cast<uint8_t>(r / 8), br);
  AscendC::PipeBarrier<PIPE_V>();
  for (int ri = 0; ri < r; ri += 8)
    AscendC::Mul(value[ri * x], value[ri * x], rows[ri * 8], uint64_t(64), static_cast<uint8_t>(x / 8), rm);
  AscendC::PipeBarrier<PIPE_V>();
  for (int ci = 0; ci < x; ci += 64)
    AscendC::Mul(value[ci], value[ci], sb[ci], uint64_t(64), static_cast<uint8_t>(r), cm);
  AscendC::PipeBarrier<PIPE_V>();
}
#endif
