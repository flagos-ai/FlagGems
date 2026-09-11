// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
// A and B are prepacked NZ. Exact INT32 accumulation is drained to GM.
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
#if defined(__DAV_C220_CUBE__)
extern "C"[aicore] __attribute__((always_inline)) void CV_CUBE_ENTRY(int64_t ap,
                                                                     int64_t bp,
                                                                     int64_t op,
                                                                     int32_t pid) {
  constexpr int NT = CV_NP / CV_BN, MT = CV_MP / CV_BM, Q = MT * NT / CV_CORES, R = MT * NT % CV_CORES;
  constexpr int BANKS = (2 * CV_BM * CV_BK <= 65536 && 2 * CV_BN * CV_BK <= 65536) ? 2 : 1;
  auto a1 = Local<int8_t>(0, CV_BM * CV_K, TPosition::A1);
  auto b1 = Local<int8_t>(CV_BM * CV_K, CV_BN * CV_K, TPosition::B1);
  auto a2 = Local<int8_t>(0, BANKS * CV_BM * CV_BK, TPosition::A2);
  auto b2 = Local<int8_t>(0, BANKS * CV_BN * CV_BK, TPosition::B2);
  auto c0 = Local<int32_t>(0, CV_BM * CV_BN, TPosition::CO1);
  int start = pid < R ? pid * (Q + 1) : R * (Q + 1) + (pid - R) * Q, count = Q + (pid < R), lastRow = -1,
      lastCol = -1;
  SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
  SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
  SetFlag<HardEvent::FIX_M>(EVENT_ID0);
  for (int i = 0; i < count; ++i) {
    int tile = start + i, row = tile / NT, col = tile % NT;
    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    GlobalTensor<int8_t> ag, bg;
    ag.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(ap) + row * CV_BM * CV_K);
    bg.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(bp) + col * CV_BN * CV_K);
    if (row != lastRow) DataCopy(a1, ag, CV_BM * CV_K / 2);
    if (col != lastCol) DataCopy(b1, bg, CV_BN * CV_K / 2);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
    if (row != lastRow) DataCopy(a1[CV_BM * CV_K / 2], ag[CV_BM * CV_K / 2], CV_BM * CV_K / 2);
    if (col != lastCol) DataCopy(b1[CV_BN * CV_K / 2], bg[CV_BN * CV_K / 2], CV_BN * CV_K / 2);
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
    for (int ki = 0; ki < CV_K / CV_BK; ++ki) {
      int bank = ki % BANKS;
      auto e = bank ? EVENT_ID1 : EVENT_ID0;
      WaitFlag<HardEvent::M_MTE1>(e);
      if (ki == 0) WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
      if (ki == CV_K / (2 * CV_BK)) WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID5);
      for (int mi = 0; mi < CV_BM / 16; ++mi) {
        LoadData2DParams ld;
        ld.repeatTimes = CV_BK / 32;
        ld.srcStride = CV_BM / 16;
        LoadData(a2[bank * CV_BM * CV_BK + mi * CV_BK * 16], a1[ki * CV_BK * CV_BM + mi * 512], ld);
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
    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
    GlobalTensor<int32_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(op) + row * CV_BM * CV_NP + col * CV_BN);
    FixpipeParamsV220 f;
    f.nSize = CV_BN;
    f.mSize = CV_BM;
    f.srcStride = CV_BM;
    f.dstStride = CV_NP;
    f.quantPre = QuantMode_t::NoQuant;
    Fixpipe(out, c0, f);
    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    lastRow = row;
    lastCol = col;
  }
  WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
  WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
  WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
  PipeBarrier<PIPE_ALL>();
}
#endif
