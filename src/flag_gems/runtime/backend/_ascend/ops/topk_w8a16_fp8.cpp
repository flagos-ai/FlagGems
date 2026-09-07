// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
template <class T>
__aicore__ inline AscendC::LocalTensor<T> Local(int64_t addr, uint32_t count) {
  AscendC::TBuffAddr b;
  b.dataLen = count * sizeof(T);
  b.bufferAddr = static_cast<uint32_t>(addr);
  b.bufferHandle = nullptr;
  b.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
  AscendC::LocalTensor<T> t;
  t.SetAddr(b);
  return t;
}
extern "C"[aicore] __attribute__((always_inline)) void _mlir_ciface_topk_sort_pairs(
    int64_t values, int64_t indices, int64_t scratch, int32_t n, int64_t output) {
  auto v = Local<float>(values, n);
  auto i = Local<uint32_t>(indices, n);
  auto tmp = Local<float>(scratch, 2 * n);
  auto dst = Local<float>(output, 2 * n);
  AscendC::PipeBarrier<PIPE_ALL>();
  AscendC::Sort<float, true>(dst, v, i, tmp, n / 32);
  AscendC::PipeBarrier<PIPE_ALL>();
}

extern "C"[aicore] __attribute__((always_inline)) void _mlir_ciface_topk_select_codes(int64_t qp,
                                                                                      int64_t indexp,
                                                                                      int64_t vp,
                                                                                      int64_t ip,
                                                                                      int32_t row,
                                                                                      int32_t n,
                                                                                      int32_t k,
                                                                                      int32_t flip,
                                                                                      int64_t workspace,
                                                                                      int64_t unused) {
  using namespace AscendC;
  uint32_t b = static_cast<uint32_t>(workspace);
  auto key = Local<half>(b, n);
  auto tmp = Local<half>(b + 2 * n, n);
  auto raw = Local<int8_t>(b + 4 * n, n);
  auto mask = Local<uint8_t>(b + 5 * n, n / 8);
  auto eqmask = Local<uint8_t>(b + 5 * n + n / 8, n / 8);
  uint32_t extra = b + 5 * n + n / 4;
  auto chosen = Local<half>(extra, 512);
  auto chosenidx = Local<int16_t>(extra + 1024, 512);
  auto outv = Local<float>(extra + 2048, 512);
  auto outi = Local<int32_t>(extra + 4096, 512);
  GlobalTensor<int8_t> q;
  q.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(qp));
  GlobalTensor<float> v;
  v.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vp));
  GlobalTensor<int32_t> i;
  i.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ip));
  PipeBarrier<PIPE_ALL>();
  DataCopy(raw, q[static_cast<int64_t>(row) * n], n);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  Cast(key, raw, RoundMode::CAST_NONE, n);
  PipeBarrier<PIPE_V>();
  CompareScalar(mask, key, half(0), CMPMODE::LT, n);
  Muls(tmp, key, half(-1), n);
  PipeBarrier<PIPE_V>();
  Adds(tmp, tmp, half(-1), n);
  Adds(key, key, half(128), n);
  PipeBarrier<PIPE_V>();
  Select(key, mask, tmp, key, SELMODE::VSEL_TENSOR_TENSOR_MODE, n);
  PipeBarrier<PIPE_V>();
  if (flip) {
    Muls(key, key, half(-1), n);
    PipeBarrier<PIPE_V>();
    Adds(key, key, half(255), n);
    PipeBarrier<PIPE_V>();
  }
  int lo = 0, hi = 255;
  uint64_t count = 0;
  GatherMaskParams gp {1, 1, 8, 1};
  for (int it = 0; it < 8; ++it) {
    int mid = (lo + hi + 1) / 2;
    CompareScalar(mask, key, half(mid), CMPMODE::GE, n);
    PipeBarrier<PIPE_V>();
    GatherMask(tmp, key, mask.ReinterpretCast<uint16_t>(), true, n, gp, count);
    PipeBarrier<PIPE_ALL>();
    if (count >= static_cast<uint64_t>(k))
      lo = mid;
    else
      hi = mid - 1;
  }
  CompareScalar(mask, key, half(lo), CMPMODE::GT, n);
  CompareScalar(eqmask, key, half(lo), CMPMODE::EQ, n);
  PipeBarrier<PIPE_V>();
  Duplicate(chosen, half(lo), k);
  PipeBarrier<PIPE_V>();
  GatherMask(chosen, key, mask.ReinterpretCast<uint16_t>(), true, n, gp, count);
  PipeBarrier<PIPE_ALL>();
  int above = static_cast<int>(count);
  auto ids = tmp.ReinterpretCast<int16_t>();
  GlobalTensor<int16_t> indexGm;
  indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(indexp));
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
  DataCopy(ids, indexGm, n);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  auto idxf = Local<float>(extra + 6144, 512);
  GatherMask(chosenidx, ids, mask.ReinterpretCast<uint16_t>(), true, n, gp, count);
  PipeBarrier<PIPE_V>();
  Cast(idxf, chosenidx, RoundMode::CAST_NONE, k);
  PipeBarrier<PIPE_V>();
  Cast(outi, idxf, RoundMode::CAST_RINT, k);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  if (above)
    DataCopyPad(i[static_cast<int64_t>(row) * k], outi, {1, static_cast<uint32_t>(above * 4), 0, 0, 0});
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  auto eqids = key.ReinterpretCast<int16_t>();
  GatherMask(eqids, ids, eqmask.ReinterpretCast<uint16_t>(), true, n, gp, count);
  PipeBarrier<PIPE_V>();
  Cast(idxf, eqids, RoundMode::CAST_NONE, k);
  PipeBarrier<PIPE_V>();
  Cast(outi, idxf, RoundMode::CAST_RINT, k);
  Cast(outv, chosen, RoundMode::CAST_NONE, k);
  SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
  DataCopyPad(i[static_cast<int64_t>(row) * k + above],
              outi,
              {1, static_cast<uint32_t>((k - above) * 4), 0, 0, 0});
  DataCopy(v[static_cast<int64_t>(row) * k], outv, k);
  PipeBarrier<PIPE_ALL>();
}

// Merge only the requested prefix of each sorted run. Discarded suffixes
// cannot enter a global top-k, even when there are ties.
extern "C"[aicore] __attribute__((always_inline)) void _mlir_ciface_topk_sort_prefix(
    int64_t values, int64_t indices, int64_t scratch, int32_t n, int32_t keep, int64_t output) {
  using namespace AscendC;
  auto v = Local<float>(values, n);
  auto ids = Local<uint32_t>(indices, n);
  auto tmp = Local<float>(scratch, 2 * n);
  auto dst = Local<float>(output, 2 * n);
  PipeBarrier<PIPE_ALL>();
  Sort32(dst, v, ids, n / 32);
  PipeBarrier<PIPE_V>();
  int groups = n / 32;
  DataCopy(tmp,
           dst,
           {static_cast<uint16_t>(groups),
            static_cast<uint16_t>(keep / 4),
            static_cast<uint16_t>((32 - keep) / 4),
            0});
  PipeBarrier<PIPE_V>();
  const uint16_t lengths[4] = {static_cast<uint16_t>(keep),
                               static_cast<uint16_t>(keep),
                               static_cast<uint16_t>(keep),
                               static_cast<uint16_t>(keep)};
  while (groups > 1) {
    int full = groups / 4, tail = groups % 4;
    if (full) {
      MrgSortSrcList<float> lists(tmp, tmp[2 * keep], tmp[4 * keep], tmp[6 * keep]);
      MrgSort(dst, lists, MrgSort4Info(lengths, false, 15, full));
    }
    if (tail == 2) {
      int start = full * 8 * keep;
      MrgSortSrcList<float> lists(tmp[start], tmp[start + 2 * keep], tmp[start], tmp[start]);
      const uint16_t tailLengths[4] = {static_cast<uint16_t>(keep), static_cast<uint16_t>(keep), 0, 0};
      MrgSort(dst[start], lists, MrgSort4Info(tailLengths, false, 3, 1));
    }
    PipeBarrier<PIPE_V>();
    groups = full + (tail != 0);
    if (groups == 1) break;
    if (full)
      DataCopy(tmp,
               dst,
               {static_cast<uint16_t>(full),
                static_cast<uint16_t>(keep / 4),
                static_cast<uint16_t>(3 * keep / 4),
                0});
    if (tail) DataCopy(tmp[full * 2 * keep], dst[full * 8 * keep], 2 * keep);
    PipeBarrier<PIPE_V>();
  }
  PipeBarrier<PIPE_ALL>();
}
