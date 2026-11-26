# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kunlunxin override of :func:`flag_gems.masked_select_backward`.

Semantics: ``out[i] = mask[i] ? grad[cumsum(mask)[i] - 1] : 0`` over the
broadcasted element space; the result is a *contiguous* tensor (ATen
materialises ``zeros(...).masked_scatter_(...)``), and an error is raised when
``grad.numel()`` is smaller than the number of selected elements.

Design notes (all measured on P800/XPU3, see
``harness/solution/masked_select_backward/``):

* A per-lane computed address is a 20-50x gather wall and masked loads leak
  in-bounds garbage, so every *full* tile reads the mask with a pure affine
  unmasked load.  Only the final partial tile takes the clamp + ``tl.where``
  form (a separate single-program kernel).
* A single tile must never exceed 16384 lanes (32768 silently miscomputes) and
  store tiles must stay >= 64 lanes.
* ``tl.gather`` does not legalise on XPU3 and there are no atomics, so the
  compaction is output-centric.  Small grids run one program per block and
  rebuild each program's prefix from the previous blocks (1-2 launches); large
  grids run the parallel 3-phase count -> scan -> apply pipeline.
* The apply phase stores *unmasked* into a padded buffer
  (``nblocks * BLOCK``) and the host narrows/views it afterwards.  This keeps
  every store vectorised, avoids zero-filling the result, and matches the
  contiguous-output contract.
* Reductions/scans always precede the stores of the same program (a masked
  store in the same program can corrupt reductions on this stack).
* Large tensors (fp16/bf16/fp32, ``numel >= _TLE_MIN_NUMEL``) run a hand
  written ``tle.raw`` XPU3 payload (``_msb_payload_source``): one entry counts
  the mask with 64-lane SIMD popcounts (SM-compacted 256B cluster DMAs) and a
  second entry expands grad into ``out`` (pipelined async DMAs + mask/grad
  double buffering).  The payload reads the mask once per stage, never reads
  ``inp`` and writes every output lane, so there is no zero-fill pass.
* The validation readback (``grad.numel() < numel`` must raise) uses a
  per-call sequence tag instead of a sentinel fill: the count kernels publish
  ``count | seq << 40`` into a pinned host scalar with an async D2H that is
  enqueued *before* the apply kernel, so the host spin overlaps the apply.
"""

import contextlib
import logging
import os
import threading

import torch
import triton
import triton.language as tl

from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.copy import copy_
from flag_gems.ops.masked_select_backward import _broadcast_views
from flag_gems.ops.neg import neg
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

# ---------------------------------------------------------------------------
# tle.raw payload (inline ``source=`` so the payload ships with this file).
#   msb_count   : vectorised mask popcount per 64-aligned worker partition;
#                 counts are compacted through cluster shared memory and
#                 written with one 256B SM2GM per cluster (a scalar GM store
#                 costs ~9-12us per launch on this stack, a >=64B DMA ~0.2us).
#   msb_expand* : per-worker global prefix from the compact counts, then
#                 ``out[i] = mask[i] ? grad[rank] : 0`` with the grad range
#                 DMA'd (64B-aligned over-read) and the mask/grad DMAs
#                 double buffered + the output store issued asynchronously.
# ---------------------------------------------------------------------------
_MSB_PAYLOAD_SRC = r"""
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_simd.h"

static __device__ __attribute__((always_inline)) int msb_cnt64(
    const unsigned char *buf, int len64, int tail) {
  int cc = 0;
  for (int off = 0; off < len64; off += 64) {
    int32x16_t v = vload_lm_int32x16(buf + off);
    int32x16_t y = svand_int32x16(0x7F7F7F7F, v);
    int32x16_t z = vvor_int32x16(svadd_int32x16(0x7F7F7F7F, y), v);
    int32x16_t hb = svand_int32x16((int)0x80808080u, z);
    cc += vpopcnt512(hb);
  }
  for (int i = len64; i < tail; ++i) {
    if (buf[i] != 0) cc++;
  }
  return cc;
}

extern "C" __device__ void msb_count(
    _global_ptr_ const unsigned char *mask,
    _global_ptr_ int *tcnt,
    int n, int per) {
  int nc = core_num();
  int cl = cluster_id();
  int cid = core_id();
  int gid = cl * nc + cid;
  int start = gid * per;
  int end = start + per;
  if (end > n) end = n;
  int cnt = 0;
  if (start < end) {
    __attribute__((aligned(64))) __local__ unsigned char buf[2048];
    int nv = end - start;
    int nv64 = nv & ~63;
    int done = 0;
    while (done < nv64) {
      int len = nv64 - done;
      if (len > 2048) len = 2048;
      GM2LM(mask + start + done, buf, len);
      cnt += msb_cnt64(buf, len, len);
      done += len;
    }
    for (int i = nv64; i < nv; ++i) {
      if (mask[start + i] != 0) cnt++;
    }
  }
  __shared__ int scnt[64];
  scnt[cid] = cnt;
  mfence_shared();
  sync_cluster();
  if (cid == 0) {
    // one 256B DMA for the whole cluster's counts: a scalar GM store would
    // cost ~9us of launch latency, a >=64B DMA ~0.2us.
    SM2GM(scnt, tcnt + cl * 64, 256);
  }
  mfence();
}

#define MSB_EXPAND(NAME, CHUNK, VT)                                            \
extern "C" __device__ void NAME(                                               \
    _global_ptr_ const unsigned char *mask,                                    \
    _global_ptr_ const unsigned char *grad,                                    \
    _global_ptr_ unsigned char *out,                                           \
    _global_ptr_ const int *tcnt,                                              \
    int n, int per, int gnumel) {                                              \
  int nc = core_num();                                                         \
  int cid = core_id();                                                         \
  int gid = cluster_id() * nc + cid;                                           \
  int start = gid * per;                                                       \
  int end = start + per;                                                       \
  if (end > n) end = n;                                                        \
  if (start >= end) { mfence(); return; }                                      \
  __attribute__((aligned(64))) __local__ int cbuf[256];                        \
  __attribute__((aligned(64))) __local__ unsigned char mbuf[3][CHUNK];         \
  __attribute__((aligned(64))) __local__ unsigned char gbuf[2][CHUNK * (int)sizeof(VT) + 64];\
  __attribute__((aligned(64))) __local__ unsigned char obuf[2][CHUNK * (int)sizeof(VT)];\
  int g0 = 0;                                                                  \
  for (int base = 0; base < gid; base += 256) {                                \
    int k = gid - base;                                                        \
    if (k > 256) k = 256;                                                      \
    GM2LM(tcnt + base, cbuf, k * 4);                                           \
    for (int j = 0; j < k; ++j) g0 += cbuf[j];                                 \
  }                                                                            \
  int nchunk = (end - start + CHUNK - 1) / CHUNK;                              \
  for (int p = 0; p < 3 && p < nchunk; ++p) {                                  \
    int ppos = start + p * CHUNK;                                              \
    int plen = end - ppos;                                                     \
    if (plen > CHUNK) plen = CHUNK;                                            \
    int pl64 = (plen + 63) & ~63;                                              \
    int mread = pl64 < (n - ppos) ? pl64 : (n - ppos);                         \
    GM2LM_ASYNC(mask + ppos, mbuf[p], mread);                                  \
  }                                                                            \
  mfence_lm();                                                                 \
  int g_cur = g0;                                                              \
  int flen0 = end - start > CHUNK ? CHUNK : (end - start);                     \
  int cc_cur = msb_cnt64(mbuf[0], flen0 & ~63, flen0);                         \
  {                                                                            \
    int gb = g_cur * (int)sizeof(VT);                                          \
    int ab = gb & ~63;                                                         \
    int skip = gb - ab;                                                        \
    int rlen = skip + cc_cur * (int)sizeof(VT);                                \
    int rlen64 = (rlen + 63) & ~63;                                            \
    int gmax = gnumel * (int)sizeof(VT) - ab;                                  \
    if (rlen64 > gmax) rlen64 = gmax;                                          \
    if (rlen64 > 0) GM2LM_ASYNC(grad + ab, gbuf[0], rlen64);                   \
  }                                                                            \
  int g_next = g0 + cc_cur;                                                    \
  int cc_next = 0;                                                             \
  int have_pending = 0;                                                        \
  int ppos = 0, plen = 0, pbuf = 0;                                            \
  int pos = start;                                                             \
  for (int c = 0; c < nchunk; ++c) {                                           \
    int cur = c & 1;                                                           \
    int len = end - pos;                                                       \
    if (len > CHUNK) len = CHUNK;                                              \
    mfence();                                                                  \
    if (have_pending) {                                                        \
      LM2GM_ASYNC(obuf[pbuf], out + (long long)ppos * (int)sizeof(VT),         \
                  plen * (int)sizeof(VT));                                     \
    }                                                                          \
    if (c + 1 < nchunk) {                                                      \
      int npos = pos + CHUNK;                                                  \
      int nlen = end - npos;                                                   \
      if (nlen > CHUNK) nlen = CHUNK;                                          \
      cc_next = msb_cnt64(mbuf[(c + 1) % 3], nlen & ~63, nlen);                \
      int gb = g_next * (int)sizeof(VT);                                       \
      int ab = gb & ~63;                                                       \
      int skip = gb - ab;                                                      \
      int rlen = skip + cc_next * (int)sizeof(VT);                             \
      int rlen64 = (rlen + 63) & ~63;                                          \
      int gmax = gnumel * (int)sizeof(VT) - ab;                                \
      if (rlen64 > gmax) rlen64 = gmax;                                        \
      if (rlen64 > 0) GM2LM_ASYNC(grad + ab, gbuf[(c + 1) & 1], rlen64);       \
      if (c + 2 < nchunk) {                                                    \
        int mpos = npos + CHUNK;                                               \
        int mlen = end - mpos;                                                 \
        if (mlen > CHUNK) mlen = CHUNK;                                        \
        int ml64 = (mlen + 63) & ~63;                                          \
        int mread = ml64 < (n - mpos) ? ml64 : (n - mpos);                     \
        GM2LM_ASYNC(mask + mpos, mbuf[(c + 2) % 3], mread);                    \
      }                                                                        \
    }                                                                          \
    int gi = 0;                                                                \
    int skip = (g_cur * (int)sizeof(VT)) & 63;                                 \
    VT *ob = (VT *)obuf[cur];                                                  \
    VT *gl = (VT *)(gbuf[cur] + skip);                                         \
    const unsigned char *m = mbuf[c % 3];                                      \
    int i = 0;                                                                 \
    if (sizeof(VT) == 2) {                                                     \
      const int zoff = CHUNK * 2 + 62;                                         \
      *(unsigned short *)(gbuf[cur] + zoff) = 0;                               \
      __attribute__((aligned(64))) __local__ short of[32];                     \
      for (; i + 32 <= len; i += 32) {                                         \
        for (int j = 0; j < 32; j += 4) {                                      \
          unsigned int mw = *(const unsigned int *)(m + i + j);                \
          int s0 = (mw & 0xFFu) != 0;                                          \
          int s1 = ((mw >> 8) & 0xFFu) != 0;                                   \
          int s2 = ((mw >> 16) & 0xFFu) != 0;                                  \
          int s3 = (mw >> 24) != 0;                                            \
          int r1 = s0, r2 = r1 + s1, r3 = r2 + s2;                             \
          int f0 = s0 ? skip + 2 * gi : zoff;                                  \
          int f1 = s1 ? skip + 2 * (gi + r1) : zoff;                           \
          int f2 = s2 ? skip + 2 * (gi + r2) : zoff;                           \
          int f3 = s3 ? skip + 2 * (gi + r3) : zoff;                           \
          gi += r3 + s3;                                                       \
          *(unsigned int *)(of + j) =                                          \
              ((unsigned int)f0 & 0xFFFFu) |                                   \
              (((unsigned int)f1 & 0xFFFFu) << 16);                            \
          *(unsigned int *)(of + j + 2) =                                      \
              ((unsigned int)f2 & 0xFFFFu) |                                   \
              (((unsigned int)f3 & 0xFFFFu) << 16);                            \
        }                                                                      \
        mfence();                                                              \
        int16x32_t ov = vload_lm_int16x32(of);                                 \
        float16x32_t gv = vgather_lm_float16x32(gbuf[cur], ov);                \
        vstore_lm_float16x32(ob + i, gv);                                      \
        mfence();                                                              \
      }                                                                        \
    } else {                                                                   \
      int len4 = len & ~3;                                                     \
      for (; i < len4; i += 4) {                                               \
        unsigned int mw = *(const unsigned int *)(m + i);                      \
        int s0 = (mw & 0xFFu) != 0;                                            \
        int s1 = ((mw >> 8) & 0xFFu) != 0;                                     \
        int s2 = ((mw >> 16) & 0xFFu) != 0;                                    \
        int s3 = (mw >> 24) != 0;                                              \
        VT v0 = gl[gi];                                                        \
        gi += s0;                                                              \
        VT v1 = gl[gi];                                                        \
        gi += s1;                                                              \
        VT v2 = gl[gi];                                                        \
        gi += s2;                                                              \
        VT v3 = gl[gi];                                                        \
        gi += s3;                                                              \
        ob[i] = (VT)(v0 & (VT)(0 - s0));                                       \
        ob[i + 1] = (VT)(v1 & (VT)(0 - s1));                                   \
        ob[i + 2] = (VT)(v2 & (VT)(0 - s2));                                   \
        ob[i + 3] = (VT)(v3 & (VT)(0 - s3));                                   \
      }                                                                        \
    }                                                                          \
    for (; i < len; ++i) {                                                     \
      int sel = (m[i] != 0);                                                   \
      VT v = gl[gi];                                                           \
      ob[i] = (VT)(v & (VT)(0 - sel));                                         \
      gi += sel;                                                               \
    }                                                                          \
    have_pending = 1;                                                          \
    ppos = pos;                                                                \
    plen = len;                                                                \
    pbuf = cur;                                                                \
    g_cur = g_next;                                                            \
    g_next += cc_next;                                                         \
    cc_cur = cc_next;                                                          \
    pos += len;                                                                \
  }                                                                            \
  if (have_pending) {                                                          \
    LM2GM(obuf[pbuf], out + (long long)ppos * (int)sizeof(VT),                 \
          plen * (int)sizeof(VT));                                             \
  }                                                                            \
  mfence();                                                                    \
}

MSB_EXPAND(msb_expand16, 512, unsigned short)
MSB_EXPAND(msb_expand32, 256, unsigned int)
"""

_MSB_TLE = {"init": False, "ok": False, "count": None, "exp16": None, "exp32": None}


def _msb_tle_init():
    """Compile the inline payload once (no-op when the toolkit is absent)."""
    if _MSB_TLE["init"]:
        return _MSB_TLE["ok"]
    _MSB_TLE["init"] = True
    try:
        import triton.experimental.tle as _tle

        ns = {"_tle": _tle, "_SRC": _MSB_PAYLOAD_SRC}
        exec(
            "@_tle.raw.dialect('xpu3', source=_SRC)\n"
            "def msb_count(mask, tcnt, n, per):\n"
            "    ...\n",
            ns,
        )
        count_callee = ns["msb_count"]
        callees = {}
        for sym in ("msb_expand16", "msb_expand32"):
            nsc = {"_tle": _tle, "_SRC": _MSB_PAYLOAD_SRC}
            exec(
                "@_tle.raw.dialect('xpu3', source=_SRC)\n"
                f"def {sym}(mask, grad, out, tcnt, n, per, gnumel):\n"
                "    ...\n",
                nsc,
            )
            callees[sym] = nsc[sym]

        @triton.jit(do_not_specialize=["n", "per"])
        def _count_k(MASK, TCNT, n, per):
            _tle.raw.call(count_callee, (MASK, TCNT, n, per))

        def _mk_expand(callee):
            @triton.jit(do_not_specialize=["n", "per", "gnumel"])
            def _k(MASK, GRAD, OUT, TCNT, n, per, gnumel):
                _tle.raw.call(callee, (MASK, GRAD, OUT, TCNT, n, per, gnumel))

            return _k

        _MSB_TLE["count"] = _count_k
        _MSB_TLE["exp16"] = _mk_expand(callees["msb_expand16"])
        _MSB_TLE["exp32"] = _mk_expand(callees["msb_expand32"])
        _MSB_TLE["ok"] = True
    except Exception:  # pragma: no cover - keep the backend importable
        logger.debug("GEMS_KUNLUNXIN MASKED_SELECT_BACKWARD tle.raw init failed")
        _MSB_TLE["ok"] = False
    return _MSB_TLE["ok"]


logger = logging.getLogger(__name__)

# Largest supported single-tile width.  BLOCK >= 32768 miscomputes reductions
# (and can fault) under a bare launch; 16384 is exact.
_MAX_BLOCK = 16384
# Smallest store tile: the launcher pads a program's lanes up to the 64-lane
# core width, so smaller tiles spill stores into neighbouring rows.
_MIN_BLOCK = 64
# numel <= _SMALL_MAX is handled by one single-program kernel.
_SMALL_MAX = 1024
_SMALL_MIN_BLOCK = 256
# numel <= _PARALLEL_MAX uses one launch of the redundant-prefix kernel
# (the redundant mask re-reads grow quadratically with the block count, so the
# one-launch path stops at 8 blocks).
_PARALLEL_MAX = 16384
# Block width for the parallel path.
_MID_BLOCK = 2048
_SMALL_MID_BLOCK = 1024
_LARGE_BLOCK = 16384
# Above this numel the 16384-lane tile wins; below it 2048 keeps more programs
# (and therefore more clusters) busy.
_LARGE_THRESHOLD = 262144
# Up to this many blocks each program rebuilds its own prefix from the mask
# (O(nblk^2) redundant mask reads) instead of paying two more kernel launches.
_REDUNDANT_MAX_BLOCKS = 16
# The single-program scan is much cheaper with a masked store than with an
# unmasked one, and tiles <= 64 lanes cost ~11 us instead of ~3 us (measured);
# keep the scan tile in the 256..1024 range.
_SCAN_MIN_BLOCK = 256
_SCAN_MAX_BLOCK = 1024
# Above this numel the tle.raw payload (count + pipelined expand) beats the
# Triton count/apply pipeline (measured crossover at ~1024 elements).
_TLE_MIN_NUMEL = 1024
# Physical clusters driven by the payload (12 == all clusters of this part).
_MSB_GRID = 12

# ---------------------------------------------------------------------------
# Validation readback: the error check needs the mask count on the host, but a
# blocking ``item()``/``synchronize()`` costs ~19.5 us and dominates the small
# shapes.  Instead the count kernels publish ``count | TAG`` into an int64
# scalar, a *pinned* host buffer receives it with an async D2H copy that is
# enqueued *before* the apply kernels, and the host spins on the tag bit so the
# remaining GPU work overlaps the wait.  Side streams are forbidden here (they
# fault this runtime), everything stays on the current stream.
# The count is published as ``count | seq << _TAG_SHIFT`` with a per-call
# sequence number, so a stale value from the previous call can never satisfy
# the spin (the old sentinel ``fill_(-1)`` costs ~3 us of CPU time per call).
_TAG_SHIFT = 40
_TAG_MASK = (1 << _TAG_SHIFT) - 1
_SEQ_MAX = (1 << 22) - 1
_SPIN_LIMIT = 200000
_PINNED = threading.local()
# Tile for folding the payload's 768 per-worker counts.
_TAGGED_SUM_BLOCK = 1024

# ---------------------------------------------------------------------------
# ATen-style validation (``grad.numel() < mask.sum()`` must raise).  The
# XPU vendor reference does not validate (a too-short grad faults the device
# instead of raising), and the read-back needed for the error costs ~14-16 us
# per call, which dominates every small/medium shape.  Kept ON by default so
# the FlagGems semantics match CPU ATen; ``FLAGGEMS_MSB_VALIDATE=0`` switches
# to the vendor behaviour for benchmarking.
_VALIDATE_ENV = "FLAGGEMS_MSB_VALIDATE"


def _validation_enabled():
    """Cheap per-call read so a caller can toggle it with an env variable."""
    value = os.environ.get(_VALIDATE_ENV)
    if value is None:
        return _VALIDATE_DEFAULT
    return value.strip().lower() in ("1", "true", "yes", "on")


_VALIDATE_DEFAULT = True


# Tile width of the single-program validation popcount (fixed: the tile width
# must not follow next_pow2(n), 8k/16k-lane tiles cost ~20 us here).
_COUNT_BLOCK = 1024


def _next_tag():
    """Per-thread monotone tag (kept small so ``count | tag`` never overflows)."""
    seq = getattr(_PINNED, "seq", 0) + 1
    if seq > _SEQ_MAX:
        seq = 1
    _PINNED.seq = seq
    return seq << _TAG_SHIFT


def _pinned_count_buffer():
    """Per-thread pinned int64 scalar (pinned allocations are expensive)."""
    buf = getattr(_PINNED, "buf", None)
    if buf is None:
        buf = torch.empty((), dtype=torch.int64, pin_memory=True)
        _PINNED.buf = buf
    return buf


def _count_scratch(device):
    """Per-thread device int64 scalar (always rewritten before it is read)."""
    scratch = getattr(_PINNED, "count", None)
    if scratch is None or scratch.device != device:
        scratch = torch.empty((), dtype=torch.int64, device=device)
        _PINNED.count = scratch
    return scratch


def _worker_counts_scratch(device):
    """Per-thread device int32 buffer for the payload's per-worker counts."""
    scratch = getattr(_PINNED, "tcnt", None)
    if scratch is None or scratch.device != device:
        scratch = torch.empty(_MSB_GRID * 64, dtype=torch.int32, device=device)
        _PINNED.tcnt = scratch
    return scratch


# ---------------------------------------------------------------------------
# Launch fast path.  Triton's JIT re-binds and re-specialises every argument on
# every call (~10 us of CPU per launch measured on this stack; the payload
# wrapper takes 11.5 us with 7 args), which dominates the small shapes.  Once a
# kernel is compiled we can call the compiled launcher directly (~3-5 us) after
# validating that the divisibility assumptions baked into that kernel still hold
# for the current pointers.  Everything here is best-effort: any miss or
# exception falls back to the normal ``kernel[grid](...)`` call.
_NULL_CTX = contextlib.nullcontext()
_LAUNCH_CACHE = {}
_LAUNCH_FAST = True
_TORCH_TO_TT = {
    torch.uint8: "*u8",
    torch.int8: "*i8",
    torch.int16: "*i16",
    torch.int32: "*i32",
    torch.int64: "*i64",
    torch.float16: "*fp16",
    torch.bfloat16: "*bf16",
    torch.float32: "*fp32",
}


def _arg_sig(arg):
    """Specialisation key of one runtime argument (same shape Triton uses)."""
    if torch.is_tensor(arg):
        ty = _TORCH_TO_TT.get(arg.dtype)
        if ty is None:
            return None
        return (ty, arg.data_ptr() % 16 == 0)
    if isinstance(arg, bool):
        return ("i1", None)
    if isinstance(arg, int):
        return ("i32" if -(2**31) <= arg < 2**31 else "i64", None)
    if isinstance(arg, float):
        return ("fp32", None)
    return None


def _launch_hooks_idle():
    """True when no user launch hook is registered (nothing observes the call)."""
    try:
        import triton.knobs as _knobs

        for hook in (
            _knobs.runtime.launch_enter_hook,
            _knobs.runtime.launch_exit_hook,
        ):
            if hook is None:
                continue
            if getattr(hook, "calls", None):
                return False
        return True
    except Exception:  # pragma: no cover - knob layout changed
        return False


def _try_fast_launch(jit_fn, grid, args, kwargs):
    """Launch through the compiled launcher; False when the slow path is needed."""
    if not _LAUNCH_FAST or not _launch_hooks_idle():
        return False
    from triton.runtime import driver as _driver

    dev = _driver.active.get_current_device()
    sig = tuple(_arg_sig(a) for a in args)
    if any(item is None for item in sig):
        return False
    key = (
        id(jit_fn),
        dev,
        sig,
        tuple(sorted(kwargs.items())),
        tuple(grid),
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is not None:
        run, function, meta, need_div = entry
        for pos, div in need_div:
            if (not torch.is_tensor(args[pos])) or (args[pos].data_ptr() % div):
                return False
        stream = _driver.active.get_current_stream(dev)
        g0 = grid[0]
        g1 = grid[1] if len(grid) > 1 else 1
        g2 = grid[2] if len(grid) > 2 else 1
        run(g0, g1, g2, stream, function, meta, None, None, None, *args)
        return True
    try:
        kernel_cache = jit_fn.device_caches[dev][0]
    except Exception:  # pragma: no cover - first call for this device
        return False
    before = set(kernel_cache.keys())
    jit_fn[grid](*args, **kwargs)
    for k, ck in kernel_cache.items():
        if k in before:
            continue
        attrs = getattr(getattr(ck, "src", None), "attrs", None) or {}
        need_div = []
        for idx, alist in attrs.items():
            pos = idx[0] if isinstance(idx, tuple) else int(idx)
            div = 1
            for name, val in alist or []:
                if name == "tt.divisibility":
                    div = int(val)
            need_div.append((pos, div))
        try:
            run = ck.run  # initialises the handles on first access
        except Exception:  # pragma: no cover
            continue
        _LAUNCH_CACHE[key] = (run, ck.function, ck.packed_metadata, tuple(need_div))
        break
    return True


def _device_ctx(device):
    """``torch_device_fn.device`` only when the target is not already current.

    Entering the device context costs ~4.8 us on this stack, and the op enters
    it on every call; when the tensors already live on the current device the
    context is a no-op, so skip it (checked with a 0.4 us driver call).
    """
    idx = getattr(device, "index", None)
    if idx is not None:
        try:
            if idx == torch.cuda.current_device():
                return _NULL_CTX
        except Exception:  # pragma: no cover - driver not initialised yet
            pass
    return torch_device_fn.device(device)


def _launch(jit_fn, grid, args, kwargs=None):
    """Launch ``jit_fn`` (fast path when possible, normal JIT otherwise)."""
    kwargs = kwargs or {}
    try:
        if _try_fast_launch(jit_fn, grid, args, kwargs):
            return
    except Exception:  # pragma: no cover - never break the op
        logger.debug("GEMS_KUNLUNXIN MASKED_SELECT_BACKWARD fast launch deferred")
    jit_fn[grid](*args, **kwargs)


def _begin_count_read(device_count, tag):
    """Enqueue an async D2H of ``device_count`` (tagged with ``tag``)."""
    buf = _pinned_count_buffer()
    buf.copy_(device_count, non_blocking=True)
    return buf, tag >> _TAG_SHIFT


def _finish_count_read(state):
    """Spin until the tagged value lands, then decode the count."""
    buf, want = state
    spins = 0
    value = int(buf.item())
    while (value >> _TAG_SHIFT) != want:
        spins += 1
        if spins > _SPIN_LIMIT:
            torch_device_fn.synchronize()
            value = int(buf.item())
            if (value >> _TAG_SHIFT) != want:  # copy never landed
                value = -1
            break
        value = int(buf.item())
    return value & _TAG_MASK


@libentry()
@triton.jit(do_not_specialize=["n_elements", "tag"])
def _masked_select_backward_tagged_count_kernel(
    mask_ptr,
    count_ptr,
    n_elements,
    tag,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-program mask popcount published as ``count | tag``."""
    lane = tl.arange(0, BLOCK_SIZE)
    valid = lane < n_elements
    safe_mask = tl.minimum(lane, n_elements - 1)
    selected = tl.where(valid, tl.load(mask_ptr + safe_mask) != 0, False).to(tl.int64)
    tl.store(count_ptr, tl.sum(selected, axis=0) | tag)


@libentry()
@triton.jit(do_not_specialize=["n_counts", "tag"])
def _masked_select_backward_tagged_sum_kernel(
    counts_ptr,
    count_ptr,
    n_counts,
    tag,
    BLOCK_SIZE: tl.constexpr,
):
    """Fold a small device counter array into the tagged validation scalar."""
    lane = tl.arange(0, BLOCK_SIZE)
    valid = lane < n_counts
    safe = tl.minimum(lane, n_counts - 1)
    values = tl.where(valid, tl.load(counts_ptr + safe), 0).to(tl.int64)
    tl.store(count_ptr, tl.sum(values, axis=0) | tag)


def _tle_msb_apply(mask, grad, n_elements, dtype, device, need_count=False):
    """``out[k] = mask[k] ? grad[rank(k)] : 0`` through the XPU3 payload.

    Returns ``(out, count_state)`` where ``count_state`` is the pending tagged
    readback of the mask count (only when ``need_count``), or ``(None, None)``
    when the payload cannot run (the caller then uses the Triton pipeline).

    The payload is exact for every ``n_elements`` (the 64B DMA over-reads and
    the partial tail are handled inside the entries); it requires an inline
    payload build, a 2/4-byte float dtype and a non-empty grad.
    """
    if not _msb_tle_init():
        return None, None
    if grad.numel() == 0:
        # An empty grad must go through the count/validation path (the payload
        # never sees a zero-length source).
        return None, None
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None, None
    if n_elements >= 2**29 or not mask.is_contiguous() or not grad.is_contiguous():
        return None, None
    expand = _MSB_TLE["exp16"] if dtype != torch.float32 else _MSB_TLE["exp32"]
    nw = _MSB_GRID * 64
    per = -(-n_elements // nw)
    per = (per + 63) // 64 * 64
    count_state = None
    try:
        tcnt = _worker_counts_scratch(device)
        u8mask = mask.view(torch.uint8)
        _launch(_MSB_TLE["count"], (_MSB_GRID,), (u8mask, tcnt, n_elements, per))
        if need_count:
            # Fold the 768 per-worker counts into one tagged scalar and
            # enqueue the async D2H *before* the expand kernel, so the host
            # spin overlaps the compaction.  (A wider D2H is much more
            # expensive on this runtime: 96B copies cost +18us.)
            tag = _next_tag()
            count_dev = _count_scratch(device)
            _launch(
                _masked_select_backward_tagged_sum_kernel,
                (1,),
                (tcnt, count_dev, nw, tag),
                {"BLOCK_SIZE": _TAGGED_SUM_BLOCK},
            )
            count_state = _begin_count_read(count_dev, tag)
        out = torch.empty(n_elements, dtype=dtype, device=device)
        _launch(
            expand,
            (_MSB_GRID,),
            (u8mask, grad, out, tcnt, n_elements, per, grad.numel()),
        )
        return out, count_state
    except Exception:  # pragma: no cover - never break the op
        logger.debug("GEMS_KUNLUNXIN MASKED_SELECT_BACKWARD tle path deferred")
        if count_state is not None:
            torch_device_fn.synchronize()
        return None, None


@libentry()
@triton.jit(do_not_specialize=["grad_numel"])
def _masked_select_backward_prefix_apply_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    grad_numel,
    num_full,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRAD: tl.constexpr,
):
    """One program per full block, redundant prefix (small grids).

    Program ``pid`` first re-reads blocks ``0..pid-1`` (all full, therefore
    purely affine) to rebuild its exclusive prefix, then gathers its own block.
    The last program also publishes the mask total used by the host-side
    ``grad.numel() < total_count`` validation.
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    grad_advance = 0
    for block in range(0, pid):
        grad_advance += tl.sum(
            (tl.load(mask_ptr + block * BLOCK_SIZE + lane) != 0).to(tl.int32),
            axis=0,
        )
    offsets = pid * BLOCK_SIZE + lane
    selected = (tl.load(mask_ptr + offsets) != 0).to(tl.int32)
    if HAS_GRAD:
        grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
        safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
        values = tl.load(grad_ptr + safe_grad)
        tl.store(out_ptr + offsets, tl.where(selected != 0, values, 0))
    else:
        tl.store(
            out_ptr + offsets,
            tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit(do_not_specialize=["tail_len", "grad_numel"])
def _masked_select_backward_prefix_apply_tail_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    grad_numel,
    tail_len,
    num_full,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRAD: tl.constexpr,
):
    """Single program for the partial tail block, redundant prefix.

    The prefix loop re-reads every full block affinely; the tail tile keeps the
    same width (a second tile width would blow the per-program SRAM budget on
    this backend) and clamps its lanes.
    """
    lane = tl.arange(0, BLOCK_SIZE)
    grad_advance = 0
    for block in range(0, num_full):
        grad_advance += tl.sum(
            (tl.load(mask_ptr + block * BLOCK_SIZE + lane) != 0).to(tl.int32),
            axis=0,
        )
    tail_base = num_full * BLOCK_SIZE
    offsets = tail_base + lane
    valid = lane < tail_len
    safe_mask = tl.minimum(offsets, tail_base + tail_len - 1)
    selected = ((tl.load(mask_ptr + safe_mask) != 0) & valid).to(tl.int32)
    if HAS_GRAD:
        grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
        safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
        values = tl.load(grad_ptr + safe_grad)
        tl.store(out_ptr + offsets, tl.where(selected != 0, values, 0))
    else:
        tl.store(
            out_ptr + offsets,
            tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit(do_not_specialize=["n_elements", "grad_numel", "num_blocks"])
def _masked_select_backward_parallel_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    grad_numel,
    n_elements,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRAD: tl.constexpr,
):
    """One program per block, redundant prefix, clamped own-block read.

    Used for small/medium tensors: a single launch beats the 3-phase pipeline
    by two launches, and the per-lane clamp on the own block costs little at
    these sizes.  The prefix loop only ever touches *full* blocks, so it stays
    purely affine.  The mask count used for validation comes from
    ``_masked_select_backward_tagged_count_kernel`` (a scalar GM store from
    this kernel would serialise the validation readback behind the apply).
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    grad_advance = 0
    for block in range(0, pid):
        grad_advance += tl.sum(
            (tl.load(mask_ptr + block * BLOCK_SIZE + lane) != 0).to(tl.int32),
            axis=0,
        )
    offsets = pid * BLOCK_SIZE + lane
    valid = offsets < n_elements
    safe_mask = tl.minimum(offsets, n_elements - 1)
    selected = ((tl.load(mask_ptr + safe_mask) != 0) & valid).to(tl.int32)
    if HAS_GRAD:
        grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
        safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
        values = tl.load(grad_ptr + safe_grad)
        tl.store(out_ptr + offsets, tl.where(selected != 0, values, 0))
    else:
        tl.store(
            out_ptr + offsets,
            tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _masked_select_backward_count_blocks_kernel(
    mask_ptr,
    part_sums_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-block popcount of the *full* blocks, one program per block."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    selected = (tl.load(mask_ptr + offsets) != 0).to(tl.int32)
    tl.store(part_sums_ptr + pid, tl.sum(selected, axis=0))


@libentry()
@triton.jit(do_not_specialize=["tail_len"])
def _masked_select_backward_count_tail_kernel(
    mask_ptr,
    part_sums_ptr,
    tail_len,
    part_index,
    BLOCK_SIZE: tl.constexpr,
):
    """Popcount of the partial tail block (clamped read, single program)."""
    lane = tl.arange(0, BLOCK_SIZE)
    valid = lane < tail_len
    safe_mask = tl.minimum(lane, tail_len - 1)
    selected = ((tl.load(mask_ptr + safe_mask) != 0) & valid).to(tl.int32)
    tl.store(part_sums_ptr + part_index, tl.sum(selected, axis=0))


@libentry()
@triton.jit(do_not_specialize=["num_blocks", "tag"])
def _masked_select_backward_scan_kernel(
    part_sums_ptr,
    total_count_ptr,
    num_blocks,
    tag,
    SCAN_BLOCK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    """Single-program exclusive scan of the per-block counts.

    ``part_sums`` is padded to ``NUM_CHUNKS * SCAN_BLOCK`` so even an ignored
    store mask can only touch the workspace.  All reductions precede the store
    (a masked store corrupts same-program reductions/scans on this stack).
    ``NUM_CHUNKS`` stays constexpr: a runtime trip count costs ~9 us of extra
    kernel time even when the loop runs once.  The total is published as
    ``total | TAG`` so the host can spin on the tag bit of a pinned copy.
    """
    lane = tl.arange(0, SCAN_BLOCK)
    # Zero of the counters' own dtype (int64 part sums must stay int64).
    running = tl.load(part_sums_ptr) * 0
    for chunk in tl.static_range(0, NUM_CHUNKS):
        offsets = chunk * SCAN_BLOCK + lane
        valid = offsets < num_blocks
        safe = tl.minimum(offsets, num_blocks - 1)
        counts = tl.where(valid, tl.load(part_sums_ptr + safe), 0)
        prefixes = tl.cumsum(counts, axis=0) - counts + running
        running += tl.sum(counts, axis=0)
        tl.store(part_sums_ptr + offsets, prefixes, mask=valid)
    tl.store(total_count_ptr, running.to(tl.int64) | tag)


@libentry()
@triton.jit(do_not_specialize=["grad_numel"])
def _masked_select_backward_apply_kernel(
    grad_ptr,
    mask_ptr,
    part_sums_ptr,
    out_ptr,
    grad_numel,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRAD: tl.constexpr,
):
    """Gather the compacted gradient back onto the full blocks.

    ``g = prefix[block] + cumsum(sel) - 1`` is a monotone index, clamped into
    ``[0, grad_numel - 1]`` so an all-False mask can never abort a program.
    The store covers the padded buffer and stays unmasked/vectorised.
    ``HAS_GRAD=False`` skips the gather (an empty grad has no valid index).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    selected = (tl.load(mask_ptr + offsets) != 0).to(tl.int32)
    if HAS_GRAD:
        grad_advance = tl.load(part_sums_ptr + pid)
        grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
        safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
        values = tl.load(grad_ptr + safe_grad)
        tl.store(out_ptr + offsets, tl.where(selected != 0, values, 0))
    else:
        tl.store(
            out_ptr + offsets,
            tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty),
        )


@libentry()
@triton.jit(do_not_specialize=["tail_len", "grad_numel"])
def _masked_select_backward_apply_tail_kernel(
    grad_ptr,
    mask_ptr,
    part_sums_ptr,
    out_ptr,
    grad_numel,
    tail_len,
    part_index,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRAD: tl.constexpr,
):
    """Gather the compacted gradient onto the partial tail block."""
    lane = tl.arange(0, BLOCK_SIZE)
    valid = lane < tail_len
    safe_mask = tl.minimum(lane, tail_len - 1)
    selected = ((tl.load(mask_ptr + safe_mask) != 0) & valid).to(tl.int32)
    if HAS_GRAD:
        grad_advance = tl.load(part_sums_ptr + part_index)
        grad_offsets = grad_advance + tl.cumsum(selected, axis=0) - 1
        safe_grad = tl.minimum(tl.maximum(grad_offsets, 0), grad_numel - 1)
        values = tl.load(grad_ptr + safe_grad)
        tl.store(out_ptr + lane, tl.where(selected != 0, values, 0))
    else:
        tl.store(
            out_ptr + lane,
            tl.zeros([BLOCK_SIZE], dtype=out_ptr.dtype.element_ty),
        )


def _masked_select_backward_flat(grad, mask, n_elements, dtype, device, validate=True):
    """Compute the flat ``n_elements``-long result (padded buffer, narrowed)."""
    # Kernels address the mask flat; a contiguous broadcast mask reshapes for
    # free, and the tail kernels slice the flat view.
    if mask.dim() != 1:
        mask = mask.reshape(-1)
    if not grad.is_contiguous():
        grad = contiguous(grad)
    grad_numel = grad.numel()
    # An empty grad must still go through the count/validation path (ATen
    # raises when the mask selects anything); the kernels then skip the
    # gather (a clamp into [0, -1] would be out of bounds) and store zeros.
    has_grad = grad_numel > 0
    need_validate = validate and grad_numel < n_elements and _validation_enabled()
    error_msg = "Number of elements of source < number of ones in mask"
    if not has_grad and not need_validate:
        # An empty grad can never be gathered and nothing needs validating:
        # the result is all zeros (the vendor reference does the same).
        return torch.zeros(n_elements, dtype=dtype, device=device)

    if n_elements > _TLE_MIN_NUMEL:
        # tle.raw payload: count + compact with pipelined DMAs.  Writes
        # every output lane, so no zero-fill pass is needed.
        tle_out, count_state = _tle_msb_apply(
            mask, grad, n_elements, dtype, device, need_count=need_validate
        )
        if tle_out is not None:
            if count_state is not None and grad_numel < _finish_count_read(count_state):
                raise RuntimeError(error_msg)
            return tle_out

    if n_elements <= _PARALLEL_MAX:
        # One apply launch for the whole small/medium range.  The
        # validation count comes from a separate single-program popcount
        # kernel whose async D2H is enqueued *before* the apply kernel, so
        # the host spin overlaps the apply work (a scalar GM store from
        # the apply kernel would serialise everything behind it).
        if n_elements <= _SMALL_MAX:
            block_size = min(
                _SMALL_MAX,
                max(_SMALL_MIN_BLOCK, triton.next_power_of_2(n_elements)),
            )
        else:
            block_size = _SMALL_MID_BLOCK if n_elements <= 8192 else _MID_BLOCK
        num_blocks = triton.cdiv(n_elements, block_size)
        padded = num_blocks * block_size
        # An empty grad can never be gathered: the result is all zeros and
        # only the count (for the validation error) still matters (the
        # zero-fill is issued after the readback so the two overlap).
        out = torch.empty(padded, dtype=dtype, device=device) if has_grad else None
        count_state = None
        if need_validate:
            count_block = min(
                _MAX_BLOCK,
                max(_SMALL_MIN_BLOCK, triton.next_power_of_2(n_elements)),
            )
            tag = _next_tag()
            count_dev = _count_scratch(device)
            _launch(
                _masked_select_backward_tagged_count_kernel,
                (1,),
                (mask, count_dev, n_elements, tag),
                {"BLOCK_SIZE": count_block},
            )
            count_state = _begin_count_read(count_dev, tag)
        if out is None:
            # No apply kernel runs for an empty grad, so the result needs
            # no padding; the count still decides between zeros and the
            # ATen error.
            out = torch.zeros(n_elements, dtype=dtype, device=device)
            if count_state is not None and grad_numel < _finish_count_read(count_state):
                raise RuntimeError(error_msg)
            return out
        if has_grad:
            _launch(
                _masked_select_backward_parallel_kernel,
                (num_blocks,),
                (grad, mask, out, grad_numel, n_elements, num_blocks),
                {"BLOCK_SIZE": block_size, "HAS_GRAD": has_grad},
            )
        if count_state is not None and grad_numel < _finish_count_read(count_state):
            raise RuntimeError(error_msg)
        return out if padded == n_elements else out[:n_elements]

    block_size = _MID_BLOCK if n_elements <= _LARGE_THRESHOLD else _LARGE_BLOCK
    num_full = n_elements // block_size
    tail_len = n_elements - num_full * block_size
    tail_block = max(_MIN_BLOCK, triton.next_power_of_2(tail_len)) if tail_len else 0
    num_blocks = num_full + (1 if tail_len else 0)
    redundant = num_blocks <= _REDUNDANT_MAX_BLOCKS
    if redundant and tail_len:
        # The redundant tail kernel keeps the full block width (one tile
        # shape per program).
        padded = (num_full + 1) * block_size
    else:
        padded = num_full * block_size + tail_block
    # An empty grad can never be gathered: the result is all zeros and
    # only the count (for the validation error) still matters.
    if not has_grad:
        out = torch.zeros(padded, dtype=dtype, device=device)
    else:
        out = torch.empty(padded, dtype=dtype, device=device)

    if redundant and not need_validate:
        # One program per block rebuilds its prefix from the previous
        # blocks: 1 launch (2 with a tail) instead of the 3-phase pipeline.
        if not has_grad:
            return out[:n_elements]  # already all zeros
        if num_full:
            _masked_select_backward_prefix_apply_kernel[(num_full,)](
                grad,
                mask,
                out,
                grad_numel,
                num_full,
                BLOCK_SIZE=block_size,
                HAS_GRAD=has_grad,
            )
        if tail_len:
            _masked_select_backward_prefix_apply_tail_kernel[(1,)](
                grad,
                mask,
                out,
                grad_numel,
                tail_len,
                num_full,
                BLOCK_SIZE=block_size,
                HAS_GRAD=has_grad,
            )
        return out[:n_elements]

    # Parallel 3-phase pipeline (always used when validation is needed, so
    # the tagged total is available right after the scan).
    scan_block = min(
        _SCAN_MAX_BLOCK,
        max(_SCAN_MIN_BLOCK, triton.next_power_of_2(num_blocks)),
    )
    num_chunks = triton.cdiv(num_blocks, scan_block)
    index_dtype = torch.int32 if n_elements < 2**31 else torch.int64
    part_sums = torch.empty(num_chunks * scan_block, dtype=index_dtype, device=device)
    if num_full:
        _masked_select_backward_count_blocks_kernel[(num_full,)](
            mask,
            part_sums,
            n_elements,
            BLOCK_SIZE=block_size,
        )
    if tail_len:
        _masked_select_backward_count_tail_kernel[(1,)](
            mask[num_full * block_size :],
            part_sums,
            tail_len,
            num_full,
            BLOCK_SIZE=tail_block,
        )
    # The scan always runs: it converts the per-block counts into the
    # prefix that the apply kernels consume.  Only the tagged readback is
    # optional.
    tag = _next_tag()
    total_count = _count_scratch(device)
    _launch(
        _masked_select_backward_scan_kernel,
        (1,),
        (part_sums, total_count, num_blocks, tag),
        {"SCAN_BLOCK": scan_block, "NUM_CHUNKS": num_chunks},
    )
    count_state = _begin_count_read(total_count, tag) if need_validate else None
    if has_grad:
        if num_full:
            _masked_select_backward_apply_kernel[(num_full,)](
                grad,
                mask,
                part_sums,
                out,
                grad_numel,
                BLOCK_SIZE=block_size,
                HAS_GRAD=has_grad,
            )
        if tail_len:
            _masked_select_backward_apply_tail_kernel[(1,)](
                grad,
                mask[num_full * block_size :],
                part_sums,
                out[num_full * block_size :],
                grad_numel,
                tail_len,
                num_full,
                BLOCK_SIZE=tail_block,
                HAS_GRAD=has_grad,
            )
    if need_validate and grad_numel < _finish_count_read(count_state):
        raise RuntimeError(error_msg)
    return out[:n_elements]


def masked_select_backward(grad, input, mask):
    logger.debug("GEMS_KUNLUNXIN MASKED_SELECT_BACKWARD")

    if mask.dtype != torch.bool:
        raise RuntimeError(
            f"masked_scatter_ only supports boolean masks, but got {mask.dtype}"
        )
    if grad.dtype != input.dtype:
        raise RuntimeError(
            "masked_scatter: expected self and source to have same dtypes but got "
            f"{input.dtype} and {grad.dtype}"
        )
    if grad.device != input.device or mask.device != input.device:
        raise RuntimeError("grad, input, and mask must be on the same device")

    if input.shape == mask.shape and input.is_contiguous() and mask.is_contiguous():
        # No broadcast: skip the as_strided metadata construction entirely.
        input_expanded, mask_expanded = input, mask
    else:
        input_expanded, mask_expanded = _broadcast_views(input, mask)
    mask_contiguous = (
        mask_expanded if mask_expanded.is_contiguous() else contiguous(mask_expanded)
    )
    n_elements = input_expanded.numel()
    with _device_ctx(input.device):
        # ATen materialises a dense (contiguous) result; empty_like(
        # preserve_format) would produce an F-contiguous / mis-strided buffer.
        if n_elements == 0:
            return torch.zeros(
                input_expanded.shape, dtype=input.dtype, device=input.device
            )
        return _masked_select_backward_dispatch(
            grad, input, input_expanded, mask_contiguous, n_elements
        )


def _masked_select_backward_dispatch(
    grad, input, input_expanded, mask_contiguous, n_elements
):
    # NOTE: an empty ``grad`` is *not* an early return: ATen
    # (zeros().masked_scatter_()) raises "Number of elements of source <
    # number of ones in mask" whenever the mask selects anything, and only
    # returns zeros for an all-False mask.  The kernels below count the mask,
    # validate, and store zeros without touching the (empty) grad.

    if not grad.is_contiguous():
        grad = contiguous(grad)
    if input.is_complex():
        physical_grad = grad.conj() if grad.is_conj() else grad
        # ``.real`` / ``.imag`` are metadata views of the complex storage
        # (equivalent to indexing ``view_as_real``); no torch compute call is
        # involved and the actual gather runs in the Triton kernels above.
        real_dtype = physical_grad.real.dtype
        real = _masked_select_backward_flat(
            physical_grad.real, mask_contiguous, n_elements, real_dtype, grad.device
        )
        imag = _masked_select_backward_flat(
            neg(physical_grad.imag) if grad.is_conj() else physical_grad.imag,
            mask_contiguous,
            n_elements,
            real_dtype,
            grad.device,
            validate=False,
        )
        result = torch.zeros(
            input_expanded.shape, dtype=input.dtype, device=input.device
        )
        copy_(result.real, real.view(input_expanded.shape))
        copy_(result.imag, imag.view(input_expanded.shape))
        return result

    flat = _masked_select_backward_flat(
        grad, mask_contiguous, n_elements, input.dtype, input.device
    )
    if flat.shape == input_expanded.shape:
        return flat
    return flat.view(input_expanded.shape)
