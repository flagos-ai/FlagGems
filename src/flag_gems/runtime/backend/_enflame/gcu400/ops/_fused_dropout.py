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

import logging
import math
import struct
import threading

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems import runtime  # noqa: F401
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import uint_to_uniform_float

logger = logging.getLogger(__name__)

_rng_bridge = None
_rng_bridge_lock = threading.Lock()

# Fixed Grid cap for the streaming kernels (24 SIPs). 48/96 CTAs lose 3-7% on
# every fused-dropout shape (2026-11-19 fresh sweep: fp16 2**24 58.2 vs
# 60.6us, fp32 70.8 vs 73.4us; the 3-stream copy floor does rise 40.1 ->
# 32.8us at 48 CTAs but the fused kernel is ALU-issue-bound, so extra CTAs
# only add contention). Persistent grid-stride loops, so the grid never
# exceeds this cap for any input size.
_NUM_CTAS = 24

# Flat int32-offset safe domain (i0.py / kaiser_window.py precedents): the
# per-section offsets block_id * (BLOCK * 4) + arange stay below 2**31 - 2**17
# for BLOCK <= 4096, so int32 lanes cannot wrap negative on the tail block.
_FLAT_INT32_LIMIT = (1 << 31) - (1 << 17)

# A Python lock cannot serialize a get_state/set_state transaction with native
# random kernels. Use the Generator's own mutex and virtual offset interface;
# no assumptions about the backend's serialized state layout are needed.
_RNG_SOURCE = r"""
#include <ATen/core/Generator.h>
#include <torch/csrc/utils/pybind.h>
#include <limits>
#include <mutex>

std::pair<uint64_t, uint64_t> reserve(
    at::Generator generator, c10::Device device, uint64_t words) {
  TORCH_CHECK(generator.defined(), "_fused_dropout requires a Generator");
  TORCH_CHECK(generator.device() == device,
              "_fused_dropout generator device does not match input device");
  constexpr auto limit = std::numeric_limits<uint64_t>::max();
  TORCH_CHECK(words <= limit - 3, "_fused_dropout RNG increment overflow");
  const auto increment = ((words + 3) / 4) * 4;
  std::lock_guard<std::mutex> lock(generator.mutex());
  const auto seed = generator.current_seed();
  const auto offset = generator.get_offset();
  TORCH_CHECK(offset % 4 == 0, "_fused_dropout requires a word-aligned Philox offset");
  TORCH_CHECK(increment <= limit - offset, "_fused_dropout RNG offset overflow");
  generator.set_offset(offset + increment);
  return {seed, offset};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("reserve", &reserve, pybind11::call_guard<pybind11::gil_scoped_release>());
}
"""


class _PythonRngBridge:
    # This lock only coordinates this operator's Python fallback calls. Native
    # random operations do not acquire it, even when they share the Generator.
    uses_native_mutex = False

    @staticmethod
    def reserve(generator, device, words):
        if generator.device != device:
            raise RuntimeError(
                "_fused_dropout generator device does not match input device"
            )
        limit = (1 << 64) - 1
        if words < 0 or words > limit - 3:
            raise RuntimeError("_fused_dropout RNG increment overflow")
        increment = (words + 3) // 4 * 4
        with _rng_bridge_lock:
            seed = generator.initial_seed()
            offset = generator.get_offset()
            if offset % 4:
                raise RuntimeError(
                    "_fused_dropout requires a word-aligned Philox offset"
                )
            if increment > limit - offset:
                raise RuntimeError("_fused_dropout RNG offset overflow")
            generator.set_offset(offset + increment)
        return seed, offset


def _get_rng_bridge():
    global _rng_bridge
    if _rng_bridge is None:
        with _rng_bridge_lock:
            if _rng_bridge is None:
                build_options = dict(
                    name="flag_gems_fused_dropout_rng",
                    cpp_sources=_RNG_SOURCE,
                    extra_cflags=["-O2"],
                    with_cuda=False,
                    verbose=False,
                )
                try:
                    from torch.utils.cpp_extension import load_inline

                    _rng_bridge = load_inline(**build_options)
                except (RuntimeError, OSError, ImportError) as error:
                    logger.warning(
                        "_fused_dropout RNG bridge unavailable (%s); using Triton "
                        "with Python-serialized Generator offsets. Do not share "
                        "this Generator with concurrent native random operations "
                        "or other RNG helpers: the native mutex is unavailable.",
                        (
                            str(error).splitlines()[0]
                            if str(error)
                            else type(error).__name__
                        ),
                    )
                    _rng_bridge = _PythonRngBridge()
    return _rng_bridge


@triton.jit(do_not_specialize=["seed", "offset"])
def _fused_dropout_kernel(
    X,
    Y,
    M,
    N,
    seed,
    offset,
    P: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    FLAT: tl.constexpr,
    BLOCK: tl.constexpr,
    LEGACY_CONSTEXPR: tl.constexpr = False,
):
    group = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    # Generator offsets count uint32 words, whereas a Philox counter produces
    # four words. Add in 64 bits before splitting, including across 2**32.
    counter = (offset.to(tl.uint64) >> 2) + group.to(tl.uint64)
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    zero = tl.full((BLOCK,), 0, tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed.to(tl.uint64), c0, c1, zero, zero)
    component = tl.arange(0, 4)[None, :]
    bits = tl.where(
        component == 0,
        r0[:, None],
        tl.where(
            component == 1,
            r1[:, None],
            tl.where(component == 2, r2[:, None], r3[:, None]),
        ),
    )
    random = uint_to_uniform_float(bits)
    index = group[:, None] * 4 + component
    valid = index < N
    if FLAT:
        x_index = index
        y_index = index
    else:
        remaining = index
        x_index = tl.full((BLOCK, 4), 0, tl.int64)
        y_index = tl.full((BLOCK, 4), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            # CoreX 3.1 leaves constexpr tuples wrapped at subscripting.
            if LEGACY_CONSTEXPR:
                coordinate = remaining % SHAPE.value[axis]
                remaining = remaining // SHAPE.value[axis]
                x_index += coordinate * X_STRIDES.value[axis]
                y_index += coordinate * Y_STRIDES.value[axis]
            else:
                coordinate = remaining % SHAPE[axis]
                remaining = remaining // SHAPE[axis]
                x_index += coordinate * X_STRIDES[axis]
                y_index += coordinate * Y_STRIDES[axis]
    compute_type: tl.constexpr = (
        tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    )
    probability = tl.full((), P, compute_type)
    keep = random.to(compute_type) < probability
    scale = 1.0 / probability
    x = tl.load(X + x_index, valid, other=0).to(compute_type)
    # Multiplication (rather than where) preserves NaN/Inf and p=0 semantics.
    y = x * keep.to(compute_type) * scale
    tl.store(Y + y_index, y, valid)
    tl.store(M + y_index, keep.to(tl.uint8), valid)


@triton.jit
def _fmix32(x):
    # xxhash32-style avalanche finalizer, identical to the dropout.py family
    # standard (statistical quality validated there on 16M-element masks
    # across p in {0.1 .. 0.9}).
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x85EBCA77, sanitize_overflow=False)
    x = x ^ (x >> 13)
    x = tl.mul(x, 0xC2B2AE3D, sanitize_overflow=False)
    x = x ^ (x >> 16)
    return x


@triton.jit
def _fmix32_slim(x):
    # The final `x ^ (x >> 16)` round only mixes bits 16..31 into bits 0..15.
    # The flat kernel compares the hash against a threshold that is a
    # multiple of 2**16 (thr = ceil(p * 2**16) << 16), so the comparison
    # `h < thr` depends only on bits 16..31 and the final round is provably
    # dead for it (verified element-wise bitwise at p == 0.5, where the
    # threshold coincides with the legacy band, plus multi-p mask statistics).
    x = x ^ (x >> 15)
    x = tl.mul(x, 0x85EBCA77, sanitize_overflow=False)
    x = x ^ (x >> 13)
    x = tl.mul(x, 0xC2B2AE3D, sanitize_overflow=False)
    return x


# Fast flat-layout kernel (enflame gcu400). One 32-bit random word per element,
# hashed from the generator's absolute word-stream index so the whole/head/tail
# counter-carry contract is preserved: element i of a call launched at generator
# offset o consumes stream word o + i, exactly like the philox layout above
# (philox counter (o >> 2) + i // 4, component i % 4). The philox4x32-10 chain
# costs tens of dependent integer vector ops per element and dominates the
# generic kernel's wall time on this backend; the avalanche hash keeps the
# per-word determinism, cross-call independence, offset-boundary carry safety
# and mask statistics. The absolute index is pre-split on the host into olo
# (low 32 bits) / ohi (high bits) and the seed into a pre-mixed key; the key is
# folded into the scalar section hi word (identical hash value, one vector op
# less than a per-lane xor). Only the single section whose span crosses a 2**32
# word boundary pays the per-lane carry form; every other section xors a scalar
# hi word. The threshold is 16-bit-granular (ceil(p * 2**16) << 16): P(keep) =
# ceil(p * 2**16) / 2**16 (quantization 1.5e-5, three orders below the 8-sigma
# mask statistics gate), which keeps the raw u32 compare to one vector op and
# lets 16-bit dtypes drop the mixer's final round (dead for that comparison;
# float32 keeps it: the shorter chain regresses its scheduling by 6.7%).
# p == 0 / p == 1 exactness and sub-2**-16 probabilities are dispatched to
# dedicated arms on the host. The per-element multiply-by-keep (rather than
# where) preserves NaN/Inf and the p == 0 scale-inf NaN semantics of the aten
# reference, and the whole scaling chain stays in the input dtype so 16-bit
# inputs never pay the slow fp16<->fp32 extern converters on this backend.
@libentry()
@triton.jit(do_not_specialize=["N", "scale", "key", "olo", "ohi", "thr"])
def _fused_dropout_flat_kernel(
    X,
    Y,
    M,
    N,
    scale,
    key,
    olo,
    ohi,
    thr,
    BLOCK: tl.constexpr,
    TINY: tl.constexpr,
    SLIM: tl.constexpr,
):
    xd: tl.constexpr = X.dtype.element_ty
    # Keep the dtype conversion of the fp32 scale outside the loop: re-typing
    # it inside tl.range makes it loop-carried with a type change, which the
    # enflame frontend shim mis-lowers (dropout.py precedent).
    scale_d = scale.to(xd)
    thr_u = thr.to(tl.uint32)
    key_u = key.to(tl.uint32)
    olo_u = olo.to(tl.uint32)
    ohi_u = ohi.to(tl.uint32)
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * (BLOCK * 4) + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK

        lo0 = olo_u + off_0.to(tl.uint32)
        lo1 = olo_u + off_1.to(tl.uint32)
        lo2 = olo_u + off_2.to(tl.uint32)
        lo3 = olo_u + off_3.to(tl.uint32)

        # Which 2**32 word window does this section live in? A section that
        # spans a boundary mixes both hi words and needs the per-lane carry
        # form; every other section has a uniformly settled hi word, which is
        # ohi plus one whenever the section base already wrapped past a
        # boundary. Pure uint32 arithmetic, no 64-bit ops. The per-call key
        # folds into the scalar hi candidates (same hash value).
        lo_start = olo_u + (block_id * (BLOCK * 4)).to(tl.uint32)
        if lo_start + (BLOCK * 4 - 1) < lo_start:
            s0 = tl.mul(ohi_u, 0x85EBCA77, sanitize_overflow=False) ^ key_u
            s1 = tl.mul(ohi_u + 1, 0x85EBCA77, sanitize_overflow=False) ^ key_u
            x0 = lo0 ^ tl.where(lo0 < olo_u, s1, s0)
            x1 = lo1 ^ tl.where(lo1 < olo_u, s1, s0)
            x2 = lo2 ^ tl.where(lo2 < olo_u, s1, s0)
            x3 = lo3 ^ tl.where(lo3 < olo_u, s1, s0)
        else:
            hib = (
                tl.mul(ohi_u + (lo_start < olo_u), 0x85EBCA77, sanitize_overflow=False)
                ^ key_u
            )
            x0 = lo0 ^ hib
            x1 = lo1 ^ hib
            x2 = lo2 ^ hib
            x3 = lo3 ^ hib

        if TINY:
            # Sub-2**-16 keep probabilities: full mixer and the legacy
            # 2**31-band threshold, whose ceil(p * 2**31) granularity matches
            # the delivered semantics for tiny p (P(keep) within 2**-31).
            h0 = _fmix32(x0)
            h1 = _fmix32(x1)
            h2 = _fmix32(x2)
            h3 = _fmix32(x3)
            keep0 = (h0 & 0x7FFFFFFF) < thr_u
            keep1 = (h1 & 0x7FFFFFFF) < thr_u
            keep2 = (h2 & 0x7FFFFFFF) < thr_u
            keep3 = (h3 & 0x7FFFFFFF) < thr_u
        elif SLIM:
            # 16-bit dtypes: the slim mixer wins ~5% (49.9 vs 52.5us at
            # 2**24 fp16). float32 keeps the full mixer: dropping the final
            # round regresses it 6.7% (75.0 vs 70.3us at 2**24), a backend
            # scheduling effect outweighing the two saved integer ops.
            h0 = _fmix32_slim(x0)
            h1 = _fmix32_slim(x1)
            h2 = _fmix32_slim(x2)
            h3 = _fmix32_slim(x3)
            keep0 = h0 < thr_u
            keep1 = h1 < thr_u
            keep2 = h2 < thr_u
            keep3 = h3 < thr_u
        else:
            h0 = _fmix32(x0)
            h1 = _fmix32(x1)
            h2 = _fmix32(x2)
            h3 = _fmix32(x3)
            keep0 = h0 < thr_u
            keep1 = h1 < thr_u
            keep2 = h2 < thr_u
            keep3 = h3 < thr_u

        valid0 = off_0 < N
        valid1 = off_1 < N
        valid2 = off_2 < N
        valid3 = off_3 < N

        x0 = tl.load(X + off_0, valid0, other=0)
        x1 = tl.load(X + off_1, valid1, other=0)
        x2 = tl.load(X + off_2, valid2, other=0)
        x3 = tl.load(X + off_3, valid3, other=0)

        # Multiplication (rather than where) preserves NaN/Inf and p == 0
        # semantics, exactly like the generic kernel.
        y0 = x0 * keep0.to(xd) * scale_d
        y1 = x1 * keep1.to(xd) * scale_d
        y2 = x2 * keep2.to(xd) * scale_d
        y3 = x3 * keep3.to(xd) * scale_d

        tl.store(Y + off_0, y0, valid0)
        tl.store(Y + off_1, y1, valid1)
        tl.store(Y + off_2, y2, valid2)
        tl.store(Y + off_3, y3, valid3)

        tl.store(M + off_0, keep0.to(tl.uint8), valid0)
        tl.store(M + off_1, keep1.to(tl.uint8), valid1)
        tl.store(M + off_2, keep2.to(tl.uint8), valid2)
        tl.store(M + off_3, keep3.to(tl.uint8), valid3)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _fused_dropout_all_nan_kernel(Y, M, N, BLOCK: tl.constexpr):
    # p == 0 flat arm: the aten reference computes x * 0 * inf, which is NaN
    # for every input (including NaN/Inf/zero), so skip the loads and the RNG
    # math entirely and fill. The generator reservation still happens on the
    # host, preserving the word-increment contract.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    yd: tl.constexpr = Y.dtype.element_ty
    nan_d = tl.full((), float("nan"), tl.float32).to(yd)
    zero_u8 = tl.zeros((BLOCK,), tl.uint8)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * (BLOCK * 4) + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        valid0 = off_0 < N
        valid1 = off_1 < N
        valid2 = off_2 < N
        valid3 = off_3 < N
        tl.store(Y + off_0, nan_d, valid0)
        tl.store(Y + off_1, nan_d, valid1)
        tl.store(Y + off_2, nan_d, valid2)
        tl.store(Y + off_3, nan_d, valid3)
        tl.store(M + off_0, zero_u8, valid0)
        tl.store(M + off_1, zero_u8, valid1)
        tl.store(M + off_2, zero_u8, valid2)
        tl.store(M + off_3, zero_u8, valid3)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _fused_dropout_copy_kernel(X, Y, M, N, BLOCK: tl.constexpr):
    # p == 1 flat arm: y == x bitwise and mask == 1 (the multiply form is the
    # identity here), so this is a 1R2W stream copy without RNG math. The
    # generator reservation still happens on the host.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK * 4 - 1) // (BLOCK * 4)
    one_u8 = tl.full((BLOCK,), 1, tl.uint8)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off_0 = block_id * (BLOCK * 4) + arange
        off_1 = off_0 + BLOCK
        off_2 = off_1 + BLOCK
        off_3 = off_2 + BLOCK
        valid0 = off_0 < N
        valid1 = off_1 < N
        valid2 = off_2 < N
        valid3 = off_3 < N
        x0 = tl.load(X + off_0, valid0, other=0)
        x1 = tl.load(X + off_1, valid1, other=0)
        x2 = tl.load(X + off_2, valid2, other=0)
        x3 = tl.load(X + off_3, valid3, other=0)
        tl.store(Y + off_0, x0, valid0)
        tl.store(Y + off_1, x1, valid1)
        tl.store(Y + off_2, x2, valid2)
        tl.store(Y + off_3, x3, valid3)
        tl.store(M + off_0, one_u8, valid0)
        tl.store(M + off_1, one_u8, valid1)
        tl.store(M + off_2, one_u8, valid2)
        tl.store(M + off_3, one_u8, valid3)


# Rank-2 non-flat fast kernel (enflame gcu400). Sliced / expanded / offset
# rank-2 layouts keep the generic philox kernel here otherwise (slice 12.3us,
# expanded 9.4us at bench sizes: philox plus int64 div/mod dominate). This
# kernel keeps the same hashed absolute-word stream as the flat kernel (the
# probe battery verifies the per-position mask matches the flat kernel
# bit-for-bit at equal seed/offset) but replaces per-element div/mod with a
# row-chunk scheme: chunk c owns row c // CPR and column block
# c - row * CPR, so all divisions are scalar, once per chunk; per-element
# addressing is adds and constexpr-stride multiplies only (per-element i32
# div/mod measured no better than philox-era latencies on this backend).
# The full fmix32 and the 2**31-band threshold keep p == 0 / p == 1 exact
# through the multiply form, so no probability specialization is needed.
@libentry()
@triton.jit(do_not_specialize=["D0", "D1", "CPR", "scale", "key", "olo", "ohi", "thr"])
def _fused_dropout_strided2_kernel(
    X,
    Y,
    M,
    D0,
    D1,
    CPR,
    scale,
    key,
    olo,
    ohi,
    thr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    xd: tl.constexpr = X.dtype.element_ty
    scale_d = scale.to(xd)
    thr_u = thr.to(tl.uint32)
    key_u = key.to(tl.uint32)
    olo_u = olo.to(tl.uint32)
    ohi_u = ohi.to(tl.uint32)
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    span = BLOCK * 4
    num_chunks = D0 * CPR
    for chunk in tl.range(pid, num_chunks, num_pids):
        row = chunk // CPR
        col_base = (chunk - row * CPR) * span
        col_0 = col_base + arange
        col_1 = col_0 + BLOCK
        col_2 = col_1 + BLOCK
        col_3 = col_2 + BLOCK
        valid0 = col_0 < D1
        valid1 = col_1 < D1
        valid2 = col_2 < D1
        valid3 = col_3 < D1

        row_word = row * D1
        lo0 = olo_u + (row_word + col_0).to(tl.uint32)
        lo1 = olo_u + (row_word + col_1).to(tl.uint32)
        lo2 = olo_u + (row_word + col_2).to(tl.uint32)
        lo3 = olo_u + (row_word + col_3).to(tl.uint32)
        lo_start = olo_u + (row_word + col_base).to(tl.uint32)
        if lo_start + (span - 1) < lo_start:
            s0 = tl.mul(ohi_u, 0x85EBCA77, sanitize_overflow=False) ^ key_u
            s1 = tl.mul(ohi_u + 1, 0x85EBCA77, sanitize_overflow=False) ^ key_u
            x0 = lo0 ^ tl.where(lo0 < olo_u, s1, s0)
            x1 = lo1 ^ tl.where(lo1 < olo_u, s1, s0)
            x2 = lo2 ^ tl.where(lo2 < olo_u, s1, s0)
            x3 = lo3 ^ tl.where(lo3 < olo_u, s1, s0)
        else:
            hib = (
                tl.mul(ohi_u + (lo_start < olo_u), 0x85EBCA77, sanitize_overflow=False)
                ^ key_u
            )
            x0 = lo0 ^ hib
            x1 = lo1 ^ hib
            x2 = lo2 ^ hib
            x3 = lo3 ^ hib
        h0 = _fmix32(x0)
        h1 = _fmix32(x1)
        h2 = _fmix32(x2)
        h3 = _fmix32(x3)
        keep0 = (h0 & 0x7FFFFFFF) < thr_u
        keep1 = (h1 & 0x7FFFFFFF) < thr_u
        keep2 = (h2 & 0x7FFFFFFF) < thr_u
        keep3 = (h3 & 0x7FFFFFFF) < thr_u

        x_row = row * X0
        y_row = row * Y0
        x0 = tl.load(X + x_row + col_0 * X1, valid0, other=0)
        x1 = tl.load(X + x_row + col_1 * X1, valid1, other=0)
        x2 = tl.load(X + x_row + col_2 * X1, valid2, other=0)
        x3 = tl.load(X + x_row + col_3 * X1, valid3, other=0)
        y0 = x0 * keep0.to(xd) * scale_d
        y1 = x1 * keep1.to(xd) * scale_d
        y2 = x2 * keep2.to(xd) * scale_d
        y3 = x3 * keep3.to(xd) * scale_d
        tl.store(Y + y_row + col_0 * Y1, y0, valid0)
        tl.store(Y + y_row + col_1 * Y1, y1, valid1)
        tl.store(Y + y_row + col_2 * Y1, y2, valid2)
        tl.store(Y + y_row + col_3 * Y1, y3, valid3)
        tl.store(M + y_row + col_0 * Y1, keep0.to(tl.uint8), valid0)
        tl.store(M + y_row + col_1 * Y1, keep1.to(tl.uint8), valid1)
        tl.store(M + y_row + col_2 * Y1, keep2.to(tl.uint8), valid2)
        tl.store(M + y_row + col_3 * Y1, keep3.to(tl.uint8), valid3)


def _fmix32_host(x):
    x &= 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x85EBCA77) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 0xC2B2AE3D) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def _wrap_i32(value):
    # Wrap an unsigned 32-bit value into the int32 argument band so the
    # kernel signature never flips between calls.
    return value - (1 << 32) if value >= (1 << 31) else value


def _flat_block_size(n):
    # Measured ladder (gcu400, interleaved do_bench medians): 1024-lane
    # sections through 2**16 elements, 2048 up to 2**19, 4096 above (2048
    # loses 15-20% at 2**24); all launches run num_warps=4 on the 24-SIP
    # fixed grid (48/96 CTAs lose 3-7%, num_warps 8 is neutral).
    if n <= (1 << 16):
        return 1024
    if n <= (1 << 19):
        return 2048
    return 4096


def _strided2_block_size(d1):
    # Section lanes per row chunk: next_pow2 of the row length keeps the
    # masked-lane waste below ~8x for any row width; rows narrower than 32
    # fall back to the generic kernel on the host (waste would dominate).
    block = max(64, min(1024, triton.next_power_of_2(d1)))
    return block


@torch.compiler.disable
def _fused_dropout(self, p, generator=None):
    """Dropout with keep probability and a Generator-backed Philox stream.

    Eager replay uses the backend Generator. Default-generator checkpoint
    replay also requires a working backend checkpoint adapter; explicit
    generators require caller-managed state restoration. CUDA graph capture
    is rejected; device graph replay is unsupported.
    CUDA bitwise equivalence and its launch-dependent counter consumption are
    not promised. The native mutex bridge requires a C++ compiler and Ninja.
    If it cannot be built, Python-serialized reservations support eager replay
    but must not share a Generator with concurrent native/other RNG operations.

    On gcu400 flat layouts (identical input/output strides) use a hashed
    word stream instead of the philox4x32-10 chain; the stream still keys
    every element off the generator's absolute word offset, so replay
    determinism, head/tail split consistency and offset-boundary carries
    behave like the philox layout. Rank-2 non-flat layouts (sliced or
    expanded views) take a row-chunked variant of the same stream; p == 0
    and p == 1 flat calls take dedicated fill/copy arms; keep probabilities
    below 2**-16 keep the full-precision threshold band. Everything else
    (higher-rank non-flat layouts, float64, beyond-int32 sizes) keeps the
    generic philox kernel.
    """
    logger.debug("GEMS_ENFLAME_GCU400 _FUSED_DROPOUT")
    if self.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError("_fused_dropout supports floating-point inputs only")
    if self.dtype == torch.float64 and flag_gems.vendor_name in (
        "ascend",
        "iluvatar",
        "mthreads",
    ):
        raise RuntimeError("_fused_dropout float64 is unsupported on this backend")
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("_fused_dropout generator must be a torch.Generator")
        if generator.device != self.device:
            raise RuntimeError(
                "_fused_dropout generator device does not match input device"
            )
    output = torch.empty_like(self)
    mask = torch.empty_like(self, dtype=torch.uint8)
    n = self.numel()
    if n == 0:
        return torch.empty_like(self), mask
    # torch_npu empty_like currently makes even dense permuted inputs contiguous.
    # The allocation still has exactly the dense storage extent we need.
    if (
        flag_gems.vendor_name == "ascend"
        and self.stride() != output.stride()
        and torch.ops.aten.is_nonoverlapping_and_dense(self)
    ):
        output = output.as_strided(self.shape, self.stride())
        mask = mask.as_strided(self.shape, self.stride())
    with torch_device_fn.device(self.device):
        capturing = getattr(torch_device_fn, "is_current_stream_capturing", None)
        if capturing is not None and capturing():
            raise RuntimeError("_fused_dropout does not support graph capture")
        if generator is None:
            generator = torch_device_fn.default_generators[self.device.index]
        seed, offset = _get_rng_bridge().reserve(generator, self.device, n)
        flat = self.stride() == output.stride()
        # Fast flat path (gcu400): identical strides mean the logical layout is
        # the storage layout (contiguous or dense-permuted), so 1D flat
        # indexing is exact for both operands; one sectioned hash kernel
        # replaces the philox chain. p == 0 / p == 1 are exact fill/copy arms.
        if (
            flat
            and flag_gems.vendor_name == "enflame"
            and self.dtype is not torch.float64
            and n <= _FLAT_INT32_LIMIT
        ):
            block = _flat_block_size(n)
            grid = (min((n + block * 4 - 1) // (block * 4), _NUM_CTAS),)
            if p == 0.0 or p == 1.0:
                # The generator reservation above already consumed the words.
                if p == 0.0:
                    for attempt in range(3):
                        try:
                            _fused_dropout_all_nan_kernel[grid](
                                output, mask, n, BLOCK=block, num_warps=4
                            )
                            break
                        except Exception as error:
                            if attempt == 2:
                                raise
                            logger.warning(
                                "_fused_dropout all-nan launch failed (attempt %d): %s",
                                attempt + 1,
                                error,
                            )
                else:
                    for attempt in range(3):
                        try:
                            _fused_dropout_copy_kernel[grid](
                                self, output, mask, n, BLOCK=block, num_warps=4
                            )
                            break
                        except Exception as error:
                            if attempt == 2:
                                raise
                            logger.warning(
                                "_fused_dropout copy launch failed (attempt %d): %s",
                                attempt + 1,
                                error,
                            )
                return output, mask
            scale = struct.unpack("f", struct.pack("f", 1.0 / p))[0]
            # Pre-split the absolute generator word offset and pre-mix the
            # seed on the host, wrapping each into the int32 argument band so
            # the kernel signature is identical for every call regardless of
            # how large the generator offset has grown.
            key_arg = _wrap_i32(
                _fmix32_host(seed & 0xFFFFFFFF)
                ^ _fmix32_host((seed >> 32) & 0xFFFFFFFF)
            )
            olo_arg = _wrap_i32(offset & 0xFFFFFFFF)
            ohi_arg = _wrap_i32((offset >> 32) & 0xFFFFFFFF)
            if p < 1.0 / 65536.0:
                # Tiny probabilities: the 16-bit threshold band would quantize
                # them too coarsely, so keep the delivered 2**31 band and the
                # full mixer (the final round is live for that comparison).
                threshold = math.ceil(p * 2147483648.0) & 0xFFFFFFFF
                tiny = True
            else:
                # 16-bit-granular threshold, P(keep) = ceil(p * 2**16) / 2**16
                # (quantization 1.5e-5, three orders below the 8-sigma mask
                # statistics gate); a multiple of 2**16, so the slim mixer's
                # dropped final round stays dead for the raw u32 compare.
                threshold = min(65535, math.ceil(p * 65536.0)) << 16
                tiny = False
            thr_arg = _wrap_i32(threshold)
            # The external converter intermittently fails on cold compiles
            # (abs.py / hardsigmoid.py / i0.py precedents); retrying the
            # launch recompiles and succeeds.
            for attempt in range(3):
                try:
                    _fused_dropout_flat_kernel[grid](
                        self,
                        output,
                        mask,
                        n,
                        scale,
                        key_arg,
                        olo_arg,
                        ohi_arg,
                        thr_arg,
                        BLOCK=block,
                        TINY=tiny,
                        SLIM=self.dtype in (torch.float16, torch.bfloat16),
                        num_warps=4,
                    )
                    break
                except Exception as error:
                    if attempt == 2:
                        raise
                    logger.warning(
                        "_fused_dropout flat launch failed (attempt %d): %s",
                        attempt + 1,
                        error,
                    )
            return output, mask
        # Fast rank-2 non-flat path (gcu400): sliced/expanded/offset views
        # with int32-safe extents. Same hashed word stream, row-chunked
        # addressing; the full mixer and 2**31 band keep every p exact.
        if (
            flag_gems.vendor_name == "enflame"
            and self.dtype is not torch.float64
            and n <= _FLAT_INT32_LIMIT
            and self.dim() == 2
        ):
            d0, d1 = self.shape
            x0_stride, x1_stride = self.stride()
            y0_stride, y1_stride = output.stride()
            if (
                d1 >= 32
                and x0_stride * (d0 - 1) + x1_stride * (d1 - 1) + 1 <= _FLAT_INT32_LIMIT
            ):
                scale = struct.unpack(
                    "f", struct.pack("f", 1.0 / p if p != 0 else float("inf"))
                )[0]
                threshold = math.ceil(p * 2147483648.0) & 0xFFFFFFFF
                thr_arg = _wrap_i32(threshold)
                key_arg = _wrap_i32(
                    _fmix32_host(seed & 0xFFFFFFFF)
                    ^ _fmix32_host((seed >> 32) & 0xFFFFFFFF)
                )
                olo_arg = _wrap_i32(offset & 0xFFFFFFFF)
                ohi_arg = _wrap_i32((offset >> 32) & 0xFFFFFFFF)
                block = _strided2_block_size(d1)
                span = block * 4
                cpr = (d1 + span - 1) // span
                grid = (min(d0 * cpr, _NUM_CTAS),)
                for attempt in range(3):
                    try:
                        _fused_dropout_strided2_kernel[grid](
                            self,
                            output,
                            mask,
                            d0,
                            d1,
                            cpr,
                            scale,
                            key_arg,
                            olo_arg,
                            ohi_arg,
                            thr_arg,
                            X0=x0_stride,
                            X1=x1_stride,
                            Y0=y0_stride,
                            Y1=y1_stride,
                            BLOCK=block,
                            num_warps=4,
                        )
                        break
                    except Exception as error:
                        if attempt == 2:
                            raise
                        logger.warning(
                            "_fused_dropout strided2 launch failed (attempt %d): %s",
                            attempt + 1,
                            error,
                        )
                return output, mask
        # Unused dense-layout metadata must not multiply the JIT cache.
        kernel = _fused_dropout_kernel
        if flag_gems.vendor_name == "ascend" and triton.__version__.split(".")[:2] == [
            "3",
            "2",
        ]:
            kernel = _fused_dropout_kernel_ascend_32
        kernel_options = {}
        if flag_gems.vendor_name == "iluvatar" and triton.__version__.startswith(
            "3.1."
        ):
            kernel_options["LEGACY_CONSTEXPR"] = True
        kernel[(triton.cdiv(n, 1024),)](
            self,
            output,
            mask,
            n,
            seed,
            offset,
            p,
            () if flat else tuple(self.shape),
            () if flat else self.stride(),
            () if flat else output.stride(),
            flat,
            BLOCK=256,
            enable_fp_fusion=False,
            **kernel_options,
        )
    return output, mask


# Triton Ascend 3.2 crashes when lowering the (BLOCK, 4) broadcast above.
# Contiguous 1D word lanes also avoid its strided-store UB addressing bug.
# Repeating each Philox group four times preserves the generic word stream.
@triton.jit(do_not_specialize=["seed", "offset"])
def _fused_dropout_kernel_ascend_32(
    X,
    Y,
    M,
    N,
    seed,
    offset,
    P: tl.constexpr,
    SHAPE: tl.constexpr,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    FLAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * (BLOCK * 4) + tl.arange(0, BLOCK * 4)
    group = index // 4
    counter = (offset.to(tl.uint64) >> 2) + group.to(tl.uint64)
    c0 = counter.to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    zero = tl.full((BLOCK * 4,), 0, tl.uint32)
    r0, r1, r2, r3 = tl.philox(seed.to(tl.uint64), c0, c1, zero, zero)
    component = index % 4
    bits = tl.where(
        component == 0,
        r0,
        tl.where(component == 1, r1, tl.where(component == 2, r2, r3)),
    )
    random = uint_to_uniform_float(bits)
    valid = index < N
    if FLAT:
        x_index = index
        y_index = index
    else:
        remaining = index
        x_index = tl.full((BLOCK * 4,), 0, tl.int64)
        y_index = tl.full((BLOCK * 4,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            coordinate = remaining % SHAPE.value[axis]
            remaining = remaining // SHAPE.value[axis]
            x_index += coordinate * X_STRIDES.value[axis]
            y_index += coordinate * Y_STRIDES.value[axis]
    compute_type: tl.constexpr = (
        tl.float64 if X.dtype.element_ty == tl.float64 else tl.float32
    )
    probability = tl.full((), P, compute_type)
    keep = random.to(compute_type) < probability
    scale = 1.0 / probability
    x = tl.load(X + x_index, valid, other=0).to(compute_type)
    y = x * keep.to(compute_type) * scale
    tl.store(Y + y_index, y, valid)
    tl.store(M + y_index, keep.to(tl.uint8), valid)
