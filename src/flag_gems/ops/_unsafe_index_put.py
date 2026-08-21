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

"""Pure-Triton ``_unsafe_index_put``.

Port of the C++ wrapper logic from ``new_feature/add_nv_op_unsafe_index_put``
without any C++ involvement:

- Index preprocessing (bool/byte mask expansion via :func:`torch.nonzero`,
  device fixup, broadcast shape/strides) happens in Python.
- Both paths use a 2D-grid Triton kernel. Grid dim 0 walks broadcast
  index-space positions, grid dim 1 walks suffix positions, so the kernel
  never performs an expensive division by ``suffix_numel``.
- ``accumulate=False``: plain store. With duplicate indices the winner is
  unspecified, matching PyTorch's contract.
- ``accumulate=True``: ``tl.atomic_add``. Dtypes without a native Triton
  atomic add (fp16/bf16, int8/int16/uint8/bool) accumulate in a widened
  scratch buffer (fp32 / int32) with a single cast on writeback, mirroring
  PyTorch's opmath_t semantics. The summation order of duplicate indices is
  arbitrary GPU scheduling (same tolerance class as PyTorch's own CPU-vs-CUDA
  divergence); integer accumulation is exact.

Performance notes for the pure-Python wrapper:

- All host-side stride/divisor scalars are packed into one small cached int64
  ``meta`` tensor per call (cached keyed by the exact shapes/strides), which
  keeps the kernel signatures short and sidesteps Triton's per-argument
  specialization cost in the Python launcher.
- The output clone is ``empty_like`` + a Triton copy kernel (vectorized flat
  copy in the common contiguous case), avoiding the patched ``aten::clone``
  dispatch under ``use_gems``.
- Kernel launches go through a cached ``CompiledKernel`` (``_fast_launch``),
  skipping triton's per-call binder/specialization, which costs ~15us of
  Python per launch and dominates small shapes. The cache key captures
  everything the compiled kernel depends on (constexpr values, argument
  dtypes, each user tensor's 16-byte alignment, num_warps); on signature
  drift in future triton versions the normal launch path is used instead.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

try:
    from triton import knobs
except ImportError:
    # old Triton：create virtual knobs object，hook set None
    class _Knobs:
        class runtime:
            launch_enter_hook = None
            launch_exit_hook = None

    knobs = _Knobs()
from triton.runtime import driver

logger = logging.getLogger(__name__)

_MAX_NDIM = 6

# Cached CompiledKernel direct launches; see the module docstring.
_FAST_CACHE = {}


def _aligned(t):
    # torch's caching allocator returns 512B-aligned blocks; only views can
    # be misaligned, so this check is only needed for user-provided tensors.
    return t.data_ptr() & 15 == 0


def _fast_launch(jit_fn, grid, args, key, num_warps=4):
    """Launch a cached CompiledKernel, skipping triton's per-call binder.

    ``key`` must capture everything the compiled kernel depends on: constexpr
    values, argument dtypes, each user tensor's 16-byte alignment and
    num_warps. Callers build it cheaply from values they already hold.
    """
    cache_key = (jit_fn, key)
    kern = _FAST_CACHE.get(cache_key)
    if kern is None:
        kern = jit_fn[grid](*args, num_warps=num_warps)
        if hasattr(kern, "result"):
            kern = kern.result()
        _FAST_CACHE[cache_key] = kern
        return
    try:
        # Direct launch on the current stream, skipping the per-call binder
        # and the launch-metadata LazyDict. The metadata is None-safe because
        # the cache key already captures 16-byte alignments and our kernels
        # declare no launch_metadata function.
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        kern.run(
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            grid[2] if len(grid) > 2 else 1,
            stream,
            kern.function,
            kern.packed_metadata,
            None,
            knobs.runtime.launch_enter_hook,
            knobs.runtime.launch_exit_hook,
            *args,
        )
    except (TypeError, AttributeError):
        # triton internals changed; the normal path is always correct
        jit_fn[grid](*args, num_warps=num_warps)


# Meta layout (offsets into the meta tensor). The main and scratch kernels
# share this single layout (the scratch kernel loads only the fields it needs):
#   [0:6]    idx_div
#   [6:42]   ts (row-major tensor strides of each index in idx space)
#   [42:48]  val_adv
#   [48:54]  self_adv_stride
#   [54:60]  self_adv_size
#   [60:66]  suf_div
#   [66:72]  self_suf_stride
#   [72:78]  val_suf_stride


@triton.jit
def _unsafe_index_put_kernel(
    out_ptr,
    values_ptr,
    scratch_ptr,
    idx0_ptr,
    idx1_ptr,
    idx2_ptr,
    idx3_ptr,
    idx4_ptr,
    idx5_ptr,
    meta_ptr,
    idx_numel,
    suffix_numel,
    M: tl.constexpr,
    IDX_NDIM: tl.constexpr,
    SUF_NDIM: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    USE_SCRATCH: tl.constexpr,
    BLOCK_IDX: tl.constexpr,
    BLOCK_SUF: tl.constexpr,
):
    # 2D grid: program_id(0) -> idx position, program_id(1) -> suffix
    # position. Each block handles BLOCK_IDX index positions x BLOCK_SUF
    # suffix positions.
    idx_div0 = tl.load(meta_ptr + 0)
    idx_div1 = tl.load(meta_ptr + 1)
    idx_div2 = tl.load(meta_ptr + 2)
    idx_div3 = tl.load(meta_ptr + 3)
    idx_div4 = tl.load(meta_ptr + 4)
    idx_div5 = tl.load(meta_ptr + 5)
    ts_0_0 = tl.load(meta_ptr + 6)
    ts_0_1 = tl.load(meta_ptr + 7)
    ts_0_2 = tl.load(meta_ptr + 8)
    ts_0_3 = tl.load(meta_ptr + 9)
    ts_0_4 = tl.load(meta_ptr + 10)
    ts_0_5 = tl.load(meta_ptr + 11)
    ts_1_0 = tl.load(meta_ptr + 12)
    ts_1_1 = tl.load(meta_ptr + 13)
    ts_1_2 = tl.load(meta_ptr + 14)
    ts_1_3 = tl.load(meta_ptr + 15)
    ts_1_4 = tl.load(meta_ptr + 16)
    ts_1_5 = tl.load(meta_ptr + 17)
    ts_2_0 = tl.load(meta_ptr + 18)
    ts_2_1 = tl.load(meta_ptr + 19)
    ts_2_2 = tl.load(meta_ptr + 20)
    ts_2_3 = tl.load(meta_ptr + 21)
    ts_2_4 = tl.load(meta_ptr + 22)
    ts_2_5 = tl.load(meta_ptr + 23)
    ts_3_0 = tl.load(meta_ptr + 24)
    ts_3_1 = tl.load(meta_ptr + 25)
    ts_3_2 = tl.load(meta_ptr + 26)
    ts_3_3 = tl.load(meta_ptr + 27)
    ts_3_4 = tl.load(meta_ptr + 28)
    ts_3_5 = tl.load(meta_ptr + 29)
    ts_4_0 = tl.load(meta_ptr + 30)
    ts_4_1 = tl.load(meta_ptr + 31)
    ts_4_2 = tl.load(meta_ptr + 32)
    ts_4_3 = tl.load(meta_ptr + 33)
    ts_4_4 = tl.load(meta_ptr + 34)
    ts_4_5 = tl.load(meta_ptr + 35)
    ts_5_0 = tl.load(meta_ptr + 36)
    ts_5_1 = tl.load(meta_ptr + 37)
    ts_5_2 = tl.load(meta_ptr + 38)
    ts_5_3 = tl.load(meta_ptr + 39)
    ts_5_4 = tl.load(meta_ptr + 40)
    ts_5_5 = tl.load(meta_ptr + 41)
    val_adv0 = tl.load(meta_ptr + 42)
    val_adv1 = tl.load(meta_ptr + 43)
    val_adv2 = tl.load(meta_ptr + 44)
    val_adv3 = tl.load(meta_ptr + 45)
    val_adv4 = tl.load(meta_ptr + 46)
    val_adv5 = tl.load(meta_ptr + 47)
    self_adv_stride0 = tl.load(meta_ptr + 48)
    self_adv_stride1 = tl.load(meta_ptr + 49)
    self_adv_stride2 = tl.load(meta_ptr + 50)
    self_adv_stride3 = tl.load(meta_ptr + 51)
    self_adv_stride4 = tl.load(meta_ptr + 52)
    self_adv_stride5 = tl.load(meta_ptr + 53)
    self_adv_size0 = tl.load(meta_ptr + 54)
    self_adv_size1 = tl.load(meta_ptr + 55)
    self_adv_size2 = tl.load(meta_ptr + 56)
    self_adv_size3 = tl.load(meta_ptr + 57)
    self_adv_size4 = tl.load(meta_ptr + 58)
    self_adv_size5 = tl.load(meta_ptr + 59)
    suf_div0 = tl.load(meta_ptr + 60)
    suf_div1 = tl.load(meta_ptr + 61)
    suf_div2 = tl.load(meta_ptr + 62)
    suf_div3 = tl.load(meta_ptr + 63)
    suf_div4 = tl.load(meta_ptr + 64)
    suf_div5 = tl.load(meta_ptr + 65)
    self_suf_stride0 = tl.load(meta_ptr + 66)
    self_suf_stride1 = tl.load(meta_ptr + 67)
    self_suf_stride2 = tl.load(meta_ptr + 68)
    self_suf_stride3 = tl.load(meta_ptr + 69)
    self_suf_stride4 = tl.load(meta_ptr + 70)
    self_suf_stride5 = tl.load(meta_ptr + 71)
    val_suf_stride0 = tl.load(meta_ptr + 72)
    val_suf_stride1 = tl.load(meta_ptr + 73)
    val_suf_stride2 = tl.load(meta_ptr + 74)
    val_suf_stride3 = tl.load(meta_ptr + 75)
    val_suf_stride4 = tl.load(meta_ptr + 76)
    val_suf_stride5 = tl.load(meta_ptr + 77)

    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    idx_off = pid0 * BLOCK_IDX + tl.arange(0, BLOCK_IDX)[:, None]  # (BI, 1)
    suf_off = pid1 * BLOCK_SUF + tl.arange(0, BLOCK_SUF)[None, :]  # (1, BS)

    mask_idx = idx_off < idx_numel
    mask_suf = suf_off < suffix_numel
    mask = mask_idx & mask_suf  # (BI, BS)

    val_off = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    self_off = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)

    toff0 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff1 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff2 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff3 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff4 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff5 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)

    # ---- index-space coordinate decomposition ----
    rem_idx = idx_off
    if IDX_NDIM >= 1:
        c0 = rem_idx // idx_div0
        rem_idx = rem_idx % idx_div0
        val_off += c0 * val_adv0
        if M >= 1:
            toff0 += c0 * ts_0_0
        if M >= 2:
            toff1 += c0 * ts_1_0
        if M >= 3:
            toff2 += c0 * ts_2_0
        if M >= 4:
            toff3 += c0 * ts_3_0
        if M >= 5:
            toff4 += c0 * ts_4_0
        if M >= 6:
            toff5 += c0 * ts_5_0
    if IDX_NDIM >= 2:
        c1 = rem_idx // idx_div1
        rem_idx = rem_idx % idx_div1
        val_off += c1 * val_adv1
        if M >= 1:
            toff0 += c1 * ts_0_1
        if M >= 2:
            toff1 += c1 * ts_1_1
        if M >= 3:
            toff2 += c1 * ts_2_1
        if M >= 4:
            toff3 += c1 * ts_3_1
        if M >= 5:
            toff4 += c1 * ts_4_1
        if M >= 6:
            toff5 += c1 * ts_5_1
    if IDX_NDIM >= 3:
        c2 = rem_idx // idx_div2
        rem_idx = rem_idx % idx_div2
        val_off += c2 * val_adv2
        if M >= 1:
            toff0 += c2 * ts_0_2
        if M >= 2:
            toff1 += c2 * ts_1_2
        if M >= 3:
            toff2 += c2 * ts_2_2
        if M >= 4:
            toff3 += c2 * ts_3_2
        if M >= 5:
            toff4 += c2 * ts_4_2
        if M >= 6:
            toff5 += c2 * ts_5_2
    if IDX_NDIM >= 4:
        c3 = rem_idx // idx_div3
        rem_idx = rem_idx % idx_div3
        val_off += c3 * val_adv3
        if M >= 1:
            toff0 += c3 * ts_0_3
        if M >= 2:
            toff1 += c3 * ts_1_3
        if M >= 3:
            toff2 += c3 * ts_2_3
        if M >= 4:
            toff3 += c3 * ts_3_3
        if M >= 5:
            toff4 += c3 * ts_4_3
        if M >= 6:
            toff5 += c3 * ts_5_3
    if IDX_NDIM >= 5:
        c4 = rem_idx // idx_div4
        rem_idx = rem_idx % idx_div4
        val_off += c4 * val_adv4
        if M >= 1:
            toff0 += c4 * ts_0_4
        if M >= 2:
            toff1 += c4 * ts_1_4
        if M >= 3:
            toff2 += c4 * ts_2_4
        if M >= 4:
            toff3 += c4 * ts_3_4
        if M >= 5:
            toff4 += c4 * ts_4_4
        if M >= 6:
            toff5 += c4 * ts_5_4
    if IDX_NDIM >= 6:
        c5 = rem_idx // idx_div5
        rem_idx = rem_idx % idx_div5
        val_off += c5 * val_adv5
        if M >= 1:
            toff0 += c5 * ts_0_5
        if M >= 2:
            toff1 += c5 * ts_1_5
        if M >= 3:
            toff2 += c5 * ts_2_5
        if M >= 4:
            toff3 += c5 * ts_3_5
        if M >= 5:
            toff4 += c5 * ts_4_5
        if M >= 6:
            toff5 += c5 * ts_5_5

    # ---- load index values (int32 or int64, converted in-register) ----
    if M >= 1:
        ind = tl.load(idx0_ptr + toff0, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size0, ind)
        self_off += ind * self_adv_stride0
    if M >= 2:
        ind = tl.load(idx1_ptr + toff1, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size1, ind)
        self_off += ind * self_adv_stride1
    if M >= 3:
        ind = tl.load(idx2_ptr + toff2, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size2, ind)
        self_off += ind * self_adv_stride2
    if M >= 4:
        ind = tl.load(idx3_ptr + toff3, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size3, ind)
        self_off += ind * self_adv_stride3
    if M >= 5:
        ind = tl.load(idx4_ptr + toff4, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size4, ind)
        self_off += ind * self_adv_stride4
    if M >= 6:
        ind = tl.load(idx5_ptr + toff5, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size5, ind)
        self_off += ind * self_adv_stride5

    # ---- suffix coordinate decomposition ----
    rem_suf = suf_off
    if SUF_NDIM >= 1:
        cs0 = rem_suf // suf_div0
        rem_suf = rem_suf % suf_div0
        self_off += cs0 * self_suf_stride0
        val_off += cs0 * val_suf_stride0
    if SUF_NDIM >= 2:
        cs1 = rem_suf // suf_div1
        rem_suf = rem_suf % suf_div1
        self_off += cs1 * self_suf_stride1
        val_off += cs1 * val_suf_stride1
    if SUF_NDIM >= 3:
        cs2 = rem_suf // suf_div2
        rem_suf = rem_suf % suf_div2
        self_off += cs2 * self_suf_stride2
        val_off += cs2 * val_suf_stride2
    if SUF_NDIM >= 4:
        cs3 = rem_suf // suf_div3
        rem_suf = rem_suf % suf_div3
        self_off += cs3 * self_suf_stride3
        val_off += cs3 * val_suf_stride3
    if SUF_NDIM >= 5:
        cs4 = rem_suf // suf_div4
        rem_suf = rem_suf % suf_div4
        self_off += cs4 * self_suf_stride4
        val_off += cs4 * val_suf_stride4
    if SUF_NDIM >= 6:
        cs5 = rem_suf // suf_div5
        rem_suf = rem_suf % suf_div5
        self_off += cs5 * self_suf_stride5
        val_off += cs5 * val_suf_stride5

    # ---- load and store / accumulate ----
    v = tl.load(values_ptr + val_off, mask=mask, other=0)
    if ACCUMULATE:
        if USE_SCRATCH:
            # Scratch-based accumulate for outputs whose dtype lacks native
            # atomic_add. Scratch slots were seeded by the prologue; here we
            # only add the cast delta. Lossless for all supported dtypes.
            tl.atomic_add(
                scratch_ptr + self_off, v.to(scratch_ptr.dtype.element_ty), mask=mask
            )
        else:
            tl.atomic_add(out_ptr + self_off, v, mask=mask)
    else:
        tl.store(out_ptr + self_off, v, mask=mask)


@triton.jit
def _unsafe_index_put_scratch_kernel(
    out_ptr,
    scratch_ptr,
    idx0_ptr,
    idx1_ptr,
    idx2_ptr,
    idx3_ptr,
    idx4_ptr,
    idx5_ptr,
    meta_ptr,
    idx_numel,
    suffix_numel,
    M: tl.constexpr,
    IDX_NDIM: tl.constexpr,
    SUF_NDIM: tl.constexpr,
    PROLOGUE: tl.constexpr,
    BLOCK_IDX: tl.constexpr,
    BLOCK_SUF: tl.constexpr,
):
    # Scratch prologue/epilogue for the widened-scratch accumulate path.
    # Recomputes the touched offsets exactly as the main kernel does.
    #
    # PROLOGUE=True:  seeds scratch slots with cast(orig) read from the
    #                 cloned output (idempotent under duplicate slots).
    # PROLOGUE=False: stores cast(scratch) into out (idempotent too).
    # Both phases are race-free under duplicate target slots because every
    # writer stores the same value.
    idx_div0 = tl.load(meta_ptr + 0)
    idx_div1 = tl.load(meta_ptr + 1)
    idx_div2 = tl.load(meta_ptr + 2)
    idx_div3 = tl.load(meta_ptr + 3)
    idx_div4 = tl.load(meta_ptr + 4)
    idx_div5 = tl.load(meta_ptr + 5)
    ts_0_0 = tl.load(meta_ptr + 6)
    ts_0_1 = tl.load(meta_ptr + 7)
    ts_0_2 = tl.load(meta_ptr + 8)
    ts_0_3 = tl.load(meta_ptr + 9)
    ts_0_4 = tl.load(meta_ptr + 10)
    ts_0_5 = tl.load(meta_ptr + 11)
    ts_1_0 = tl.load(meta_ptr + 12)
    ts_1_1 = tl.load(meta_ptr + 13)
    ts_1_2 = tl.load(meta_ptr + 14)
    ts_1_3 = tl.load(meta_ptr + 15)
    ts_1_4 = tl.load(meta_ptr + 16)
    ts_1_5 = tl.load(meta_ptr + 17)
    ts_2_0 = tl.load(meta_ptr + 18)
    ts_2_1 = tl.load(meta_ptr + 19)
    ts_2_2 = tl.load(meta_ptr + 20)
    ts_2_3 = tl.load(meta_ptr + 21)
    ts_2_4 = tl.load(meta_ptr + 22)
    ts_2_5 = tl.load(meta_ptr + 23)
    ts_3_0 = tl.load(meta_ptr + 24)
    ts_3_1 = tl.load(meta_ptr + 25)
    ts_3_2 = tl.load(meta_ptr + 26)
    ts_3_3 = tl.load(meta_ptr + 27)
    ts_3_4 = tl.load(meta_ptr + 28)
    ts_3_5 = tl.load(meta_ptr + 29)
    ts_4_0 = tl.load(meta_ptr + 30)
    ts_4_1 = tl.load(meta_ptr + 31)
    ts_4_2 = tl.load(meta_ptr + 32)
    ts_4_3 = tl.load(meta_ptr + 33)
    ts_4_4 = tl.load(meta_ptr + 34)
    ts_4_5 = tl.load(meta_ptr + 35)
    ts_5_0 = tl.load(meta_ptr + 36)
    ts_5_1 = tl.load(meta_ptr + 37)
    ts_5_2 = tl.load(meta_ptr + 38)
    ts_5_3 = tl.load(meta_ptr + 39)
    ts_5_4 = tl.load(meta_ptr + 40)
    ts_5_5 = tl.load(meta_ptr + 41)
    self_adv_stride0 = tl.load(meta_ptr + 48)
    self_adv_stride1 = tl.load(meta_ptr + 49)
    self_adv_stride2 = tl.load(meta_ptr + 50)
    self_adv_stride3 = tl.load(meta_ptr + 51)
    self_adv_stride4 = tl.load(meta_ptr + 52)
    self_adv_stride5 = tl.load(meta_ptr + 53)
    self_adv_size0 = tl.load(meta_ptr + 54)
    self_adv_size1 = tl.load(meta_ptr + 55)
    self_adv_size2 = tl.load(meta_ptr + 56)
    self_adv_size3 = tl.load(meta_ptr + 57)
    self_adv_size4 = tl.load(meta_ptr + 58)
    self_adv_size5 = tl.load(meta_ptr + 59)
    suf_div0 = tl.load(meta_ptr + 60)
    suf_div1 = tl.load(meta_ptr + 61)
    suf_div2 = tl.load(meta_ptr + 62)
    suf_div3 = tl.load(meta_ptr + 63)
    suf_div4 = tl.load(meta_ptr + 64)
    suf_div5 = tl.load(meta_ptr + 65)
    self_suf_stride0 = tl.load(meta_ptr + 66)
    self_suf_stride1 = tl.load(meta_ptr + 67)
    self_suf_stride2 = tl.load(meta_ptr + 68)
    self_suf_stride3 = tl.load(meta_ptr + 69)
    self_suf_stride4 = tl.load(meta_ptr + 70)
    self_suf_stride5 = tl.load(meta_ptr + 71)

    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    idx_off = pid0 * BLOCK_IDX + tl.arange(0, BLOCK_IDX)[:, None]  # (BI, 1)
    suf_off = pid1 * BLOCK_SUF + tl.arange(0, BLOCK_SUF)[None, :]  # (1, BS)

    mask_idx = idx_off < idx_numel
    mask_suf = suf_off < suffix_numel
    mask = mask_idx & mask_suf  # (BI, BS)

    self_off = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)

    toff0 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff1 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff2 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff3 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff4 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)
    toff5 = tl.zeros((BLOCK_IDX, BLOCK_SUF), dtype=tl.int64)

    # ---- index-space coordinate decomposition ----
    rem_idx = idx_off
    if IDX_NDIM >= 1:
        c0 = rem_idx // idx_div0
        rem_idx = rem_idx % idx_div0
        if M >= 1:
            toff0 += c0 * ts_0_0
        if M >= 2:
            toff1 += c0 * ts_1_0
        if M >= 3:
            toff2 += c0 * ts_2_0
        if M >= 4:
            toff3 += c0 * ts_3_0
        if M >= 5:
            toff4 += c0 * ts_4_0
        if M >= 6:
            toff5 += c0 * ts_5_0
    if IDX_NDIM >= 2:
        c1 = rem_idx // idx_div1
        rem_idx = rem_idx % idx_div1
        if M >= 1:
            toff0 += c1 * ts_0_1
        if M >= 2:
            toff1 += c1 * ts_1_1
        if M >= 3:
            toff2 += c1 * ts_2_1
        if M >= 4:
            toff3 += c1 * ts_3_1
        if M >= 5:
            toff4 += c1 * ts_4_1
        if M >= 6:
            toff5 += c1 * ts_5_1
    if IDX_NDIM >= 3:
        c2 = rem_idx // idx_div2
        rem_idx = rem_idx % idx_div2
        if M >= 1:
            toff0 += c2 * ts_0_2
        if M >= 2:
            toff1 += c2 * ts_1_2
        if M >= 3:
            toff2 += c2 * ts_2_2
        if M >= 4:
            toff3 += c2 * ts_3_2
        if M >= 5:
            toff4 += c2 * ts_4_2
        if M >= 6:
            toff5 += c2 * ts_5_2
    if IDX_NDIM >= 4:
        c3 = rem_idx // idx_div3
        rem_idx = rem_idx % idx_div3
        if M >= 1:
            toff0 += c3 * ts_0_3
        if M >= 2:
            toff1 += c3 * ts_1_3
        if M >= 3:
            toff2 += c3 * ts_2_3
        if M >= 4:
            toff3 += c3 * ts_3_3
        if M >= 5:
            toff4 += c3 * ts_4_3
        if M >= 6:
            toff5 += c3 * ts_5_3
    if IDX_NDIM >= 5:
        c4 = rem_idx // idx_div4
        rem_idx = rem_idx % idx_div4
        if M >= 1:
            toff0 += c4 * ts_0_4
        if M >= 2:
            toff1 += c4 * ts_1_4
        if M >= 3:
            toff2 += c4 * ts_2_4
        if M >= 4:
            toff3 += c4 * ts_3_4
        if M >= 5:
            toff4 += c4 * ts_4_4
        if M >= 6:
            toff5 += c4 * ts_5_4
    if IDX_NDIM >= 6:
        c5 = rem_idx // idx_div5
        rem_idx = rem_idx % idx_div5
        if M >= 1:
            toff0 += c5 * ts_0_5
        if M >= 2:
            toff1 += c5 * ts_1_5
        if M >= 3:
            toff2 += c5 * ts_2_5
        if M >= 4:
            toff3 += c5 * ts_3_5
        if M >= 5:
            toff4 += c5 * ts_4_5
        if M >= 6:
            toff5 += c5 * ts_5_5

    # ---- load index values (int32 or int64, converted in-register) ----
    if M >= 1:
        ind = tl.load(idx0_ptr + toff0, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size0, ind)
        self_off += ind * self_adv_stride0
    if M >= 2:
        ind = tl.load(idx1_ptr + toff1, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size1, ind)
        self_off += ind * self_adv_stride1
    if M >= 3:
        ind = tl.load(idx2_ptr + toff2, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size2, ind)
        self_off += ind * self_adv_stride2
    if M >= 4:
        ind = tl.load(idx3_ptr + toff3, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size3, ind)
        self_off += ind * self_adv_stride3
    if M >= 5:
        ind = tl.load(idx4_ptr + toff4, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size4, ind)
        self_off += ind * self_adv_stride4
    if M >= 6:
        ind = tl.load(idx5_ptr + toff5, mask=mask, other=0).to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size5, ind)
        self_off += ind * self_adv_stride5

    # ---- suffix coordinate decomposition ----
    rem_suf = suf_off
    if SUF_NDIM >= 1:
        cs0 = rem_suf // suf_div0
        rem_suf = rem_suf % suf_div0
        self_off += cs0 * self_suf_stride0
    if SUF_NDIM >= 2:
        cs1 = rem_suf // suf_div1
        rem_suf = rem_suf % suf_div1
        self_off += cs1 * self_suf_stride1
    if SUF_NDIM >= 3:
        cs2 = rem_suf // suf_div2
        rem_suf = rem_suf % suf_div2
        self_off += cs2 * self_suf_stride2
    if SUF_NDIM >= 4:
        cs3 = rem_suf // suf_div3
        rem_suf = rem_suf % suf_div3
        self_off += cs3 * self_suf_stride3
    if SUF_NDIM >= 5:
        cs4 = rem_suf // suf_div4
        rem_suf = rem_suf % suf_div4
        self_off += cs4 * self_suf_stride4
    if SUF_NDIM >= 6:
        cs5 = rem_suf // suf_div5
        rem_suf = rem_suf % suf_div5
        self_off += cs5 * self_suf_stride5

    if PROLOGUE:
        orig = tl.load(out_ptr + self_off, mask=mask, other=0)
        tl.store(
            scratch_ptr + self_off, orig.to(scratch_ptr.dtype.element_ty), mask=mask
        )
    else:
        v32 = tl.load(scratch_ptr + self_off, mask=mask, other=0)
        tl.store(out_ptr + self_off, v32.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _unsafe_index_put_copy_kernel(
    in_ptr,
    out_ptr,
    meta_ptr,
    numel,
    CONTIGUOUS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Copy used to implement the output clone (empty_like + copy) without
    # dispatching to the patched aten::clone under use_gems. The common
    # contiguous case is a flat vectorized copy; the strided fallback
    # decomposes the flat position into coordinates. meta holds shape (12),
    # in_stride (12), out_stride (12) -- up to _MAX_NDIM index dims plus
    # _MAX_NDIM suffix dims; dims beyond the tensor rank are padded with
    # shape 1 / stride 0 so the decomposition below works for any rank
    # <= 12 without a rank constexpr.
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < numel
    if CONTIGUOUS:
        v = tl.load(in_ptr + off, mask=mask, other=0)
        tl.store(out_ptr + off, v, mask=mask)
    else:
        # shape0 = tl.load(meta_ptr + 0)
        shape1 = tl.load(meta_ptr + 1)
        shape2 = tl.load(meta_ptr + 2)
        shape3 = tl.load(meta_ptr + 3)
        shape4 = tl.load(meta_ptr + 4)
        shape5 = tl.load(meta_ptr + 5)
        shape6 = tl.load(meta_ptr + 6)
        shape7 = tl.load(meta_ptr + 7)
        shape8 = tl.load(meta_ptr + 8)
        shape9 = tl.load(meta_ptr + 9)
        shape10 = tl.load(meta_ptr + 10)
        shape11 = tl.load(meta_ptr + 11)
        in_stride0 = tl.load(meta_ptr + 12)
        in_stride1 = tl.load(meta_ptr + 13)
        in_stride2 = tl.load(meta_ptr + 14)
        in_stride3 = tl.load(meta_ptr + 15)
        in_stride4 = tl.load(meta_ptr + 16)
        in_stride5 = tl.load(meta_ptr + 17)
        in_stride6 = tl.load(meta_ptr + 18)
        in_stride7 = tl.load(meta_ptr + 19)
        in_stride8 = tl.load(meta_ptr + 20)
        in_stride9 = tl.load(meta_ptr + 21)
        in_stride10 = tl.load(meta_ptr + 22)
        in_stride11 = tl.load(meta_ptr + 23)
        out_stride0 = tl.load(meta_ptr + 24)
        out_stride1 = tl.load(meta_ptr + 25)
        out_stride2 = tl.load(meta_ptr + 26)
        out_stride3 = tl.load(meta_ptr + 27)
        out_stride4 = tl.load(meta_ptr + 28)
        out_stride5 = tl.load(meta_ptr + 29)
        out_stride6 = tl.load(meta_ptr + 30)
        out_stride7 = tl.load(meta_ptr + 31)
        out_stride8 = tl.load(meta_ptr + 32)
        out_stride9 = tl.load(meta_ptr + 33)
        out_stride10 = tl.load(meta_ptr + 34)
        out_stride11 = tl.load(meta_ptr + 35)

        rem = off
        c11 = rem % shape11
        rem = rem // shape11
        c10 = rem % shape10
        rem = rem // shape10
        c9 = rem % shape9
        rem = rem // shape9
        c8 = rem % shape8
        rem = rem // shape8
        c7 = rem % shape7
        rem = rem // shape7
        c6 = rem % shape6
        rem = rem // shape6
        c5 = rem % shape5
        rem = rem // shape5
        c4 = rem % shape4
        rem = rem // shape4
        c3 = rem % shape3
        rem = rem // shape3
        c2 = rem % shape2
        rem = rem // shape2
        c1 = rem % shape1
        rem = rem // shape1
        c0 = rem

        src = (
            c0 * in_stride0
            + c1 * in_stride1
            + c2 * in_stride2
            + c3 * in_stride3
            + c4 * in_stride4
            + c5 * in_stride5
            + c6 * in_stride6
            + c7 * in_stride7
            + c8 * in_stride8
            + c9 * in_stride9
            + c10 * in_stride10
            + c11 * in_stride11
        )
        dst = (
            c0 * out_stride0
            + c1 * out_stride1
            + c2 * out_stride2
            + c3 * out_stride3
            + c4 * out_stride4
            + c5 * out_stride5
            + c6 * out_stride6
            + c7 * out_stride7
            + c8 * out_stride8
            + c9 * out_stride9
            + c10 * out_stride10
            + c11 * out_stride11
        )
        v = tl.load(in_ptr + src, mask=mask, other=0)
        tl.store(out_ptr + dst, v, mask=mask)


# ---------------------------------------------------------------------------
# Host-side helpers (ports of the C++ wrapper utilities)
# ---------------------------------------------------------------------------


def _volume(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _broadcast_shapes(shapes):
    """Fast local replacement for torch.broadcast_shapes (no dispatch)."""
    ndim = max(len(s) for s in shapes)
    out = [1] * ndim
    for shape in shapes:
        pad = ndim - len(shape)
        for i, s in enumerate(shape):
            j = pad + i
            if s != 1:
                if out[j] == 1:
                    out[j] = s
                elif out[j] != s:
                    raise RuntimeError(
                        f"The size of tensor a ({out[j]}) must match the size of "
                        f"tensor b ({s}) at non-singleton dimension {j}"
                    )
    return tuple(out)


def _broadcast_strides(shape, stride, target_shape):
    """Broadcast (shape, stride) into target_shape; 0 stride on expanded dims."""
    ndim = len(target_shape)
    pad = ndim - len(shape)
    out = [0] * ndim
    for i, s in enumerate(shape):
        j = pad + i
        if s == 1 and target_shape[j] != 1:
            out[j] = 0
        elif s != target_shape[j]:
            raise RuntimeError(
                f"The expanded size of the tensor ({target_shape[j]}) must match "
                f"the existing size ({s}) at non-singleton dimension {i}."
            )
        else:
            out[j] = stride[i]
    return out


def _trailing_divisors(shape):
    ndim = len(shape)
    div = [1] * ndim
    acc = 1
    for i in range(ndim - 1, -1, -1):
        div[i] = acc
        acc *= shape[i]
    return div


def _pad(seq, n, fill):
    out = list(seq)
    out.extend([fill] * (n - len(out)))
    return out


def _pad_2d(arr, rows, cols, fill):
    out = [row + [fill] * (cols - len(row)) for row in arr]
    out.extend([[fill] * cols for _ in range(rows - len(out))])
    return out


def _floor_pow2(x):
    if x <= 1:
        return 1
    v = 1
    while v * 2 <= x:
        v *= 2
    return v


def _nearest_pow2(x, cap):
    v = 1
    while v * 2 <= x and v * 2 <= cap:
        v *= 2
    return max(1, min(v, cap))


def _heuristic_2d_blocks(idx_numel, suffix_numel):
    """Pick (BLOCK_IDX, BLOCK_SUF) for the 2D grid, mirroring the C++ wrapper.

    Triton requires tl.arange ranges to be power-of-2.
    """
    k_target = 256
    if suffix_numel <= 32:
        # Small suffix: minimize grid_y (virtually 1D) to reduce launch overhead.
        block_suf = _floor_pow2(max(1, suffix_numel))
        block_idx = _floor_pow2(max(1, min(k_target // block_suf, idx_numel)))
    elif suffix_numel >= idx_numel * 4:
        # Large suffix relative to idx: benefit from 2D grid.
        block_idx = 1
        block_suf = _nearest_pow2(suffix_numel, 256)
    elif idx_numel >= suffix_numel * 4:
        block_suf = max(1, _nearest_pow2(suffix_numel, 256))
        block_idx = _nearest_pow2(idx_numel, k_target // block_suf)
    else:
        ratio = idx_numel // max(1, suffix_numel)
        if ratio >= 16:
            block_idx, block_suf = 32, 8
        elif ratio >= 4:
            block_idx, block_suf = 16, 16
        else:
            block_idx, block_suf = 8, 32
    block_idx = _floor_pow2(max(1, min(block_idx, idx_numel)))
    block_suf = _floor_pow2(max(1, min(block_suf, suffix_numel)))
    return block_idx, block_suf


def _idx_key(indices):
    return tuple((tuple(t.shape), tuple(t.stride())) for t in indices)


@lru_cache(maxsize=1024)
def _store_meta(
    device,
    out_shape,
    out_stride,
    idx_key,
    val_shape,
    val_stride,
    idx_shape,
    suffix_shape,
    m,
):
    """Small int64 kernel-parameter buffer, cached per exact shapes/strides."""
    tensor_strides = [
        _broadcast_strides(idx_key[i][0], idx_key[i][1], idx_shape) for i in range(m)
    ]
    ts = _pad_2d(tensor_strides, _MAX_NDIM, _MAX_NDIM, 0)
    val_strides = _broadcast_strides(
        val_shape, val_stride, tuple(idx_shape) + tuple(suffix_shape)
    )
    idx_ndim = len(idx_shape)
    vals = (
        list(_pad(_trailing_divisors(idx_shape), _MAX_NDIM, 1))
        + [ts[r][c] for r in range(_MAX_NDIM) for c in range(_MAX_NDIM)]
        + list(_pad(val_strides[:idx_ndim], _MAX_NDIM, 0))
        + list(_pad(out_stride[:m], _MAX_NDIM, 0))
        + list(_pad(out_shape[:m], _MAX_NDIM, 1))
        + list(_pad(_trailing_divisors(suffix_shape), _MAX_NDIM, 1))
        + list(_pad(out_stride[m:], _MAX_NDIM, 0))
        + list(_pad(val_strides[idx_ndim:], _MAX_NDIM, 0))
    )
    return torch.tensor(vals, dtype=torch.int64, device=device)


@lru_cache(maxsize=1024)
def _copy_meta(device, shape, in_stride, out_stride):
    # Pad to 2 * _MAX_NDIM dims: the op supports up to _MAX_NDIM index dims
    # plus _MAX_NDIM suffix dims.
    vals = (
        list(_pad(list(shape), 2 * _MAX_NDIM, 1))
        + list(_pad(list(in_stride), 2 * _MAX_NDIM, 0))
        + list(_pad(list(out_stride), 2 * _MAX_NDIM, 0))
    )
    return torch.tensor(vals, dtype=torch.int64, device=device)


def _launch_copy(dst, src):
    contiguous = src.is_contiguous() and dst.is_contiguous()
    meta = _copy_meta(
        str(dst.device),
        tuple(src.shape),
        tuple(src.stride()),
        tuple(dst.stride()),
    )
    numel = src.numel()
    if contiguous:
        block, num_warps = 8192, 8
    else:
        block, num_warps = 1024, 4
    _fast_launch(
        _unsafe_index_put_copy_kernel,
        (triton.cdiv(numel, block),),
        (src, dst, meta, numel, contiguous, block),
        (contiguous, block, num_warps, src.dtype, dst.dtype, _aligned(src)),
        num_warps=num_warps,
    )


def _clone_like(inp):
    """empty_like + Triton copy: avoids the patched aten::clone under use_gems."""
    out = torch.empty_like(inp)
    if inp.numel() > 0:
        _launch_copy(out, inp)
    return out


_SCRATCH_DTYPES = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.int8: torch.int32,
    torch.int16: torch.int32,
    torch.uint8: torch.int32,
    torch.bool: torch.int32,
}


def _launch(
    out, indices, values, idx_shape, suffix_shape, idx_numel, suffix_numel, accumulate
):
    """2D-grid kernel launch; accumulate=True uses scratch for narrow dtypes."""
    m = len(indices)
    idx_ndim = len(idx_shape)
    suf_ndim = len(suffix_shape)

    meta = _store_meta(
        str(out.device),
        tuple(out.shape),
        tuple(out.stride()),
        _idx_key(indices),
        tuple(values.shape),
        tuple(values.stride()),
        tuple(idx_shape),
        tuple(suffix_shape),
        m,
    )
    # Pad index tensor list for kernel args (kernel always takes _MAX_NDIM
    # pointers; padded slots are never read thanks to the M constexpr).
    kernel_idx = indices + [indices[0]] * (_MAX_NDIM - m)
    # dtype + 16-byte-alignment tags of the index tensors (part of the fast
    # launch cache key; the padded pointer slots reuse indices[0]).
    key = tuple((t.dtype, _aligned(t)) for t in indices)

    block_idx, block_suf = _heuristic_2d_blocks(idx_numel, suffix_numel)
    grid = (
        triton.cdiv(idx_numel, block_idx),
        triton.cdiv(suffix_numel, block_suf),
    )

    use_scratch = accumulate and out.dtype in _SCRATCH_DTYPES
    if use_scratch:
        scratch_dtype = _SCRATCH_DTYPES[out.dtype]
        # scratch is indexed by the same element offsets used for `out` (valid
        # for any positive-stride layout), so size it to cover out's maximum
        # reachable offset + 1. No contiguous working copy is needed.
        max_off = 0
        for d in range(out.dim()):
            if out.size(d) > 0:
                max_off += (out.size(d) - 1) * out.stride(d)
        scratch = torch.empty(max_off + 1, dtype=scratch_dtype, device=out.device)

        # Prologue: seed the scratch slots with cast(orig).
        _fast_launch(
            _unsafe_index_put_scratch_kernel,
            grid,
            (
                out,
                scratch,
                *kernel_idx,
                meta,
                idx_numel,
                suffix_numel,
                m,
                idx_ndim,
                suf_ndim,
                True,
                block_idx,
                block_suf,
            ),
            (m, idx_ndim, suf_ndim, True, block_idx, block_suf, out.dtype, *key),
            num_warps=4,
        )
        # Main: atomic_add cast deltas into scratch.
        _fast_launch(
            _unsafe_index_put_kernel,
            grid,
            (
                out,
                values,
                scratch,
                *kernel_idx,
                meta,
                idx_numel,
                suffix_numel,
                m,
                idx_ndim,
                suf_ndim,
                True,
                True,
                block_idx,
                block_suf,
            ),
            (
                m,
                idx_ndim,
                suf_ndim,
                True,
                True,
                block_idx,
                block_suf,
                out.dtype,
                values.dtype,
                *key,
            ),
            num_warps=4,
        )
        # Epilogue: out = cast(scratch) with a single rounding.
        _fast_launch(
            _unsafe_index_put_scratch_kernel,
            grid,
            (
                out,
                scratch,
                *kernel_idx,
                meta,
                idx_numel,
                suffix_numel,
                m,
                idx_ndim,
                suf_ndim,
                False,
                block_idx,
                block_suf,
            ),
            (m, idx_ndim, suf_ndim, False, block_idx, block_suf, out.dtype, *key),
            num_warps=4,
        )
    else:
        # scratch_ptr is unused when USE_SCRATCH is False
        _fast_launch(
            _unsafe_index_put_kernel,
            grid,
            (
                out,
                values,
                out,
                *kernel_idx,
                meta,
                idx_numel,
                suffix_numel,
                m,
                idx_ndim,
                suf_ndim,
                accumulate,
                False,
                block_idx,
                block_suf,
            ),
            (
                m,
                idx_ndim,
                suf_ndim,
                accumulate,
                False,
                block_idx,
                block_suf,
                out.dtype,
                values.dtype,
                *key,
            ),
            num_warps=4,
        )


# ---------------------------------------------------------------------------
# Operator entry point
# ---------------------------------------------------------------------------


def _unsafe_index_put(inp, indices, values, accumulate=False):
    """_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor

    Functional advanced indexing scatter. Returns a new tensor. Bool/byte
    masks are expanded via nonzero; each of the first ``len(indices)`` dims
    of ``self`` is indexed by the corresponding index tensor and the
    remaining dims form the value suffix. ``unsafe`` means no upper bounds
    checking is performed (negative indices wrap), matching PyTorch's
    contract. accumulate=True uses atomic_add (with a widened scratch buffer
    for dtypes without native atomics), so duplicate-index summation order is
    arbitrary GPU scheduling; integer accumulation is exact.
    """
    logger.debug("GEMS _UNSAFE_INDEX_PUT")

    if torch.is_tensor(indices):
        indices = [indices]
    indices = list(indices)

    if not indices:
        raise ValueError("At least one index tensor is required")

    # Device/dtype fixups mirroring aten's _index_put_impl_.
    if values.device != inp.device:
        if values.numel() == 1 and values.dim() == 0:
            values = values.to(inp.device)
        else:
            raise RuntimeError(
                f"Expected all tensors to be on the same device, but found at "
                f"least two devices, {inp.device} and {values.device}!"
            )
    if values.dtype != inp.dtype:
        raise RuntimeError(
            f"Index put requires the source and destination dtypes match, got "
            f"{inp.dtype} for the destination and {values.dtype} for the source."
        )

    # Index preprocessing: expand bool/byte masks, fix devices.
    processed = []
    for idx in indices:
        if idx is None:
            raise TypeError(
                "_unsafe_index_put does not accept None indices (expected Tensor, but got NoneType)"
            )
        if idx.device != inp.device:
            idx = idx.to(inp.device)
        if idx.dtype in (torch.bool, torch.uint8):
            processed.extend(idx.nonzero(as_tuple=True))
        elif idx.dtype in (torch.int32, torch.int64):
            processed.append(idx)
        else:
            raise TypeError(
                "tensors used as indices must be long, int, byte or bool tensors"
            )

    m = len(processed)
    if m > inp.dim():
        raise IndexError(f"too many indices for tensor of dimension {inp.dim()}")
    if m > _MAX_NDIM:
        raise IndexError(f"too many index tensors (max {_MAX_NDIM})")

    suffix_shape = inp.shape[m:]
    suf_ndim = len(suffix_shape)
    if suf_ndim > _MAX_NDIM:
        raise IndexError(f"suffix ndim out of range: {suf_ndim}")

    idx_shape = _broadcast_shapes([t.shape for t in processed])
    idx_ndim = len(idx_shape)
    if idx_ndim > _MAX_NDIM:
        raise IndexError(f"index space rank too large: {idx_ndim}")

    idx_numel = _volume(idx_shape)
    suffix_numel = _volume(suffix_shape)

    if inp.device.type != "cuda":
        # Non-CUDA fallback: native implementation (bypasses flag_gems' own
        # _index_put_impl_ patch which only applies to the CUDA dispatch key).
        out = inp.clone()
        torch._index_put_impl_(out, processed, values, accumulate, unsafe=True)
        return out

    # Functional contract: the input is never modified. empty_like + Triton
    # copy avoids the patched aten::clone dispatch under use_gems.
    out = _clone_like(inp)

    if idx_numel == 0 or suffix_numel == 0:
        return out

    _launch(
        out,
        processed,
        values,
        idx_shape,
        suffix_shape,
        idx_numel,
        suffix_numel,
        accumulate,
    )
    return out
