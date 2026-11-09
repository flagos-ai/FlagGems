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
"""HW MRADIX TopK for GCU400, written as TLE-Struct + TLE-Raw.

One hw thread per SIP owns a strided row set (persistent kernel, grid =
min(SIP_COUNT, M)); the TLE-Struct side stages a row into VDMEM with
``tle.gpu.copy``, hands it to the TLE-Raw region (topk_mradix_l1.tops) which
runs the radix-select on the hardware MRADIX unit, and then reads the K
staged lanes back with ``tle.gpu.local_ptr``. The row's compute therefore
costs one HBM read plus a handful of VDMEM passes instead of the ``k``
sequential max/argmax sweeps the generic chunked stage1/stage2 path needs.

Ported from the Enflame kurama tree, where it is developed and validated as
``triton_gcu/test/triton36/python/triton/gcu_tle/raw/tops/
04-topk-mradix-struct-raw-persistent.{py,tops}``. Kurama is the upstream
source of truth for this kernel; keep the two copies in sync.

Capacity envelope (enforced by ``topk_mradix_supported``, same numbers as the
raw region):

- ``x.dtype`` in fp32 / fp16 / bf16. A fp16/bf16 row is staged raw (2B per
  element) and the raw region folds it to 16-bit sort keys in place, so those
  dtypes scan half the bytes a fp32 row does and need four MRADIX digits
  instead of seven.
- ``N <= L1_MAX_COLS`` (65536): the launcher caches the whole row in the VDMEM
  window it allocates (256KB of the SIP's 896KB).
- ``K <= TOPK_K_MAX`` (640): the reclaim staging inside the 16KB workspace has
  ``WS_STAGE_K_MAX`` slots for each of the value and index streams.
- ``sorted=True`` additionally needs ``K <= SORT_K_MAX`` (512): the region's S4
  bitonic argsort works on whole 128-lane vectors and rewrites the staging in
  place, and 512 lanes is where the staging window (640 slots) and the
  per-subthread cost still make sense.

``sorted=False`` returns the reclaimed (unspecified) order, which is all that
contract promises; ``sorted=True`` is produced by the region's own S4 stage, so
both orders cost one HBM read per row.
"""

from pathlib import Path

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle  # noqa: F401
    from triton.experimental.tle.language import raw as tle_raw
    from triton.experimental.tle.raw import dialect

    HAS_TLE_RAW = True
except ImportError:  # pragma: no cover - TLE is part of the gcu400 toolchain
    HAS_TLE_RAW = False

_EXTERN_FUNC_NAME = "topk_mradix_l1_kernel"

# LEVEL=1 radix capacity, mirrored from the raw region (see its header).
L1_MAX_COLS = 65536
TOPK_K_MAX = 640
# Largest K the region's S4 bitonic argsort covers (4 VEC_LEN chunks, see the
# banner above sort_staged_pairs in the .tops file).
SORT_K_MAX = 512

# VDMEM windows handed to the raw region. L1_WS_ELEMS must equal WS_SIZE,
# WS_STAGE_OFF / WS_STAGE_I_OFF / WS_STAGE_K_MAX mirror the staging layout the
# raw region pins with static_asserts.
L1_WS_ELEMS = tl.constexpr(4096)
WS_STAGE_OFF = tl.constexpr(2560)
WS_STAGE_K_MAX = tl.constexpr(640)
WS_STAGE_I_OFF = tl.constexpr(2560 + 640)

# No support for topk_per_row_hetero, which uses 8 warps.
NUM_WARPS = 4

# Rows staged per iteration: one (BLOCK_SIZE is the row-strided loop's step and
# is kept as a constexpr so a future multi-row variant does not touch the loop).
BLOCK_SIZE = 1

# fp16/bf16 rows keep their own dtype in the VDMEM window (see IN_KIND in the
# launcher); the f16/bf16 -> f32 trip the region does for the staged values is
# exact in both directions, so the returned values are bit-identical to the
# input.
_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

if HAS_TLE_RAW:

    _TOPS_FILE = Path(__file__).with_name("topk_mradix_l1.tops")

    # The raw region: topk_mradix_l1.tops compiled as a standalone device
    # function. deferred=True makes the call site a dsl_region that is
    # materialized at the end of the pipeline instead of being inlined at the
    # call site, so the region can sit inside the row loop and is traced once,
    # not once per iteration.
    @dialect(
        name="tops",
        file=_TOPS_FILE,
        extern_func_name=_EXTERN_FUNC_NAME,
        deferred=True,
    )
    def _edsl_topk_mradix_l1(*args, **kwargs): ...

    # The Struct side of the pipeline: one launch per shape, one raw call per
    # row. The loop lives here (not in the region) because this side owns the
    # row buffer and the staging arena, and because the row mapping is then
    # plain Triton arithmetic instead of a second copy of it inside the region.
    @triton.jit(do_not_specialize=["M", "N"])
    def topk_mradix_l1_launch(
        input_ptr,
        output_vals_ptr,
        output_idx_ptr,
        M,
        N,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        ROW_STEP: tl.constexpr,
        SORTED: tl.constexpr,
        IN_KIND: tl.constexpr,
        OUT_KIND: tl.constexpr,
    ):
        # Buffer ownership: both VDMEM windows are allocated HERE, on the
        # TLE-Struct side, and handed to the raw region as plain pointers (see
        # the OWNERSHIP note at its entry in topk_mradix_l1.tops). The raw
        # region must not declare them __local__ as well: both allocators start
        # at VDMEM offset 0 and neither side knows about the other, so raw
        # static buffers would silently overlap this window.
        row_base = tl.program_id(0) * ROW_STEP
        row_stride = tl.num_programs(0) * ROW_STEP
        # Exact trip count for this hw thread: no tail clamp, so every iteration
        # owns a real row and the writeback can store at row*K unconditionally.
        nrows = tl.cdiv(M - row_base, row_stride)
        # The row window is one BLOCK_N-element buffer of the INPUT dtype: a
        # fp16/bf16 row stays 16-bit here and the raw region folds it straight
        # to 16-bit sort keys (2B per element, half the HBM read and half the
        # S2 scan traffic). fp32 keeps its 4B window.
        l1 = tle.gpu.alloc(
            [1, BLOCK_N],
            dtype=input_ptr.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        ws = tle.gpu.alloc(
            [L1_WS_ELEMS],
            dtype=tl.uint32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        ws_ptr = tle.gpu.local_ptr(ws, (0,))
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, K_PAD)
        k_mask = offs_k < K
        for r in range(0, nrows):
            row = row_base + r * row_stride
            # S1 load: pad lanes arrive as -inf, which the raw region's S1
            # convert maps below every finite key, so they can never be
            # selected.
            tle.gpu.copy(
                input_ptr + row * N + offs_n,
                l1.slot(0), [BLOCK_N], mask=offs_n < N, other=float("-inf"),
            )
            # S1 convert .. S4 over the staged row, in place. Both VDMEM
            # windows are written by the region (l1 holds the sort keys, ws the
            # published counts and the k-element staging), so both are declared
            # as outputs: that is what orders the region against the S5 read
            # below and keeps its effects visible to the compiler.
            tle_raw.call(
                _edsl_topk_mradix_l1,
                [tle.gpu.local_ptr(l1, (0, 0)), ws_ptr, N, K, int(SORTED), IN_KIND],
                output_indices=[0, 1],
            )
            # S5 writeback: both staged streams are 32-bit words; the values are
            # fp32 bit patterns and the indices the (int32) row positions.
            sv = tl.load(tle.gpu.local_ptr(ws, (WS_STAGE_OFF + offs_k,)), mask=k_mask)
            si = tl.load(tle.gpu.local_ptr(ws, (WS_STAGE_I_OFF + offs_k,)), mask=k_mask)
            kv = sv.to(tl.float32, bitcast=True)
            ki = si.to(tl.int32, bitcast=True)
            if OUT_KIND == 1:
                kv = kv.to(tl.float16)
            elif OUT_KIND == 2:
                kv = kv.to(tl.bfloat16)
            tl.store(output_vals_ptr + row * K + offs_k, kv, mask=k_mask)
            # The index tensor is int64 (torch.topk's dtype) but a 64-bit store
            # with a per-lane mask loses half of its lanes on GCU, and
            # tensor.view(int32) is not a reliable alias of int64 storage on
            # this backend, so the indices leave the kernel as a plain int32
            # stream and are widened by torch in topk_mradix().
            tl.store(output_idx_ptr + row * K + offs_k, ki, mask=k_mask)
            # Staging is reused by the next iteration, whose reclaim scatters
            # into it again. That is already ordered by the region's own S1/S2
            # barriers (no subthread reaches reclaim before all of them have
            # passed S1, i.e. before all of them have finished this read), so
            # this barrier is defence in depth.
            tl.debug_barrier()


_MAX_GRID = None


def _max_grid(device_index=None):
    """Rows (hw threads) that can be in flight at once: one per SIP."""
    global _MAX_GRID
    if _MAX_GRID is None:
        props = torch.gcu.get_device_properties(device_index)
        _MAX_GRID = int(props.sip_count)
    return _MAX_GRID


def topk_mradix_supported(x, k, dim_is_last, largest, sorted_):
    """Is ``x`` inside the envelope the MRADIX level-1 kernel implements?

    Any rank is accepted: the reduction is always over the last dim and the
    caller flattens the leading dims into the row index before calling
    ``topk_mradix``. Everything outside the envelope (other dtypes, other dims,
    ascending order, oversized N/K, non-contiguous input) is handled by the
    generic chunked path in topk.py.
    """
    if not HAS_TLE_RAW:
        return False
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.ndim < 1:
        return False
    if not dim_is_last or not largest:
        return False
    if x.numel() == 0 or x.shape[-1] == 0:
        return False
    if not (0 < k <= TOPK_K_MAX):
        return False
    if sorted_ and k > SORT_K_MAX:
        return False
    if k > x.shape[-1] or x.shape[-1] > L1_MAX_COLS:
        return False
    if not x.is_contiguous():
        return False
    return True


def topk_mradix(x, k, out_shape, sorted_=True):
    """Top-k over the last dim of a contiguous 2-D view of ``x``."""
    n_rows, n_cols = x.shape
    y_vals = torch.empty((n_rows, k), device=x.device, dtype=x.dtype)
    y_idx32 = torch.empty((n_rows, k), device=x.device, dtype=torch.int32)
    k_pad = triton.next_power_of_2(k)
    grid = min(_max_grid(x.device.index), n_rows)
    # The raw region scans a 16-bit row in 256-lane vectors, so its window and
    # its pad lanes must be a multiple of 256 (a fp32 row scans in 128-lane
    # vectors). Padding beyond next_pow2(N) is still -inf, i.e. below every
    # finite key, so it changes nothing below.
    out_kind = 0 if x.dtype is torch.float32 else (1 if x.dtype is torch.float16 else 2)
    block_n = max(256 if out_kind != 0 else 128, triton.next_power_of_2(n_cols))
    topk_mradix_l1_launch[(grid,)](
        x,
        y_vals,
        y_idx32,
        n_rows,
        n_cols,
        K=k,
        K_PAD=k_pad,
        BLOCK_N=block_n,
        ROW_STEP=BLOCK_SIZE,
        SORTED=int(sorted_),
        IN_KIND=out_kind,
        OUT_KIND=out_kind,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return y_vals.reshape(out_shape), y_idx32.to(torch.int64).reshape(out_shape)
