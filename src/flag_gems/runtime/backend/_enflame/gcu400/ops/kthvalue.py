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
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.topk import topk as gems_topk
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _gcu_int64_indices():
    # torch_gcu reads an int64 tensor with a true 8-byte element stride only
    # when TORCH_GCU_ENABLE_INT64_AND_UINT64 selects the int64 layout; with the
    # env off it applies the Long->Int replacement and reads the same tensor
    # with a 4-byte stride (tensor metadata is identical in both regimes, so
    # the env is the only discriminator). The kernel must store indices at the
    # matching width or torch reads back interleaved halves. Accepted values
    # mirror torch_gcu's parse (measured 2026-11-12): "1", "true", and nonzero
    # numerics enable int64; unset, "0", "false", "TRUE"/"True"/"on"/"yes" keep
    # the replacement. Checked per launch because torch_gcu's own
    # AllowINT64AndUINT64Ctx flips the env inside a process.
    val = os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64")
    if val is None:
        return False
    if val == "true":
        return True
    try:
        return int(val) != 0
    except ValueError:
        return False


# The external /opt/triton_gcu converter pipeline occasionally dies with a
# host-level SIGSEGV in one of its shell helpers (perl / rm / gcu-compiler-opt
# / topscc; ~4% of fresh compiles under load, same binary+args succeeds on
# retry -- see compiler_issues/flaggems_kthvalue_toolchain_flaky_segv.md).
# The crash surfaces as a Python Exception carrying the helper's stderr, and a
# failed compile is not cached, so re-invoking the launch retries it.
_TOOLCHAIN_FLAKE_MARKERS = (
    "Segmentation fault",
    "converterResult",
    "serializing the module",
    "core dumped",
)
_TOOLCHAIN_FLAKE_ATTEMPTS = 3


def _launch_with_toolchain_retry(launch):
    for attempt in range(_TOOLCHAIN_FLAKE_ATTEMPTS):
        try:
            return launch()
        except Exception as err:
            msg = str(err)
            if attempt < _TOOLCHAIN_FLAKE_ATTEMPTS - 1 and any(
                marker in msg for marker in _TOOLCHAIN_FLAKE_MARKERS
            ):
                logger.warning(
                    "toolchain flake during kernel launch (attempt %d), retrying: %s",
                    attempt + 1,
                    msg.splitlines()[0][:200],
                )
                continue
            raise


# Envelope of the k-pass selection, mirroring the gcu400 topk small-K path:
# K <= 16 keeps the unrolled selection loop bounded, and a row (or a merge
# candidate set) of <= 4096 lanes fits vector registers instead of spilling.
_KTH_MAX_K = 16
_KTH_MAX_COLS = 4096
_KTH_CHUNK = 4096
# A program slot costs ~50ns while a row of selection costs far less, and the
# row loads only overlap when several rows travel together, so one program
# walks a [U, BLOCK_N] tile of rows (all tile loads issue back to back, the
# k passes reduce along the column axis for the whole tile at once) and the
# grid is capped far below the 65535 hardware limit. U*BLOCK_N is bounded so
# wide rows shrink the tile instead of spilling. Measured on gcu400 fp32 k=2
# vs torch.kthvalue across the kthvalue core shapes: geomean 5.2x.
_KTH_GRID_CAP = 36
_KTH_UNROLL_MAX = 16
_KTH_TILE_BUDGET = 8192
_MAX_GRID_X = 65535


@triton.jit
def _to_key(xv, live):
    # Monotone signed-int key of the float bits (bijective, so the winner's
    # exact bit pattern is recovered by inverting the key). Key order matches
    # torch's float ordering: -NaN < -inf < ... < -0.0 < +0.0 < ... < +inf
    # < +NaN. Dead lanes sink to INT32_MAX so they never win a min pass.
    bits = xv.to(tl.float32).to(tl.int32, bitcast=True)
    key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    return tl.where(live, key, 2147483647)


@triton.jit
def _from_key(vk):
    return tl.where(vk >= 0, vk, vk ^ 0x7FFFFFFF).to(tl.float32, bitcast=True)


@triton.jit
def _kth_select(key, live, cols, PAD: tl.constexpr, K: tl.constexpr):
    # K unrolled min passes; the j-th pass masks out the j-1 smaller winners.
    # Leftmost column among the minimum keys: torch's ascending-index tie
    # break. Two plain reductions beat `tl.min(..., return_indices=True)` on
    # GCU, same as the topk small-K kernel measured in this backend.
    taken = tl.zeros([PAD], tl.int1)
    for j in tl.static_range(K):
        vk = tl.min(tl.where(taken, 2147483647, key), axis=0)
        i = tl.min(tl.where((key == vk) & live & (~taken), cols, PAD), axis=0)
        taken = taken | (cols == i)
    return vk, i


@libentry()
@triton.jit
def kthvalue_kpass_kernel(
    X,
    Yv,
    Yi,
    stride_xm,
    n_rows,
    n_cols,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    U: tl.constexpr,
    INT64_IDX: tl.constexpr,
):
    # One program walks [U, BLOCK_N] tiles with a grid-stride loop (grid.x is
    # capped at 65535 by the hardware, compiler_problems P19, and far lower by
    # the per-program cost). The whole tile loads in one masked 2D access, the
    # k selection passes reduce along the column axis for all U rows at once,
    # and the winners store as one [U] vector per pass output.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    live = cols < n_cols
    cols_b = cols[None, :]
    live_b = live[None, :]
    row = pid * U
    step = nprog * U
    while row < n_rows:
        rows = row + tl.arange(0, U)
        row_ok = rows < n_rows
        offs = rows[:, None] * stride_xm + cols_b
        mask2 = live_b & row_ok[:, None]
        xv = tl.load(X + offs, mask=mask2, other=0.0)
        key = _to_key(xv, mask2)
        taken = tl.zeros([U, BLOCK_N], tl.int1)
        for j in tl.static_range(K):
            vk = tl.min(tl.where(taken, 2147483647, key), axis=1)
            # Leftmost column among the minimum keys: torch's ascending-index
            # tie break, vectorized over the U rows of the tile.
            match = (key == vk[:, None]) & mask2 & (~taken)
            i = tl.min(tl.where(match, cols_b, BLOCK_N), axis=1)
            taken = taken | (cols_b == i[:, None])
        tl.store(Yv + rows, _from_key(vk), mask=row_ok)
        # Store the winner's column at the element width torch will use when it
        # reads the int64 output: a true int64 store when the int64 layout is
        # enabled, and an int32 view of the same buffer when torch_gcu's
        # Long->Int replacement makes it read 4-byte elements (see
        # _gcu_int64_indices). Either way this stays a single launch.
        if INT64_IDX:
            tl.store(Yi + rows, i.to(tl.int64), mask=row_ok)
        else:
            tl.store(Yi.to(tl.pointer_type(tl.int32)) + rows, i, mask=row_ok)
        row += step


@libentry()
@triton.jit
def kthvalue_stage1_kernel(
    X,
    V,
    I,
    stride_xm,
    n_rows,
    n_cols,
    chunk_num,
    K: tl.constexpr,
    CHUNK: tl.constexpr,
):
    # Per (row, chunk): write the k smallest of the chunk as candidates for
    # the merge pass. Short trailing chunks pad with NaN keys / out-of-range
    # columns, which the merge filters exactly like dead lanes.
    p = tl.program_id(0)
    nprog = tl.num_programs(0)
    total = n_rows * chunk_num
    cols = tl.arange(0, CHUNK)
    while p < total:
        row = p // chunk_num
        chunk = p % chunk_num
        gcols = chunk * CHUNK + cols
        live = gcols < n_cols
        xv = tl.load(X + row * stride_xm + gcols, mask=live, other=0.0)
        key = _to_key(xv, live)
        taken = tl.zeros([CHUNK], tl.int1)
        base = p * K
        for j in tl.static_range(K):
            vk = tl.min(tl.where(taken, 2147483647, key), axis=0)
            i = tl.min(tl.where((key == vk) & live & (~taken), cols, CHUNK), axis=0)
            taken = taken | (cols == i)
            tl.store(V + base + j, _from_key(vk))
            tl.store(I + base + j, chunk * CHUNK + i)
        p += nprog


@libentry()
@triton.jit
def kthvalue_stage2_kernel(
    V,
    I,
    Yv,
    Yi,
    m,
    n_rows,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    INT64_IDX: tl.constexpr,
):
    # Merge the per-chunk candidates: the k-th smallest of the row is the
    # k-th smallest of the candidate union. Candidates are laid out in column
    # order (chunk asc, ascending inside a chunk), so the leftmost tie break
    # over candidate positions equals the leftmost column tie break.
    row = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_M)
    live = cols < m
    while row < n_rows:
        v = tl.load(V + row * m + cols, mask=live, other=0.0)
        key = _to_key(v, live)
        vk, i = _kth_select(key, live, cols, BLOCK_M, K)
        col = tl.load(I + row * m + i)
        tl.store(Yv + row, _from_key(vk))
        # Same dual-width index store as the k-pass kernel: match the element
        # stride torch_gcu uses for the int64 output (see _gcu_int64_indices).
        if INT64_IDX:
            tl.store(Yi + row, col.to(tl.int64))
        else:
            tl.store(Yi.to(tl.pointer_type(tl.int32)) + row, col)
        row += nprog


def _launch_kpass_selection(inp, k, out_shape):
    n_rows = inp.numel() // inp.shape[-1]
    n_cols = inp.shape[-1]
    x_2d = inp.reshape(n_rows, n_cols)
    y_vals = torch.empty((n_rows,), device=inp.device, dtype=torch.float32)
    y_idx = torch.empty((n_rows,), device=inp.device, dtype=torch.int64)
    chunk_num = triton.cdiv(n_cols, _KTH_CHUNK)
    int64_idx = _gcu_int64_indices()
    with torch_device_fn.device(inp.device):
        if n_cols <= _KTH_MAX_COLS:
            block_n = max(128, triton.next_power_of_2(n_cols))
            unroll = min(_KTH_UNROLL_MAX, _KTH_TILE_BUDGET // block_n)
            _launch_with_toolchain_retry(
                lambda: kthvalue_kpass_kernel[
                    (min(triton.cdiv(n_rows, unroll), _KTH_GRID_CAP),)
                ](
                    x_2d,
                    y_vals,
                    y_idx,
                    x_2d.stride(0),
                    n_rows,
                    n_cols,
                    K=k,
                    BLOCK_N=block_n,
                    U=unroll,
                    INT64_IDX=int64_idx,
                    num_warps=2,
                    num_stages=1,
                )
            )
        else:
            m = chunk_num * k
            v_buf = torch.empty((n_rows * m,), device=inp.device, dtype=torch.float32)
            i_buf = torch.empty((n_rows * m,), device=inp.device, dtype=torch.int32)
            _launch_with_toolchain_retry(
                lambda: kthvalue_stage1_kernel[(min(n_rows * chunk_num, _MAX_GRID_X),)](
                    x_2d,
                    v_buf,
                    i_buf,
                    x_2d.stride(0),
                    n_rows,
                    n_cols,
                    chunk_num,
                    K=k,
                    CHUNK=_KTH_CHUNK,
                    num_warps=2,
                    num_stages=1,
                )
            )
            _launch_with_toolchain_retry(
                lambda: kthvalue_stage2_kernel[(min(n_rows, _MAX_GRID_X),)](
                    v_buf,
                    i_buf,
                    y_vals,
                    y_idx,
                    m,
                    n_rows,
                    K=k,
                    BLOCK_M=max(128, triton.next_power_of_2(m)),
                    INT64_IDX=int64_idx,
                    num_warps=2,
                    num_stages=1,
                )
            )
    return y_vals.reshape(out_shape), y_idx.reshape(out_shape)


def kthvalue(inp, k, dim=-1, keepdim=False):
    """
    Returns the kth smallest elements and their indices along the given dimension.

    The fast path runs one k-pass selection kernel (k unrolled min reductions
    per row, mirror of the gcu400 topk small-K path in the ascending
    direction). Rows wider than the register-file budget go through a
    chunk/merge pair of the same primitive. Exotic cases (k > 16, or merge
    candidates beyond the register budget) fall back to the generic topk
    composition.
    """
    logger.debug("GEMS KTHVALUE")

    original_dtype = inp.dtype
    ndim = inp.ndim

    # Validate dim before normalization (match PyTorch error semantics)
    if dim < -ndim or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
        )

    # Handle negative dim
    if dim < 0:
        dim = dim + ndim

    dim_size = inp.shape[dim]

    # Check empty reduction dim before k validation (match PyTorch)
    if dim_size == 0:
        raise IndexError(
            f"kthvalue(): Expected reduction dim {dim} to have non-zero size."
        )

    if k < 1 or k > dim_size:
        raise RuntimeError(
            f"kthvalue(): selected number k out of range for dimension {dim}"
        )

    # Handle empty tensor
    if inp.numel() == 0:
        out_shape = list(inp.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            del out_shape[dim]
        return (
            torch.empty(out_shape, dtype=original_dtype, device=inp.device),
            torch.empty(out_shape, dtype=torch.int64, device=inp.device),
        )

    # The selection kernel compares fp32 bit patterns; other dtypes round-trip
    # through fp32 like the generic implementation.
    if original_dtype is not torch.float32:
        inp = inp.to(torch.float32)

    # Determine if we need to transpose
    need_transpose = dim != ndim - 1

    if need_transpose:
        # Transpose so that dim becomes the last dimension
        perm = list(range(ndim))
        perm[dim] = ndim - 1
        perm[-1] = dim
        inp = inp.permute(perm).contiguous()
    elif not inp.is_contiguous():
        inp = inp.contiguous()

    n_cols = inp.shape[-1]
    chunk_num = triton.cdiv(n_cols, _KTH_CHUNK)

    if k <= _KTH_MAX_K and (n_cols <= _KTH_MAX_COLS or chunk_num * k <= _KTH_MAX_COLS):
        out_shape = list(inp.shape[:-1])
        values_kth, indices_kth = _launch_kpass_selection(inp, k, out_shape)
        if need_transpose:
            # Transpose back: unsqueeze the reduced axis, apply the inverse
            # permutation (the swap is its own inverse), then drop the axis
            # unless keepdim.
            inv_perm = list(range(ndim))
            inv_perm[dim] = ndim - 1
            inv_perm[-1] = dim

            values = values_kth.unsqueeze(-1).permute(inv_perm)
            indices = indices_kth.unsqueeze(-1).permute(inv_perm)

            if not keepdim:
                values = values.squeeze(dim)
                indices = indices.squeeze(dim)
        elif keepdim:
            values = values_kth.unsqueeze(-1)
            indices = indices_kth.unsqueeze(-1)
        else:
            values = values_kth
            indices = indices_kth
    else:
        # Fallback for the exotic cases: generic topk composition (k smallest
        # sorted ascending), then take the k-th element.
        values, indices = gems_topk(inp, k, dim=-1, largest=False, sorted=True)

        slice_idx = k - 1
        if k == 1:
            values_kth = values
            indices_kth = indices
        else:
            values_kth = values[..., slice_idx : slice_idx + 1]
            indices_kth = indices[..., slice_idx : slice_idx + 1]

        if need_transpose:
            inv_perm = list(range(ndim))
            inv_perm[dim] = ndim - 1
            inv_perm[-1] = dim

            values = values_kth.permute(inv_perm)
            indices = indices_kth.permute(inv_perm)

            if not keepdim:
                values = values.squeeze(dim)
                indices = indices.squeeze(dim)
        else:
            if not keepdim:
                values = values_kth.squeeze(-1)
                indices = indices_kth.squeeze(-1)
            else:
                values = values_kth
                indices = indices_kth

    if original_dtype is not torch.float32:
        values = values.to(original_dtype)
    return values, indices


# ---------------------------------------------------------------------------
# aten::kthvalue dispatch guard (torch_gcu P21 defect workaround)
#
# torch_gcu's native kthvalue writes the index output at the Long->Int
# replacement stride (4 bytes per element) even when
# TORCH_GCU_ENABLE_INT64_AND_UINT64 selects the true int64 layout, so torch
# reads the int64 indices back as interleaved halves while the values stay
# correct (compiler_problems P21, flaggems_kthvalue_int64_index_interleave.md).
# Every torch.kthvalue caller on a GCU tensor then gets garbage indices,
# including the accuracy tests' reference leg. While that layout is active,
# route the aten::kthvalue dispatch for this backend's key to this fixed
# implementation, mirroring the resize_as_sparse_ / linalg_norm in-repo
# guards: idempotent via a torch-attached flag, double-name import safe, and
# the registrar's later use_gems()/enable() registration of the same op is
# absorbed as a warning by op_registrar.for_each (same kernel either way).
# Under the replacement regime (env off) the native op is correct and the
# guard stays dormant.
# ---------------------------------------------------------------------------
_KTHVALUE_GUARD_FLAG = "_flag_gems_gcu400_kthvalue_aten_guard_registered"
_KTHVALUE_GUARD_LIBRARY = None


def _install_native_kthvalue_guard():
    global _KTHVALUE_GUARD_LIBRARY
    if not _gcu_int64_indices():
        return
    if getattr(torch, _KTHVALUE_GUARD_FLAG, False):
        return
    library = torch.library.Library("aten", "IMPL")
    try:
        library.impl("kthvalue", kthvalue, runtime.device.dispatch_key)
    except RuntimeError as error:
        logger.warning(
            "kthvalue %s-key registration skipped: %s",
            runtime.device.dispatch_key,
            error,
        )
        return
    _KTHVALUE_GUARD_LIBRARY = library
    setattr(torch, _KTHVALUE_GUARD_FLAG, True)


_install_native_kthvalue_guard()
