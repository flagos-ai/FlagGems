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

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .cumsum import cumsum

logger = logging.getLogger(__name__)


def nonzero_kernel_heur_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["n_elements"], 12))  # cluster_num


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("nonzero"),
#     key=[
#         "n_elements",
#     ],
# )
@triton.heuristics(
    values={
        "BLOCK_SIZE": nonzero_kernel_heur_block_size,
    },
)
@triton.jit
def nonzero_kernel(
    inp,
    prefix_sum,
    out,
    n_elements: tl.constexpr,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements

    inp_vals = tl.load(inp + offset, mask=mask).to(tl.int1)
    out_offset = tl.load(prefix_sum + offset, mask=mask) - 1

    nonzero_mask = mask and inp_vals  # noqa

    idx_flat = offset
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + out_offset * ndim + dim, remainder, mask=nonzero_mask)


def _dense_block_size(n):
    # Keep the dense coordinate tile small enough for XPU LLVM lowering.
    if n <= 2048:
        return triton.next_power_of_2(n)
    return 2048


@libentry()
@triton.jit
def nonzero_dense_flat_kernel(
    out,
    n_out,
    strides,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # DENSE (no zeros): row-major output [N, ndim]. One lane per OUTPUT element,
    # j = i*ndim + d, coord = (i // stride[d]) % shape[d]. Fully contiguous store.
    # FALLBACK kernel for ndim > 8 (per-lane metadata loads); the default dense
    # path uses nonzero_dense_flat_args_kernel which keeps shape/strides in
    # kernel arguments (no global loads, no masked loads).
    pid = ext.program_id(0)
    j = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = j < n_out
    i = j // ndim
    d = j % ndim
    stride_d = tl.load(strides + d, mask=mask)
    shape_d = tl.load(shape + d, mask=mask)
    coord = (i // stride_d) % shape_d
    tl.store(out + j, coord, mask=mask)


# Cap for the default dense tile. Larger tiles measured identical throughput
# to 8192 on this backend; > 8192 risks the XPU tl.sum/lowering ceiling.
_DENSE_TILE_CAP = 8192


@libentry()
@triton.jit(do_not_specialize=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
def nonzero_dense_flat_args_kernel(
    out,
    n_out,
    ndim: tl.constexpr,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    BLOCK_SIZE: tl.constexpr,
):
    # DENSE (no zeros): row-major output [N, ndim]. One lane per OUTPUT element,
    # j = i*ndim + d, coord = (i // stride[d]) % shape[d]. Row-major strides are
    # derived in-kernel from the shape arguments, so the kernel has no global
    # metadata loads and no masked loads at all: the only memory streams are the
    # contiguous int64 stores of the output.
    pid = ext.program_id(0)
    j = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = j < n_out
    i = j // ndim
    d = j % ndim
    stride_d = tl.where(
        d == 0,
        s1 * s2 * s3 * s4 * s5 * s6 * s7,
        tl.where(
            d == 1,
            s2 * s3 * s4 * s5 * s6 * s7,
            tl.where(
                d == 2,
                s3 * s4 * s5 * s6 * s7,
                tl.where(
                    d == 3,
                    s4 * s5 * s6 * s7,
                    tl.where(
                        d == 4,
                        s5 * s6 * s7,
                        tl.where(d == 5, s6 * s7, tl.where(d == 6, s7, 1)),
                    ),
                ),
            ),
        ),
    )
    shape_d = tl.where(
        d == 0,
        s0,
        tl.where(
            d == 1,
            s1,
            tl.where(
                d == 2,
                s2,
                tl.where(
                    d == 3,
                    s3,
                    tl.where(
                        d == 4,
                        s4,
                        tl.where(d == 5, s5, tl.where(d == 6, s6, s7)),
                    ),
                ),
            ),
        ),
    )
    coord = (i // stride_d) % shape_d
    tl.store(out + j, coord, mask=mask)


# ---- exact nonzero count (dense detection) ----
# Two-phase count. The main grid-stride pass is completely mask-free: every
# program reduces TILE(=8192, the documented safe tl.sum point on this backend)
# lanes per iteration, so no correctness ceiling is touched and the loads stay
# affine/contiguous. The remainder (< GRID*TILE elements) is counted by a
# separate masked pass whose clamped-offset loads (masked `other` values are
# not trusted on this backend) only ever touch a small tail, keeping the main
# pass at full bandwidth.
_COUNT_TILE = 8192
_COUNT_GRID_CAP = 256


@libentry()
@triton.jit
def nonzero_count_tile_kernel(
    inp,
    partial,
    n_elements,
    TILE: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = ext.program_id(0)
    acc = tl.zeros([TILE], dtype=tl.int32)
    for off in range(pid * TILE, n_elements, GRID * TILE):
        cols = off + tl.arange(0, TILE)
        w = tl.load(inp + cols)
        acc += (w != 0).to(tl.int32)
    s = tl.sum(acc, axis=0)
    tl.store(partial + pid, s.to(tl.int64))


@libentry()
@triton.jit
def nonzero_count_tail_kernel(
    inp,
    partial,
    n_elements,
    n_main,
    TILE: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = n_main + pid * TILE + tl.arange(0, TILE)
    last = n_elements - 1
    cclamp = tl.minimum(cols, last)
    ok = (cols < n_elements).to(tl.int32)
    w = tl.load(inp + cclamp)
    acc = (w != 0).to(tl.int32) * ok
    s = tl.sum(acc, axis=0)
    tl.store(partial + pid, s.to(tl.int64))


@libentry()
@triton.jit
def nonzero_count_reduce_kernel(partial, out, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    v = tl.load(partial + idx)
    total = tl.sum(v, axis=0)
    tl.store(out, total.to(tl.int64))


def _count_nonzero(inp, n_elements):
    # Main pass covers full GRID*TILE strides; a masked pass covers the rest.
    total = 0
    stride = _COUNT_GRID_CAP * _COUNT_TILE
    n_main = (n_elements // stride) * stride
    if n_main > 0:
        partial = torch.empty(_COUNT_GRID_CAP, dtype=torch.int64, device=inp.device)
        with torch_device_fn.device(inp.device):
            nonzero_count_tile_kernel[(_COUNT_GRID_CAP,)](
                inp,
                partial,
                n_main,
                TILE=_COUNT_TILE,
                GRID=_COUNT_GRID_CAP,
            )
            main = torch.empty((), dtype=torch.int64, device=inp.device)
            nonzero_count_reduce_kernel[(1,)](partial, main, BLOCK=_COUNT_GRID_CAP)
        total += int(main.item())
    tail = n_elements - n_main
    if tail > 0:
        tgrid = min(
            _COUNT_GRID_CAP,
            max(1, triton.next_power_of_2(triton.cdiv(tail, _COUNT_TILE))),
        )
        partial = torch.empty(tgrid, dtype=torch.int64, device=inp.device)
        with torch_device_fn.device(inp.device):
            nonzero_count_tail_kernel[(tgrid,)](
                inp,
                partial,
                n_elements,
                n_main,
                TILE=_COUNT_TILE,
                GRID=tgrid,
            )
            out = torch.empty((), dtype=torch.int64, device=inp.device)
            nonzero_count_reduce_kernel[(1,)](partial, out, BLOCK=tgrid)
        total += int(out.item())
    return total


@libentry()
@triton.jit(do_not_specialize=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
def nonzero_shape_int32_kernel(out, s0, s1, s2, s3, s4, s5, s6, s7, ndim: tl.constexpr):
    # One launch fills the whole int32 shape table (row-major dim sizes) used
    # by the sparse scatter kernel; replaces ndim single-value launches.
    d = tl.arange(0, 8)
    vals = tl.where(
        d == 0,
        s0,
        tl.where(
            d == 1,
            s1,
            tl.where(
                d == 2,
                s2,
                tl.where(
                    d == 3,
                    s3,
                    tl.where(
                        d == 4, s4, tl.where(d == 5, s5, tl.where(d == 6, s6, s7))
                    ),
                ),
            ),
        ),
    )
    tl.store(out + d, vals, mask=d < ndim)


def _shape_int32_tensor(shape, device):
    ndim = len(shape)
    padded = list(shape[:8]) + [1] * (8 - len(shape))
    out = torch.empty(8, dtype=torch.int32, device=device)
    with torch_device_fn.device(device):
        nonzero_shape_int32_kernel[(1,)](out, *padded, ndim=ndim)
    return out


def _is_dense(inp):
    """Return (inp_bool, prefix_sum, num_nonzeros). prefix_sum is None if dense."""
    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)
    inp_bool = inp_view
    if inp_view.dtype != torch.bool:
        inp_bool = inp_view != 0
    prefix_sum = cumsum(inp_bool, dim=0)
    num_nonzeros = int(prefix_sum[n_elements - 1].item()) if n_elements > 0 else 0
    return inp, inp_bool, prefix_sum, num_nonzeros


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS_KUNLUNXIN NONZERO")

    inp_ndim = inp.ndim
    n_elements = inp.numel()

    if n_elements == 0:
        # ATen: shape (0, ndim) for empty tensors (scalar never reaches here).
        out = torch.empty(
            (0, inp_ndim) if inp_ndim else (0, 0), dtype=torch.int64, device=inp.device
        )
        if as_tuple:
            return torch.unbind(out, dim=1) if inp_ndim else ()
        return out

    inp = inp.contiguous()

    # Large inputs: an exact count (one reduction pass, no scan materialization)
    # decides dense-vs-sparse. Dense needs no prefix sum at all; the count is
    # also the exact output row count for the sparse path.
    if n_elements >= 8192 and inp.dtype != torch.bool:
        num_nonzeros = _count_nonzero(inp, n_elements)
        if (
            inp_ndim >= 1
            and num_nonzeros == n_elements
            and num_nonzeros * inp_ndim < 2**31
        ):
            return _dense_result(inp, num_nonzeros, as_tuple)
        return _sparse_result(inp, inp_ndim, n_elements, num_nonzeros, as_tuple)

    # Small inputs (and bool, which is ~50% sparse in practice): keep the
    # established prefix-sum + dense/scatter paths unchanged.
    inp, inp_bool, prefix_sum, num_nonzeros = _is_dense(inp)
    n_out = num_nonzeros * inp_ndim
    if inp_ndim >= 1 and num_nonzeros == n_elements and n_out < 2**31:
        out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)
        if n_out > 0:
            strides_t = _row_major_strides(inp.shape, inp.device)
            shape_t = _device_int_tensor(inp.shape, torch.int64, inp.device)
            block = _dense_block_size(n_out)
            grid = (triton.cdiv(n_out, block),)
            with torch_device_fn.device(inp.device):
                nonzero_dense_flat_kernel[grid](
                    out,
                    n_out,
                    strides_t,
                    shape_t,
                    inp_ndim,
                    block,
                    isCloseUnrollControl=True,
                    is_use_mask_zero=True,
                )
        if as_tuple:
            return torch.unbind(out, dim=1)
        return out

    # SPARSE path: data-dependent scatter via prefix sum.
    shape = _device_int_tensor(inp.shape, torch.int32, inp.device)
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(inp.device):
        nonzero_kernel[grid](
            inp_bool,
            prefix_sum,
            out,
            n_elements,
            shape,
            inp_ndim,
            isCloseUnrollControl=True,
            is_use_mask_zero=True,
        )

    if as_tuple:
        return torch.unbind(out, dim=1)
    else:
        return out


def _dense_result(inp, num_nonzeros, as_tuple):
    inp_ndim = inp.ndim
    n_out = num_nonzeros * inp_ndim
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)
    if n_out > 0:
        if inp_ndim <= 8:
            block = (
                min(
                    _DENSE_TILE_CAP,
                    max(64, triton.next_power_of_2(min(n_out, _DENSE_TILE_CAP))),
                )
                if n_out
                else 64
            )
            grid = (triton.cdiv(n_out, block),)
            with torch_device_fn.device(inp.device):
                nonzero_dense_flat_args_kernel[grid](
                    out,
                    n_out,
                    inp_ndim,
                    *(list(inp.shape) + [1] * (8 - inp_ndim)),
                    BLOCK_SIZE=block,
                )
        else:
            strides_t = _row_major_strides(inp.shape, inp.device)
            shape_t = _device_int_tensor(inp.shape, torch.int64, inp.device)
            block = _dense_block_size(n_out)
            grid = (triton.cdiv(n_out, block),)
            with torch_device_fn.device(inp.device):
                nonzero_dense_flat_kernel[grid](
                    out,
                    n_out,
                    strides_t,
                    shape_t,
                    inp_ndim,
                    block,
                    isCloseUnrollControl=True,
                    is_use_mask_zero=True,
                )
    if as_tuple:
        return torch.unbind(out, dim=1)
    return out


def _sparse_result(inp, inp_ndim, n_elements, num_nonzeros, as_tuple):
    inp_view = inp.view(n_elements)
    inp_bool = inp_view if inp_view.dtype == torch.bool else (inp_view != 0)
    prefix_sum = cumsum(inp_bool, dim=0)
    out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)
    if inp_ndim <= 8:
        shape = _shape_int32_tensor(inp.shape, inp.device)
    else:
        shape = _device_int_tensor(inp.shape, torch.int32, inp.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(inp.device):
        nonzero_kernel[grid](
            inp_bool,
            prefix_sum,
            out,
            n_elements,
            shape,
            inp_ndim,
            isCloseUnrollControl=True,
            is_use_mask_zero=True,
        )
    if as_tuple:
        return torch.unbind(out, dim=1)
    return out


@libentry()
@triton.jit
def nonzero_dense_dimmajor_kernel(
    out,
    n_elements: tl.constexpr,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # DENSE (no zeros): dim-major output [ndim, N]. One lane per element, each dim
    # written to a contiguous run out[dim*N + offset] -> stride-1 store per dim.
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    idx_flat = offset
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + dim * n_elements + offset, remainder, mask=mask)


@libentry()
@triton.jit
def _metadata_kernel(out, value: tl.constexpr, index: tl.constexpr):
    tl.store(out + index, value)


def _device_int_tensor(values, dtype, device):
    out = torch.empty(len(values), dtype=dtype, device=device)
    with torch_device_fn.device(device):
        for index, value in enumerate(values):
            _metadata_kernel[(1,)](out, value=value, index=index)
    return out


def _row_major_strides(shape, device):
    ndim = len(shape)
    strides = [1] * ndim
    for k in range(ndim - 2, -1, -1):
        strides[k] = strides[k + 1] * shape[k + 1]
    return _device_int_tensor(strides, torch.int64, device)
