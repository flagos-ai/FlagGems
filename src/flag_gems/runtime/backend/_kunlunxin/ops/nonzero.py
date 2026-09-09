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
    num_nonzeros,
    n_elements: tl.constexpr,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # The scatter runs without any store mask. The previous kernel guarded the
    # store with `mask and inp_vals`: that predicate is data dependent, and a
    # data dependent store mask does not reliably suppress lanes on this
    # backend. A lane whose element is zero has out_offset = prefix_sum - 1
    # pointing at the previous run's row - or at -1 when no nonzero precedes
    # it - so a leaked store either corrupts a valid row or writes before the
    # buffer and kills the context (device error 719; every sparse nonzero
    # call crashed, taking unique2 and unique_dim down with it).
    #
    # Loads are clamped instead of masked, so a tail lane replays the last
    # element. Zero lanes (and tail lanes replaying a zero) are redirected to
    # a dummy row one past the real output, so every lane stores
    # unconditionally and in bounds; a tail lane replaying a nonzero last
    # element rewrites that element's own row with the same values.
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_c = tl.minimum(offset, n_elements - 1)

    inp_vals = tl.load(inp + offset_c).to(tl.int1)
    row = tl.load(prefix_sum + offset_c) - 1

    dummy = tl.full([], 0, tl.int64) + num_nonzeros
    safe_row = tl.where(inp_vals, row, dummy)

    idx_flat = offset_c
    for dim in tl.static_range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + safe_row * ndim + dim, remainder)


def _dense_block_size(n):
    # bounded tile -> stride-1 contiguous store (avoids unbounded-BLOCK explosion)
    if n <= 4096:
        return triton.next_power_of_2(n)
    if n <= 65536:
        return 65536
    return 65536


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
    #
    # No masks: with mask-zero load semantics a tail lane reads stride_d and
    # shape_d as 0, and its `// 0` / `% 0` raises a device exception (error
    # 719) whenever n_out is not a multiple of BLOCK_SIZE. Clamping j keeps
    # every lane on a real element; tail lanes rewrite the last coordinate
    # with the same value.
    pid = ext.program_id(0)
    j = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_c = tl.minimum(j, n_out - 1)
    i = j_c // ndim
    d = j_c % ndim
    stride_d = tl.load(strides + d)
    shape_d = tl.load(shape + d)
    coord = (i // stride_d) % shape_d
    tl.store(out + j_c, coord)


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
    # Clamp tail lanes because masked stores are not reliable on this backend.
    # Redundant lanes rewrite the final valid coordinate with the same value.
    offset_c = tl.minimum(offset, n_elements - 1)
    idx_flat = offset_c
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + dim * n_elements + offset_c, remainder)


def _row_major_strides(shape, device):
    ndim = len(shape)
    strides = [1] * ndim
    for k in range(ndim - 2, -1, -1):
        strides[k] = strides[k + 1] * shape[k + 1]
    return torch.tensor(strides, dtype=torch.int64, device=device)


def _is_dense(inp):
    """Return (inp_bool, prefix_sum, num_nonzeros). prefix_sum is None if dense."""
    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)
    inp_bool = inp_view
    if inp_view.dtype != torch.bool:
        inp_bool = inp_view != 0
    prefix_sum = inp_bool.cumsum(axis=0)
    num_nonzeros = int(prefix_sum[n_elements - 1].item()) if n_elements > 0 else 0
    return inp, inp_bool, prefix_sum, num_nonzeros


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS_KUNLUNXIN NONZERO")

    inp_ndim = inp.ndim
    inp, inp_bool, prefix_sum, num_nonzeros = _is_dense(inp)
    n_elements = inp.numel()

    n_out = num_nonzeros * inp_ndim
    # DENSE fast path: every element is non-zero -> coordinates are exactly the
    # row-major decomposition of the flat index, so we can use affine contiguous
    # stores and skip the data-dependent scatter entirely.
    if inp_ndim >= 1 and num_nonzeros == n_elements and n_out < 2**31:
        out = torch.empty(num_nonzeros, inp_ndim, dtype=torch.int64, device=inp.device)
        if n_out > 0:
            strides_t = _row_major_strides(inp.shape, inp.device)
            shape_t = torch.tensor(inp.shape, dtype=torch.int64, device=inp.device)
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
                )
        if as_tuple:
            return torch.unbind(out, dim=0)
        return out

    # SPARSE path: data-dependent scatter via prefix sum. The extra row is the
    # dummy target for zero/tail lanes (see kernel comment); it is sliced away.
    shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)
    out = torch.empty(num_nonzeros + 1, inp_ndim, dtype=torch.int64, device=inp.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(inp.device):
        nonzero_kernel[grid](
            inp_bool,
            prefix_sum,
            out,
            num_nonzeros,
            n_elements,
            shape,
            inp_ndim,
            isCloseUnrollControl=True,
        )
    out = out[0:num_nonzeros]

    if as_tuple:
        return torch.unbind(out, dim=0)
    else:
        return out
