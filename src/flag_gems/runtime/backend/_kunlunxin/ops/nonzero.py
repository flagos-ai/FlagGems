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
    pid = ext.program_id(0)
    j = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = j < n_out
    i = j // ndim
    d = j % ndim
    stride_d = tl.load(strides + d, mask=mask)
    shape_d = tl.load(shape + d, mask=mask)
    coord = (i // stride_d) % shape_d
    tl.store(out + j, coord, mask=mask)


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
                    is_use_mask_zero=True,
                )
        if as_tuple:
            return torch.unbind(out, dim=0)
        return out

    # SPARSE path: data-dependent scatter via prefix sum.
    shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)
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
        return torch.unbind(out, dim=0)
    else:
        return out
