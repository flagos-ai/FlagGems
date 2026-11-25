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

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24
BLOCK_SIZE = 4096
MAX_POLYNOMIAL_ORDER = 5

# Small-input tile ladder. A fixed BLOCK=8192 leaves tensors up to 8192
# elements with only one CTA, under-using the device; finer tiles raise
# occupancy. Measured on gcu:0 (do_bench warmup=20/rep=60, 3 alternating A/B
# rounds): n=256 -8.21%, 1024 -6.68%, 4096 -2.81%, 8192 -1.54%; shapes above
# 8192 stay within +-0.9% of the baseline, so they keep the original tile.
# (upper bound on n_total (inclusive), BLOCK)
CHE_TILE_LEVELS = ((1024, 1024), (8192, 4096))
CHE_DEFAULT_BLOCK = 8192


# Chebyshev polynomial of the second kind U_n(x), for n in [0, 5]:
# U_0 = 1, U_1 = 2x, U_2 = 4x^2 - 1, U_3 = 8x^3 - 4x,
# U_4 = 16x^4 - 12x^2 + 1, U_5 = 32x^5 - 32x^3 + 6x.
# The wrapper rejects orders outside this range before launching a kernel.
@triton.jit
def _chebyshev_u_scalar_poly(x_value, polynomial_order: tl.constexpr):
    """Evaluate one polynomial selected at compile time for a scalar order.

    Keeping the order as a ``tl.constexpr`` avoids emitting runtime integer
    comparisons, which are sensitive to int32/int64 promotion on GCU, and
    lets the compiler discard every polynomial that is not requested.
    """
    x_squared = x_value * x_value
    x_cubed = x_squared * x_value
    x_fourth = x_cubed * x_value
    x_fifth = x_fourth * x_value

    if polynomial_order == 0:
        result = tl.full(x_value.shape, 1.0, tl.float32)
    elif polynomial_order == 1:
        result = 2.0 * x_value
    elif polynomial_order == 2:
        result = 4.0 * x_squared - 1.0
    elif polynomial_order == 3:
        result = 8.0 * x_cubed - 4.0 * x_value
    elif polynomial_order == 4:
        result = 16.0 * x_fourth - 12.0 * x_squared + 1.0
    else:
        result = 32.0 * x_fifth - 32.0 * x_cubed + 6.0 * x_value
    return result


@triton.jit
def _chebyshev_u_tensor_poly(x_value, n_value):
    """Evaluate U_n(x) with a per-element order loaded from an int32 tensor."""
    x_squared = x_value * x_value
    x_cubed = x_squared * x_value
    x_fourth = x_cubed * x_value
    x_fifth = x_fourth * x_value

    poly_0 = tl.full(x_value.shape, 1.0, tl.float32)
    poly_1 = 2.0 * x_value
    poly_2 = 4.0 * x_squared - 1.0
    poly_3 = 8.0 * x_cubed - 4.0 * x_value
    poly_4 = 16.0 * x_fourth - 12.0 * x_squared + 1.0
    poly_5 = 32.0 * x_fifth - 32.0 * x_cubed + 6.0 * x_value

    n_order = n_value.to(tl.int32)
    return tl.where(
        n_order == 0,
        poly_0,
        tl.where(
            n_order == 1,
            poly_1,
            tl.where(
                n_order == 2,
                poly_2,
                tl.where(n_order == 3, poly_3, tl.where(n_order == 4, poly_4, poly_5)),
            ),
        ),
    )


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _chebyshev_u_scalar_kernel(
    x_ptr,
    out_ptr,
    N_total,
    POLYNOMIAL_ORDER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane_offsets = tl.arange(0, BLOCK)
    for block_id in tl.range(program_id, (N_total + BLOCK - 1) // BLOCK, num_programs):
        offsets = block_id * BLOCK + lane_offsets
        mask = offsets < N_total
        x_value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        result = _chebyshev_u_scalar_poly(x_value, POLYNOMIAL_ORDER)
        tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _chebyshev_u_tensor_kernel(x_ptr, n_ptr, out_ptr, N_total, BLOCK: tl.constexpr):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    lane_offsets = tl.arange(0, BLOCK)
    for block_id in tl.range(program_id, (N_total + BLOCK - 1) // BLOCK, num_programs):
        offsets = block_id * BLOCK + lane_offsets
        mask = offsets < N_total
        x_value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        n_value = tl.load(n_ptr + offsets, mask=mask, other=0)
        result = _chebyshev_u_tensor_poly(x_value, n_value)
        tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


def special_chebyshev_polynomial_u(x, n):
    """Compute U_n(x) on GCU400 with flat kernels and a fixed program grid.

    The generic layer routes through ``pointwise_dynamic``, whose codegen and
    broadcast scaffolding is expensive on this toolchain and produces
    zero-stride expanded loads. This implementation flattens both operands,
    uses a compile-time-specialized scalar-order kernel, and keeps the grid
    bounded by ``NUM_SIPS * 2`` with in-kernel striding for large tensors.
    """
    logger.debug("GEMS_ENFLAME SPECIAL_CHEBYSHEV_POLYNOMIAL_U")
    assert x.dtype == torch.float32, f"unsupported dtype {x.dtype}"

    n_is_scalar = not isinstance(n, torch.Tensor)
    if n_is_scalar:
        min_order = max_order = int(n)
    else:
        # Run the range guard on a CPU copy to avoid dispatching back into the
        # gems reduction kernels while use_gems() is active.
        n_reference = n.detach().to("cpu", dtype=torch.int32)
        min_order = int(n_reference.amin().item())
        max_order = int(n_reference.amax().item())

    if max_order > MAX_POLYNOMIAL_ORDER or min_order < 0:
        raise ValueError(
            f"Chebyshev polynomial order n must be in [0, {MAX_POLYNOMIAL_ORDER}], "
            f"got values in [{min_order}, {max_order}]"
        )

    x_contiguous = x.contiguous()
    out = torch.empty_like(x_contiguous)
    n_total = x_contiguous.numel()
    if n_total == 0:
        return out
    if x.device.type != flag_gems.device:
        raise ValueError(f"Input must be a {flag_gems.device} tensor")

    block = CHE_DEFAULT_BLOCK
    for limit, tile in CHE_TILE_LEVELS:
        if n_total <= limit:
            block = tile
            break
    grid_size = min((n_total + block - 1) // block, NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        if n_is_scalar:
            _chebyshev_u_scalar_kernel[(grid_size,)](
                x_contiguous,
                out,
                n_total,
                POLYNOMIAL_ORDER=min_order,
                BLOCK=block,
                num_warps=4,
            )
        else:
            # Materialize broadcast orders contiguously: a zero-stride expanded
            # tensor miscompiles on this toolchain.
            n_flat = (
                n.to(device=x.device, dtype=torch.int32)
                .expand(x_contiguous.shape)
                .reshape(-1)
                .contiguous()
            )
            _chebyshev_u_tensor_kernel[(grid_size,)](
                x_contiguous,
                n_flat,
                out,
                n_total,
                BLOCK=block,
                num_warps=4,
            )

    if x.layout == torch.strided and x.is_contiguous():
        return out
    return out.view_as(x)
