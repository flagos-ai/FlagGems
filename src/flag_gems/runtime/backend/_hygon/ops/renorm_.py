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

import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim

logger = logging.getLogger(__name__)

_pow = tl_extra_shim.pow


def _configs():
    configs = []
    for block_d, block_a, num_warps in (
        (32, 1, 1),
        (64, 1, 2),
        (128, 1, 4),
        (256, 1, 4),
        (64, 8, 2),
        (128, 8, 4),
        (64, 32, 4),
        (128, 32, 8),
    ):
        configs.append(
            triton.Config({"BLOCK_D": block_d, "BLOCK_A": block_a}, num_warps=num_warps)
        )
    return configs


@libentry()
@triton.autotune(configs=_configs(), key=["A", "D", "B"])
@triton.heuristics(
    {"BLOCK_B": lambda args: min(triton.next_power_of_2(args["B"]), 1024)}
)
@triton.jit(do_not_specialize=["p", "maxnorm"])
def renorm_kernel(
    X,
    A,
    D,
    B,
    p,
    maxnorm,
    BLOCK_A: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    # ``renorm_`` normalizes each sub-tensor selected along the kept dimension
    # ``D`` of the (A, D, B) view, reducing over the ``A`` (leading) and ``B``
    # (trailing) axes. Each program owns ``BLOCK_D`` consecutive values of
    # ``d`` and streams the reduction over ``A`` in tiles of ``BLOCK_A`` rows.
    # For a fixed ``a`` the ``d``/``b`` plane is contiguous in memory (offset
    # ``a*D*B + d*B + b``), so loads stay coalesced; blocking ``A`` gives the
    # in-flight memory parallelism the wide reductions need, while keeping
    # ``BLOCK_D`` modest spreads the ``D`` sub-tensors across many programs.
    pid = tl.program_id(0).to(tl.int64)
    d = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d < D
    stride_a = D * B

    b = tl.arange(0, BLOCK_B)
    b_mask = b < B
    # plane[a, d, b] offset within one ``a`` row, broadcast over the A tile.
    plane = d[None, :, None] * B + b[None, None, :]
    dbmask = d_mask[None, :, None] & b_mask[None, None, :]

    acc = tl.zeros([BLOCK_D, BLOCK_B], dtype=tl.float32)
    for a_off in range(0, A, BLOCK_A):
        a = a_off + tl.arange(0, BLOCK_A)
        a_mask = a < A
        offs = a[:, None, None] * stride_a + plane
        mask = a_mask[:, None, None] & dbmask
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        # ``p == 2`` (the common case) avoids the costly transcendental ``pow``.
        if p == 2.0:
            part = x * x
        else:
            part = _pow(tl.abs(x), p)
        acc += tl.sum(part, axis=0)

    # Fold the per-``b`` partials so each row holds the full sub-tensor norm.
    total = tl.sum(acc, axis=1)
    if p == 2.0:
        norm = tl_extra_shim.sqrt(total)
    else:
        norm = _pow(total, 1.0 / p)
    scale = tl.where(norm > maxnorm, maxnorm / norm, 1.0)[None, :, None]

    for a_off in range(0, A, BLOCK_A):
        a = a_off + tl.arange(0, BLOCK_A)
        a_mask = a < A
        offs = a[:, None, None] * stride_a + plane
        mask = a_mask[:, None, None] & dbmask
        x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(X + offs, x * scale, mask=mask)


def renorm_(x, p, dim, maxnorm):
    logger.debug("GEMS_HYGON RENORM_")
    dim = dim % x.ndim

    # A contiguous tensor keeps standard row-major strides, so the (A, D, B)
    # offset formula in the kernel is valid; otherwise work on a contiguous
    # copy and write the result back in place.
    target = x if x.is_contiguous() else x.contiguous()

    shape = target.shape
    D = shape[dim]
    A = math.prod(shape[:dim])
    B = math.prod(shape[dim + 1 :])

    if D == 0 or A * B == 0:
        return x

    grid = lambda meta: (triton.cdiv(D, meta["BLOCK_D"]),)
    with torch_device_fn.device(x.device):
        renorm_kernel[grid](target, A, D, B, p, maxnorm)

    if target is not x:
        x.copy_(target)

    return x
