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

import torch
import triton
import triton.language as tl

from flag_gems.ops.norm import norm as default_norm
from flag_gems.ops.norm import norm_scalar as default_norm_scalar
from flag_gems.ops.norm import norm_scalaropt_dim as default_norm_scalaropt_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as tle

pow = tl_extra_shim.pow
logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

# Reduction identifiers. Wrapped as tl.constexpr so they can be referenced from
# within @triton.jit kernels (plain module globals are not accessible there).
# 0: L2 (sum of squares), 1: +inf (max abs), 2: -inf (min abs),
# 3: L0 (count nonzero), 4: general Lp (sum of |x|^p).
_RED_L2 = tl.constexpr(0)
_RED_MAX = tl.constexpr(1)
_RED_MIN = tl.constexpr(2)
_RED_L0 = tl.constexpr(3)
_RED_LP = tl.constexpr(4)


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=1),
    ],
    key=["M"],
)
@triton.jit(do_not_specialize=["ord"])
def norm_partial_kernel(
    X,
    Partial,
    M,
    ord,
    num_blocks,
    RED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid-stride pass 1: each program folds many BLOCK_SIZE-wide tiles into a
    # single partial, so the grid stays bounded regardless of M (the generic
    # kernel launches ~sqrt(M) programs each doing one giant vector load, which
    # collapses occupancy on large tensors).
    pid = tle.program_id(0)
    if RED == _RED_MAX:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    elif RED == _RED_MIN:
        acc = tl.full([BLOCK_SIZE], float("inf"), dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    start = pid * BLOCK_SIZE
    stride = num_blocks * BLOCK_SIZE
    for off in range(start, M, stride):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < M
        if RED == _RED_MIN:
            a = tl.load(X + cols, mask=mask, other=float("inf")).to(tl.float32)
            acc = tl.minimum(tl.abs(a), acc)
        else:
            a = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            if RED == _RED_L2:
                acc += a * a
            elif RED == _RED_MAX:
                acc = tl.maximum(tl.abs(a), acc)
            elif RED == _RED_L0:
                acc += tl.where(a != 0, 1.0, 0.0)
            else:  # _RED_LP
                acc += pow(tl.abs(a), ord)

    if RED == _RED_MAX:
        val = tl.max(acc)
    elif RED == _RED_MIN:
        val = tl.min(acc)
    else:
        val = tl.sum(acc)
    tl.store(Partial + pid, val)


@libentry()
@triton.jit(do_not_specialize=["ord"])
def norm_finalize_kernel(
    Partial,
    Out,
    num_blocks,
    ord,
    RED: tl.constexpr,
    BLOCK_MID: tl.constexpr,
):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < num_blocks
    if RED == _RED_MIN:
        p = tl.load(Partial + offset, mask=mask, other=float("inf")).to(tl.float32)
        out = tl.min(p)
    elif RED == _RED_MAX:
        p = tl.load(Partial + offset, mask=mask, other=0.0).to(tl.float32)
        out = tl.max(p)
    else:
        p = tl.load(Partial + offset, mask=mask, other=0.0).to(tl.float32)
        s = tl.sum(p)
        if RED == _RED_L2:
            out = tl.sqrt(s)
        elif RED == _RED_L0:
            out = s
        else:  # _RED_LP
            out = pow(tl.abs(s), 1.0 / ord)
    tl.store(Out, out)


def _red_kind(p):
    if p == 2:
        return _RED_L2.value, 2.0
    if p == float("inf"):
        return _RED_MAX.value, 0.0
    if p == -float("inf"):
        return _RED_MIN.value, 0.0
    if p == 0:
        return _RED_L0.value, 0.0
    return _RED_LP.value, float(p)


def _is_full_reduction(x, dim) -> bool:
    if dim is None:
        return True
    if isinstance(dim, (list, tuple)):
        if len(dim) == 0:
            return True
        axes = {d % x.ndim for d in dim}
        return len(axes) == x.ndim
    return False


def _use_triton_kernel(x, p, dim) -> bool:
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    # Only the full-tensor reduction is specialized here; a partial per-dim
    # reduction defers to the generic implementation. torch expresses a full
    # reduction as dim=None, dim=[] (empty sequence), or a dim list covering
    # every axis (e.g. [0, 1] for a 2-D input) -- all handled as full reduction.
    if not _is_full_reduction(x, dim):
        return False
    if x.numel() == 0:
        return False
    if p is not None and not isinstance(p, (int, float)):
        return False
    if isinstance(p, float) and math.isnan(p):
        return False
    return True


def norm(x, p=2, dim=None, keepdim=False):
    logger.debug("GEMS_MTHREADS NORM")
    if not _use_triton_kernel(x, p, dim):
        return default_norm(x, p=p, dim=dim, keepdim=keepdim)

    dtype = x.dtype
    red, ord_val = _red_kind(p)

    x = x.contiguous()
    M = x.numel()
    # Cap the grid so pass 1 stays occupancy-bound rather than launch-bound; a
    # few thousand programs saturate the device while keeping the pass-2 reduce
    # over the partials small.
    max_blocks = 4096
    x_flat = x.view(-1)
    out = torch.empty([1] * x.ndim, dtype=dtype, device=x.device)

    with torch_device_fn.device(x.device):
        num_blocks = min(max_blocks, triton.cdiv(M, 1024))
        num_blocks = max(1, num_blocks)
        partial = torch.empty([num_blocks], dtype=torch.float32, device=x.device)
        grid = (num_blocks,)
        norm_partial_kernel[grid](x_flat, partial, M, ord_val, num_blocks, red)
        block_mid = triton.next_power_of_2(num_blocks)
        norm_finalize_kernel[(1,)](partial, out, num_blocks, ord_val, red, block_mid)

    if not keepdim:
        out = out.reshape([])
    return out


def norm_scalar(x, p=2):
    logger.debug("GEMS_MTHREADS NORM_SCALAR")
    if not _use_triton_kernel(x, p, None):
        return default_norm_scalar(x, p=p)
    return norm(x, p=p, dim=None, keepdim=False)


def norm_scalaropt_dim(x, p, dim, keepdim=False):
    logger.debug("GEMS_MTHREADS NORM_SCALAR_OPT_DIM")
    # Only the full-tensor case is specialized; dim reductions defer to generic.
    if not _use_triton_kernel(x, p, dim):
        return default_norm_scalaropt_dim(x, p, dim, keepdim=keepdim)
    return norm(x, p=p, dim=None, keepdim=keepdim)
