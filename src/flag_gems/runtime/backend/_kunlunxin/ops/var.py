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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

from ..utils.reduce_native import dim_compress_materializes, native_var_mean_parts

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["correction", "M", "N"])
def var_welford_kernel(
    X,
    Var,
    M,
    N,
    correction,
    BLOCK_N: tl.constexpr,
):
    # One row per program to avoid autotune correctness issues on some backends.
    pid = ext.program_id(0)
    X = X + pid * N
    Var = Var + pid

    # Two-pass approach using tl.sum to avoid tl.reduce correctness issues.
    _sum = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum) / N

    _acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        _acc += diff * diff
    var = tl.sum(_acc) / (N - correction)
    # Write var
    tl.store(Var, var)


def _composed_var(x, dim, correction, keepdim):
    # The vendor aten::var.correction kernel is inaccurate for large
    # multi-dim reductions (~1e-2 absolute at (200,40999,3) x dim=[0,1]
    # while sum is ~1e-7 relative), so compose var from exact sums.
    var, _ = native_var_mean_parts(x, dim, correction)
    out = var.to(x.dtype)
    if not keepdim:
        out = out.squeeze(dim=dim)
    return out


def var(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        # Global reduction. The generic two-stage kernel reduces its per-block
        # partials with a welford tl.reduce combine_fn containing
        # tl.maximum(count, 1) (int literal), which fails to lower on XPU
        # (arith.maxnumf MLIR type error, surfacing as
        # "OutOfResources: uni_sram PassManager::run failed"). The vendor
        # var.correction kernel is inaccurate for large reductions, so
        # compose from exact sums.
        return _composed_var(x, list(range(x.ndim)), correction, keepdim)

    shape = list(x.shape)
    dim = [d % x.ndim for d in dim]
    if dim_compress_materializes(x, dim):
        # Non-inner-dim reduction: dim_compress would materialize a permuted
        # copy through the broken strided copy_ path. Compose from exact
        # sums (the vendor var kernel is inaccurate for large reductions).
        return _composed_var(x, dim, correction, keepdim)
    x = dim_compress(x, dim)
    N = 1
    for i in dim:
        N *= shape[i]
        shape[i] = 1
    M = x.numel() // N
    var_out = torch.empty(shape, dtype=x.dtype, device=x.device)

    BLOCK_N = 1024
    grid = (M,)
    with torch_device_fn.device(x.device):
        var_welford_kernel[grid](x, var_out, M, N, correction, BLOCK_N=BLOCK_N)

    if not keepdim:
        var_out = var_out.squeeze(dim=dim)
    return var_out


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR_DIM")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN VAR_CORRECTION")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)
