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

from flag_gems.ops.var import var_kernel_1, var_kernel_2
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "correction"])
def iluvatar_var_twopass_kernel(
    X,
    Var,
    M,
    N,
    correction,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Two-pass variance kernel for the Iluvatar backend.

    Avoids the tl.reduce + welford combine_fn pattern which produces
    incorrect results (NaN) on Iluvatar when BLOCK_M > 1 due to a
    backend-specific issue with multi-row reductions using custom
    combine functions.
    """
    # Map the program id to the row of X it should compute.
    pid = ext.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    X = X + pid * N
    Var = Var + pid
    row_mask = pid < M

    # Pass 1: compute mean via sum
    _sum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        _sum += x

    row_mean = tl.sum(_sum, axis=1) / N  # [BLOCK_M]
    row_mean_2d = row_mean[:, None]  # [BLOCK_M, 1]

    # Pass 2: compute sum of squared deviations from mean
    _acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        col_mask = cols < N
        mask = row_mask and col_mask
        x = tl.load(X + cols, mask, other=0.0).to(tl.float32)
        diff = (x - row_mean_2d) * mask
        _acc += diff * diff

    row_acc = tl.sum(_acc, axis=1)  # [BLOCK_M]
    var = row_acc / (N - correction)
    var = var[:, None]
    tl.store(Var, var, row_mask)


def var(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ILUVATAR VAR")
    if correction is None:
        correction = 1.0

    if dim is None or len(dim) == x.ndim:
        # Full reduce: use the shared kernel_1 + kernel_2 path from ops.var
        # (these use 1D tl.reduce which works correctly on Iluvatar)
        dim = list(range(x.ndim))
        shape = [1] * x.ndim
        N = x.numel()
        var = torch.empty(shape, dtype=x.dtype, device=x.device)
        BLOCK_N = 1024
        BLOCK_NUM = triton.cdiv(N, BLOCK_N)
        acc = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        average = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)
        count = torch.empty([BLOCK_NUM], dtype=x.dtype, device=x.device)

        with torch_device_fn.device(x.device):
            var_kernel_1[(BLOCK_NUM,)](x, acc, average, count, N, BLOCK_N=BLOCK_N)
            var_kernel_2[(1,)](acc, average, count, var, N, correction, BLOCK_NUM)
    else:
        # Per-dim reduce: use the two-pass kernel to avoid the Welford
        # tl.reduce bug with BLOCK_M > 1 on Iluvatar
        shape = list(x.shape)
        dim = [d % x.ndim for d in dim]
        x = dim_compress(x, dim)
        N = 1
        for i in dim:
            N *= shape[i]
            shape[i] = 1
        M = x.numel() // N
        var = torch.empty(shape, dtype=x.dtype, device=x.device)

        BLOCK_M = 1
        BLOCK_N = 1024
        grid = (triton.cdiv(M, BLOCK_M),)
        with torch_device_fn.device(x.device):
            iluvatar_var_twopass_kernel[grid](
                x, var, M, N, correction, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
            )

    if not keepdim:
        var = var.squeeze(dim=dim)
    return var


def var_dim(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ILUVATAR VAR_DIM")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)


def var_correction(x, dim=None, *, correction=None, keepdim=False):
    logger.debug("GEMS_ILUVATAR VAR_CORRECTION")
    return var(x, dim=dim, correction=correction, keepdim=keepdim)
