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

"""MThreads Triton implementation of ``aten.mv``.

``A`` has shape ``[N, M]`` and ``B`` has shape ``[M]``: ``N`` is the number of
output rows and ``M`` is the reduction length.  Each program owns ``BLOCK_N``
output rows and walks the reduction dimension in ``BLOCK_M`` element tiles,
accumulating in FP32.

The launch schedule is selected by a static table rather than by the shared
``mv`` autotuner.  On MTT S5000 the reduction tile that wins is 512 or 1024
elements, which lies outside the ``block_m`` range the generic ``mv`` tuning
space offers for this backend, so autotuning converges on a schedule that is
slower than the vendor GEMV.  The table below is the outcome of a measured
sweep over ``BLOCK_N x BLOCK_M x num_warps x num_stages`` on the GEMV shapes
of the FlagOSTune inference configurations.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offset_n < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for m in range(0, M, BLOCK_M):
        offset_m = m + tl.arange(0, BLOCK_M)
        m_mask = offset_m < M
        a = tl.load(
            A + offset_n[:, None] * stride_an + offset_m[None, :] * stride_am,
            mask=n_mask[:, None] & m_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(B + offset_m * stride_bm, mask=m_mask, other=0.0).to(tl.float32)
        acc += tl.sum(a * b[None, :], axis=1)

    tl.store(C + offset_n * stride_cn, acc.to(C.dtype.element_ty), mask=n_mask)


def _launch_config(n, m):
    """Return ``(BLOCK_N, BLOCK_M, num_warps, num_stages)`` for ``[n, m] @ [m]``.

    The four leading branches are shapes whose swept optimum differs from the
    surrounding family; every other shape uses the general schedule below them.
    """

    # Sparse high-N rows at the 2048 reduction length need a narrower row
    # block than their neighbours.
    if m == 2048 and n in (3104, 9330):
        return 4, 512, 2, 3
    # Two independently swept mid-N families at the 4096 reduction length.
    if m == 4096 and 1024 < n <= 1036:
        return 2, 1024, 4, 2
    if m == 4096 and n == 2048:
        return 2, 512, 2, 4
    # Boundary of the small-N family, where one row per program under-fills
    # the machine but the mid-N row block is already too wide.
    if 480 < n <= 512 and m in (2048, 4096):
        return 2, 1024, 4, 2

    if n <= 480:
        block_n = 1
    elif n < 3072:
        block_n = 8
    elif n < 8192:
        block_n = 16
    else:
        block_n = 64

    # Prefer the widest reduction tile that divides M so the inner loop stays
    # mask-free; fall back to progressively narrower tiles otherwise.
    if n < 8192 and m % 512 == 0:
        block_m = 512
    elif m % 256 == 0:
        block_m = 256
    elif m % 128 == 0:
        block_m = 128
    else:
        block_m = 64
    return block_n, block_m, 8, 4


def mv(inp, vec):
    logger.debug("GEMS_MTHREADS MV")
    assert inp.dim() == 2 and vec.dim() == 1, "mv expects a matrix and a vector"
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"

    N, M = inp.shape
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    block_n, block_m, num_warps, num_stages = _launch_config(N, M)

    with torch_device_fn.device(inp.device):
        mv_kernel[(triton.cdiv(N, block_n),)](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            BLOCK_N=block_n,
            BLOCK_M=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


__all__ = ["mv", "mv_kernel"]
