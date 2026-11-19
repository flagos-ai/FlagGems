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

"""GCU400-specific addmv: like mv.py, fixed tile/grid with in-kernel striding, no autotune."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import broadcastable_to, libentry

logger = logging.getLogger(__name__)

# The fixed grid cap is set to the order of GCU400 compute units; remaining row blocks are covered by an in-kernel
# stride loop.
NUM_SIPS = 48
BLOCK_N = 16
BLOCK_M = 1024


@libentry()
@triton.jit(do_not_specialize=["M", "N", "alpha", "beta"])
def addmv_kernel(
    A,
    B,
    Inp,
    Out,
    N,
    M,
    alpha,
    beta,
    stride_an,
    stride_am,
    stride_bm,
    stride_in,
    stride_outn,
    BIAS_IS_SCALAR: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    num_programs = tl.num_programs(0)
    num_row_blocks = (N + BLOCK_N - 1) // BLOCK_N
    for row_block in tl.range(tl.program_id(0), num_row_blocks, num_programs):
        row_index = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offset_n = row_index[:, None]
        offset_m = tl.arange(0, BLOCK_M)[None, :]
        n_mask = offset_n < N

        A_ptrs = A + offset_n * stride_an + offset_m * stride_am
        B_ptrs = B + offset_m * stride_bm
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_mask = m + offset_m < M
            a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
            b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
            acc += a * b
            A_ptrs += BLOCK_M * stride_am
            B_ptrs += BLOCK_M * stride_bm

        acc = tl.sum(acc, axis=1)
        if BIAS_IS_SCALAR:
            inp = tl.load(Inp).to(tl.float32)
        else:
            inp = tl.load(
                Inp + row_index * stride_in, mask=row_index < N, other=0.0
            ).to(tl.float32)
        out_block = acc * alpha + inp * beta
        tl.store(Out + row_index * stride_outn, out_block, mask=row_index < N)


def _launch_addmv(self, mat, vec, out, beta, alpha):
    """Launch shared by addmv / addmv_out; the two differ only in how out is obtained."""
    N, M = mat.shape
    bias_is_scalar = self.numel() == 1
    self = self.broadcast_to(out.shape)
    # t8 batch Step 4 B02: with N>=512, BLOCK_N 16->32, mid/large shapes 1.25~1.43x faster;
    # small shapes keep 16 to avoid a ~7~9% regression at (256,256).
    block_n = 32 if N >= 512 else BLOCK_N
    grid = (min(NUM_SIPS, triton.cdiv(N, block_n)),)
    with torch_device_fn.device(mat.device):
        addmv_kernel[grid](
            mat,
            vec,
            self,
            out,
            N,
            M,
            alpha,
            beta,
            mat.stride(0),
            mat.stride(1),
            vec.stride(0),
            self.stride(0),
            out.stride(0),
            BIAS_IS_SCALAR=bias_is_scalar,
            BLOCK_N=block_n,
            BLOCK_M=BLOCK_M,
            num_warps=4,
        )
    return out


def addmv(self, mat, vec, *, beta=1, alpha=1):
    logger.debug("GEMS_ENFLAME ADDMV")
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert broadcastable_to(self.shape, (mat.shape[0],)), "Incompatible self shape"
    out = torch.empty((mat.shape[0],), device=mat.device, dtype=mat.dtype)
    return _launch_addmv(self, mat, vec, out, beta, alpha)


def addmv_out(self, mat, vec, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_ENFLAME ADDMV_OUT")
    assert mat.shape[1] == vec.shape[0], "incompatible dimensions"
    assert broadcastable_to(self.shape, (mat.shape[0],)), "Incompatible self shape"
    if out is None:
        out = torch.empty((mat.shape[0],), device=mat.device, dtype=mat.dtype)
    else:
        assert out.shape == (mat.shape[0],), "Incompatible output shape"
    return _launch_addmv(self, mat, vec, out, beta, alpha)
