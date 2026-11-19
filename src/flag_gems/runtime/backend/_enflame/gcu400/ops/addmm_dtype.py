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

"""t9 4/5 candidate: gcu400 tiered implementation of addmm_dtype / addmm_dtype_out.

Production flag_gems.addmm_dtype uses the @libtuner autotune kernel of the common flag_gems/ops/addmm.py;
gcu400/ops/addmm.py already has a static tiered kernel, but only addmm / addmm_out are registered.
This implementation keeps all validation and fp32 accumulation semantics of common addmm_dtype_out,
and only routes "regular large GEMM + contiguous inputs" to the gcu400 kernel (bn=64,
avoiding the redundant bias.broadcast_to(...).contiguous() copy). Small dimensions / non-contiguous /
scalar or vector bias / beta=0 all fall back to the common implementation, keeping semantics and numerics unchanged.
"""

import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.addmm import _select_addmm_out_config
from flag_gems.runtime.backend._enflame.gcu400.ops.addmm import (
    addmm_kernel as _g4_addmm_kernel,
)

logger = logging.getLogger(__name__)


def addmm_dtype(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1):
    """Allocate the out_dtype output and hand it to addmm_dtype_out."""
    logger.debug("GEMS_ENFLAME_GCU400 ADDMM_DTYPE")
    out = torch.empty(
        (mat1.shape[0], mat2.shape[1]),
        device=mat1.device,
        dtype=out_dtype,
    )
    return addmm_dtype_out(bias, mat1, mat2, out_dtype, beta=beta, alpha=alpha, out=out)


def addmm_dtype_out(bias, mat1, mat2, out_dtype, *, beta=1, alpha=1, out):
    """Validation matches the common implementation; dispatch to static tiered kernels by shape / memory layout."""
    logger.debug("GEMS_ENFLAME_GCU400 ADDMM_DTYPE_OUT")
    if mat1.dtype != mat2.dtype:
        raise RuntimeError(
            f"mat1 and mat2 must have the same dtype, but got {mat1.dtype} and {mat2.dtype}"
        )
    if out.dtype != out_dtype:
        raise RuntimeError(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    if not (
        out_dtype == mat1.dtype
        or (
            out_dtype == torch.float32 and mat1.dtype in (torch.float16, torch.bfloat16)
        )
    ):
        raise RuntimeError(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs"
        )
    if bias.dtype != out_dtype and bias.dtype != mat1.dtype:
        raise RuntimeError("self dtype must match either out_dtype or mat1 dtype")

    # With beta=0 PyTorch semantics ignore bias (including NaN/Inf); keep the common implementation's handling.
    bias_c = bias if beta == 0 else bias.to(out_dtype)
    M, K = mat1.shape
    N = mat2.shape[1]

    # Only a regular large GEMM with all tensors contiguous and bias already a full (M,N) matrix uses
    # the gcu400 kernel; all other cases fall back to the common implementation to avoid stride / broadcast semantic
    # differences.
    use_g4 = (
        beta != 0
        and M >= 64
        and N >= 64
        and K >= 64
        and mat1.is_contiguous()
        and mat2.is_contiguous()
        and out.is_contiguous()
        and bias_c.is_contiguous()
        and tuple(bias_c.shape) == (M, N)
    )
    if not use_g4:
        from flag_gems.ops.addmm import addmm_out as _public_addmm_out

        return _public_addmm_out(bias_c, mat1, mat2, beta=beta, alpha=alpha, out=out)

    block_m, block_n, block_k, num_warps, num_stages = _select_addmm_out_config(
        M, N, K, mat1.dtype
    )
    # C-family 11 rounds: bn=64 reproduced 1.386/1.391/1.398x, better than default 256's 1.298x.
    block_n = 64
    even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    with torch_device_fn.device(mat1.device):
        _g4_addmm_kernel[grid](
            mat1,
            mat2,
            bias_c,
            out,
            alpha,
            beta,
            M,
            N,
            K,
            mat1.stride(0),
            mat1.stride(1),
            mat2.stride(0),
            mat2.stride(1),
            bias_c.stride(0),
            bias_c.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            GROUP_M=8,
            EVEN=even,
            BIAS_IS_VECTOR=False,
            BIAS_IS_SCALAR=False,
            BIAS_IS_COLVEC=False,
            SKIP_BIAS=False,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out
