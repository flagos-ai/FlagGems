# Copyright 2026, The FlagOS Contributors.
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
#

"""Launch-only FP32 specializations; all other contracts use generic addmm."""

import logging
from functools import lru_cache

import torch
import triton

from flag_gems.ops.addmm import addmm_kernel
from flag_gems.ops.addmm import addmm_out as generic_addmm_out

logger = logging.getLogger(__name__)

_CONFIGS = {
    (414, 5376, 32): (32, 128, 32, 4, 2),
    (8598, 96, 5376): (128, 32, 32, 2, 5),
    (75938, 32, 5376): (128, 32, 32, 2, 4),
}


def _configuration(bias, mat1, mat2, out, beta, alpha):
    if out is None or alpha != 1 or beta != 1:
        return None
    tensors = (bias, mat1, mat2, out)
    if any(
        t.device != mat1.device or t.device.type != "cuda" or t.dtype != torch.float32
        for t in tensors
    ):
        return None
    if torch.cuda.get_device_capability(mat1.device) != (9, 0):
        return None
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        return None
    if bias.ndim != 1 or any(t.ndim != 2 for t in (mat1, mat2, out)):
        return None
    m, k = mat1.shape
    n = mat2.shape[1]
    config = _CONFIGS.get((m, n, k))
    if (
        config is None
        or mat2.shape[0] != k
        or out.shape != (m, n)
        or bias.shape != (n,)
    ):
        return None
    if (
        mat1.stride() != (k, 1)
        or mat2.stride() != (1, k)
        or out.stride() != (n, 1)
        or bias.stride() != (1,)
    ):
        return None
    if any(
        out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
        for t in tensors[:-1]
    ):
        return None
    return config


@lru_cache(maxsize=1)
def _fixed_kernel():
    # This backend owns the launch choice and reuses the unchanged generic
    # arithmetic, bypassing only its autotuner for the measured launch shapes.
    node = addmm_kernel
    while not isinstance(node, triton.runtime.jit.JITFunction):
        node = node.fn
    return node


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    config = _configuration(bias, mat1, mat2, out, beta, alpha)
    if config is None:
        return generic_addmm_out(bias, mat1, mat2, beta=beta, alpha=alpha, out=out)
    logger.debug("GEMS_NVIDIA ADDMM_OUT fixed FP32 launch")
    m, k = mat1.shape
    n = mat2.shape[1]
    bm, bn, bk, warps, stages = config
    with torch.cuda.device(mat1.device):
        _fixed_kernel()[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            mat1,
            mat2,
            bias,
            out,
            1,
            1,
            m,
            n,
            k,
            k,
            1,
            1,
            k,
            0,
            1,
            n,
            1,
            BIAS_IS_VECTOR=True,
            BIAS_IS_SCALAR=False,
            HAS_K=True,
            IS_FP64=False,
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            num_warps=warps,
            num_stages=stages,
        )
    return out
