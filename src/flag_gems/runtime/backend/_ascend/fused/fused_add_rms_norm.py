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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    X,  # pointer to the input
    R,  # pointer to the residual
    W,  # pointer to the weight
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    num_rows,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Rows are processed in-place by this kernel.  Launching more programs
    # than the number of NPU AI cores makes consecutive program "waves"
    # overwrite the same global memory while earlier waves are still running,
    # which corrupts the row handled by the first wave.  Therefore the rows
    # are distributed with a grid-stride loop and the grid is capped at the
    # AI core count so that only a single wave is launched.
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    row = pid
    while row < num_rows:
        Xr = X + row * x_stride_r
        Rr = R + row * r_stride_r

        _var_base = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(Xr + cols, mask, other=0.0).to(tl.float32)
            r = tl.load(Rr + cols, mask, other=0.0).to(tl.float32)
            x += r
            _var_base += x * x / N
        var = tl.sum(_var_base)

        rrms = 1 / tl.sqrt(var + eps)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(Xr + cols, mask, other=0.0).to(tl.float32)
            r = tl.load(Rr + cols, mask, other=0.0).to(tl.float32)
            x += r
            w = tl.load(W + cols, mask, other=0.0)
            y = (x * rrms).to(X.dtype.element_ty) * w
            # write back to residual and input
            tl.store(Rr + cols * r_stride_c, x, mask=mask)
            tl.store(Xr + cols * x_stride_c, y, mask=mask)

        row += num_programs


def fused_add_rms_norm(x, residual, normalized_shape, weight, eps=1e-5):
    """
    This function performs fused residual addition and RMS normalization **in-place**.
    Both `x` and `residual` tensors will be modified. Use with caution if these tensors
    are reused elsewhere or require gradients.
    """
    logger.debug("GEMS_ASCEND FUSED_ADD_RMS_NORM")
    dim = x.ndim - len(normalized_shape)
    M = min(math.prod(x.shape[:dim]), 65535)
    N = math.prod(normalized_shape)

    BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()

    # Never launch more programs than the number of AI cores: an in-place
    # kernel that spans several program waves corrupts data (see kernel).
    props = torch.npu.get_device_properties(x.device)
    vector_core_num = int(
        getattr(props, "vector_core_num", None)
        or getattr(props, "multi_processor_count", 1)
    )
    num_programs = max(1, min(M, vector_core_num))

    with torch_device_fn.device(x.device):
        fused_add_rms_norm_kernel[num_programs,](
            x, residual, weight, N, 1, N, 1, M, N, eps, BLOCK_SIZE
        )
    return x, residual
