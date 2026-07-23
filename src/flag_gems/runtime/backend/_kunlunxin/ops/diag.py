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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils.tensor_wrapper import StridedBuffer

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# diag is a pure element copy along a (strided) diagonal:
#   1D->2D: scatter the N input values onto the diagonal of a zeroed MxM matrix
#   2D->1D: gather the diagonal (stride0+stride1) of a 2D matrix into a 1D vector
# The old kunlunxin kernels were raw @triton.jit (no @libentry cache), so every
# shape/launch recompiled -> the IR dump shows ~950 diag_1d_to_2d_kernel + ~1060
# diag_2d_to_1d_kernel modules (120MB / 198K lines). Worse, diag_1d_to_2d used
# the UNBOUNDED BLOCK_SIZE=next_pow2(cdiv(N,12)) anti-pattern -> a giant
# constexpr tile (IR shows tensor<131072>) that ConvertTritonXPUToLLVM
# materializes per element. Route both directions through the tuned
# pointwise_dynamic `copy_func` over a StridedBuffer diagonal view: bounded
# tiles + autoGrid + libentry caching (compiled once per ndim), same recipe as
# view_copy/cat. The diagonal is expressed as a strided view so no index math is
# needed in-kernel.
config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def copy_func(x):
    return x


def _as_strided_dtype(t):
    # StridedBuffer computes byte offsets via get_dtype_bytes -> torch.iinfo,
    # which does NOT support torch.bool (raises TypeError). Reinterpret a bool
    # tensor as uint8 (identical 1-byte itemsize, no data movement) so the
    # strided diagonal copy works for bool the same as any 1-byte dtype.
    if t.dtype == torch.bool:
        return t.view(torch.uint8)
    return t


def diag_1d_to_2d(x, diagonal=0):
    N = x.shape[0]
    M = N + abs(diagonal)
    output = torch.zeros((M, M), dtype=x.dtype, device=x.device)
    if N == 0:
        return output

    # Flat position of out[i, i+diagonal] (diagonal>=0) is i*(M+1)+diagonal;
    # out[i-diagonal, i] (diagonal<0) is i*(M+1) + (-diagonal)*M. Either way the
    # diagonal is a stride-(M+1) 1D view into the MxM output.
    if diagonal >= 0:
        out_offset = diagonal
    else:
        out_offset = -diagonal * M

    x_w = _as_strided_dtype(x)
    out_w = _as_strided_dtype(output)
    in_view = StridedBuffer(x_w, (N,), (x_w.stride(0),))
    out_view = StridedBuffer(out_w, (N,), (M + 1,), offset=out_offset)
    copy_func.instantiate(1)(in_view, out0=out_view)
    return output


def diag_2d_to_1d(x, diagonal=0):
    N, M = x.shape
    if diagonal >= 0:
        diag_len = min(N, M - diagonal)
    else:
        diag_len = min(N + diagonal, M)
    if diag_len <= 0:
        return torch.empty(0, dtype=x.dtype, device=x.device)

    output = torch.empty(diag_len, dtype=x.dtype, device=x.device)
    stride0 = x.stride(0)
    stride1 = x.stride(1)

    # x[i, i+diagonal] (diagonal>=0) -> flat i*(stride0+stride1)+diagonal*stride1;
    # x[i-diagonal, i] (diagonal<0) -> flat i*(stride0+stride1)+(-diagonal)*stride0.
    if diagonal >= 0:
        in_offset = diagonal * stride1
    else:
        in_offset = -diagonal * stride0

    x_w = _as_strided_dtype(x)
    out_w = _as_strided_dtype(output)
    in_view = StridedBuffer(x_w, (diag_len,), (stride0 + stride1,), offset=in_offset)
    out_view = StridedBuffer(out_w, (diag_len,), (out_w.stride(0),))
    copy_func.instantiate(1)(in_view, out0=out_view)
    return output


def diag(x, diagonal=0):
    logger.debug("GEMS_KUNLUNXIN DIAG")
    if x.dim() == 1:
        return diag_1d_to_2d(x, diagonal)
    elif x.dim() == 2:
        return diag_2d_to_1d(x, diagonal)
    else:
        raise ValueError("Input must be a 1D or 2D tensor.")
