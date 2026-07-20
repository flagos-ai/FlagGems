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

from typing import List

import torch

from .w8a8_block_fp8_bmm import w8a8_block_fp8_bmm


def fp8_einsum(
    equation: str,
    x: torch.Tensor,
    xs: torch.Tensor,
    y: torch.Tensor,
    ys: torch.Tensor,
    block_size: List[int] = [128, 128],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-wise FP8 einsum, mirroring deep_gemm.fp8_einsum.

    Only the ``"bhr,hdr->bhd"`` contraction is supported: h is the batch
    dimension and the per-head op is ``out[b,h,d] = sum_r x[b,h,r] * y[h,d,r]``.

    Args:
        equation:     must be ``"bhr,hdr->bhd"``.
        x:            (b, h, r) FP8 data.
        xs:           (b, h, r // block_k) FP32 per-token scales.
        y:            (h, d, r) FP8 data.
        ys:           (h, d // block_n, r // block_k) FP32 per-block scales.
        block_size:   [block_n, block_k] of the FP8 scaling grid.
        output_dtype: dtype of the freshly allocated output.

    Returns:
        z: a newly allocated (b, h, d) tensor with the result.
    """
    assert (
        equation == "bhr,hdr->bhd"
    ), f"fp8_einsum only supports 'bhr,hdr->bhd', got {equation!r}"
    b, h, r = x.shape
    h2, d, r2 = y.shape
    assert h2 == h and r2 == r, f"x {tuple(x.shape)} / y {tuple(y.shape)} mismatch"

    z = torch.empty((b, h, d), device=x.device, dtype=output_dtype)

    # h is the batch dim → BMM layout (B=h, M=b, N=d, K=r). The permutes are
    # pure views (last dim r stays contiguous); the kernel handles xs's strides.
    w8a8_block_fp8_bmm(
        x.permute(1, 0, 2),  # (h, b, r)
        y,  # (h, d, r)
        xs.permute(1, 0, 2),  # (h, b, r // block_k)
        ys,  # (h, d // block_n, r // block_k)
        block_size=block_size,
        z=z.permute(1, 0, 2),  # (h, b, d) view into the (b, h, d) output
        output_dtype=output_dtype,
    )
    return z
