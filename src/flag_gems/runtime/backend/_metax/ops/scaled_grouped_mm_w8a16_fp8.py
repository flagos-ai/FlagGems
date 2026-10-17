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

"""MetaX E4M3FN or E4M3FNUZ grouped GEMM with floating-point output."""

from typing import Optional

import torch

from ._scaled_grouped_mm_w8a16 import _scaled_grouped_mm_w8a16


def scaled_grouped_mm_w8a16_fp8(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    """Multiply grouped E4M3FN or E4M3FNUZ operands with row/column scales.

    Despite the w8a16 API name, both input operands are 8-bit and must
    have the same dtype. This entry does not accept FP16/BF16 activations.
    Output defaults to BF16; FP16 and FP32 are also supported.

    Offsets contain cumulative group ends; scales apply per row and column.
    FP32 accumulation, scaling and optional bias precede the output cast.
    """
    if not isinstance(self, torch.Tensor) or not isinstance(mat2, torch.Tensor):
        raise TypeError("mat_a and mat_b must be tensors")
    if (
        self.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        or mat2.dtype != self.dtype
    ):
        raise TypeError(
            "scaled_grouped_mm_w8a16_fp8 requires matching E4M3FN or E4M3FNUZ inputs"
        )
    return _scaled_grouped_mm_w8a16(
        self,
        mat2,
        scale_a,
        scale_b,
        offs=offs,
        bias=bias,
        scale_result=scale_result,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )
