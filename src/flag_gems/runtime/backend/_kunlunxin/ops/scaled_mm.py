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

from flag_gems.ops.scaled_mm import (
    _check_inputs,
    _normalize_bias,
    _normalize_scale,
    _resolve_out_dtype,
)

logger = logging.getLogger(__name__)


def _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, out_dtype, out):
    _check_inputs(self, mat2)
    M, _ = self.shape
    N = mat2.shape[1]
    output_dtype = _resolve_out_dtype(self, out_dtype, out)

    if out is None:
        out = torch.empty((M, N), dtype=output_dtype, device=self.device)
    elif out.shape != (M, N):
        raise RuntimeError("Incompatible output shape")
    if M == 0 or N == 0:
        return out

    scale_a, _ = _normalize_scale(scale_a, M, is_left_scale=True)
    scale_b, _ = _normalize_scale(scale_b, N, is_left_scale=False)
    bias = _normalize_bias(bias, N)

    result = torch.mm(self.to(torch.float32), mat2.to(torch.float32))
    if scale_a.numel() == 1:
        result = result * scale_a
    else:
        result = result * scale_a.reshape(M, 1)
    if scale_b.numel() == 1:
        result = result * scale_b
    else:
        result = result * scale_b.reshape(1, N)
    if bias is not None:
        result = result + bias
    out.copy_(result.to(output_dtype))
    return out


def scaled_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_MM")
    return _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, out_dtype, None)


def scaled_mm_out(
    self,
    mat2,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    *,
    out,
):
    logger.debug("GEMS_KUNLUNXIN SCALED_MM_OUT")
    return _scaled_mm_impl(self, mat2, scale_a, scale_b, bias, out_dtype, out)
