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

from flag_gems import runtime
from flag_gems.ops.scaled_grouped_mm import (
    _check_dims,
    _default_out_dtype,
    _normalize_bias,
    _normalize_scale,
    _resolve_shapes,
    _scaled_grouped_mm_fallback,
    _supports_triton_dot,
)
from flag_gems.ops.scaled_grouped_mm import (
    scaled_grouped_mm_kernel as _generic_scaled_grouped_mm_kernel,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.device_info import get_sm_count

logger = logging.getLogger(__name__)

scaled_grouped_mm_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("scaled_grouped_mm"),
        key=["M", "N", "K", "A_IS_2D", "B_IS_2D"],
        warmup=2,
        rep=4,
    )(_generic_scaled_grouped_mm_kernel.jit_function)
)


def scaled_grouped_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    offs=None,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_METAX SCALED_GROUPED_MM")
    if scale_result is not None:
        raise RuntimeError("scale_result is not supported for scaled_grouped_mm")

    _check_dims(self, mat2)
    (
        a_is_2d,
        b_is_2d,
        num_groups,
        M,
        N,
        K,
        out_shape,
        offs,
    ) = _resolve_shapes(self, mat2, offs)

    output_dtype = out_dtype or _default_out_dtype(self.dtype)
    scale_multiplier = num_groups if a_is_2d and b_is_2d else 1
    scale_a = _normalize_scale(
        scale_a,
        self,
        dim=0,
        num_groups=num_groups,
        scale_multiplier=scale_multiplier,
        name="scale_a",
    )
    scale_b = _normalize_scale(
        scale_b,
        mat2,
        dim=1,
        num_groups=num_groups,
        scale_multiplier=scale_multiplier,
        name="scale_b",
    )
    bias, bias_mode = _normalize_bias(
        bias, a_is_2d=a_is_2d, b_is_2d=b_is_2d, num_groups=num_groups, N=N
    )

    if not _supports_triton_dot(self.dtype):
        return _scaled_grouped_mm_fallback(
            self,
            mat2,
            scale_a,
            scale_b,
            offs,
            bias,
            output_dtype,
            a_is_2d,
            b_is_2d,
            num_groups,
        )

    if self.stride(-2) > 1 and self.stride(-1) > 1:
        self = self.contiguous()
    if mat2.stride(-2) > 1 and mat2.stride(-1) > 1:
        mat2 = mat2.contiguous()

    out = torch.empty(out_shape, dtype=output_dtype, device=self.device)
    if out.numel() == 0:
        return out

    stride_ag = self.stride(0) if not a_is_2d else 0
    stride_am = self.stride(-2)
    stride_ak = self.stride(-1)
    stride_bg = mat2.stride(0) if not b_is_2d else 0
    stride_bk = mat2.stride(-2)
    stride_bn = mat2.stride(-1)
    stride_cg = out.stride(0) if out.dim() == 3 else 0
    stride_cm = out.stride(-2)
    stride_cn = out.stride(-1)
    stride_sag = scale_a.stride(0) if scale_a.dim() == 2 else 0
    stride_sbg = scale_b.stride(0) if scale_b.dim() == 2 else 0

    grid = (get_sm_count(),)
    with torch_device_fn.device(self.device):
        scaled_grouped_mm_kernel[grid](
            self,
            mat2,
            scale_a,
            scale_b,
            offs,
            bias,
            out,
            M,
            N,
            K,
            num_groups,
            stride_ag,
            stride_am,
            stride_ak,
            stride_bg,
            stride_bk,
            stride_bn,
            stride_cg,
            stride_cm,
            stride_cn,
            stride_sag,
            stride_sbg,
            A_IS_2D=a_is_2d,
            B_IS_2D=b_is_2d,
            BIAS_MODE=bias_mode,
        )
    return out
