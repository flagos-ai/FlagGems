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

"""Moore Threads MUSA backend specialized polar kernel.

Vendor override of ``flag_gems.ops.polar`` optimized for MTT S5000 + MUSA.
Uses a dedicated dense kernel for FP32 dense-contiguous same-shape inputs;
all other cases fall back to ``flag_gems.ops.polar``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.polar import polar as default_polar
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_INT32_MAX = 2**31 - 1


@libentry()
@triton.jit
def _polar_dense_kernel(
    abs_ptr,
    angle_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr = 1024,
    num_warps: tl.constexpr = 4,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    abs_val = tl.load(abs_ptr + offs, mask=mask)
    angle_val = tl.load(angle_ptr + offs, mask=mask)

    real = abs_val * tl.cos(angle_val)
    imag = abs_val * tl.sin(angle_val)
    results = tl.interleave(real, imag)

    out_offs = pid * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
    out_mask = out_offs < N * 2
    tl.store(output_ptr + out_offs, results, mask=out_mask)


def _supported_dense_input(abs: torch.Tensor, angle: torch.Tensor) -> bool:
    """Whether the dense specialized kernel applies.

    Dense path requires:
      * both inputs on ``musa``;
      * abs.dtype == angle.dtype == torch.float32;
      * abs.device == angle.device;
      * identical shapes (no broadcast);
      * at least 1-D (0-D falls back to preserve complex scalar shape);
      * non-empty;
      * interleaved output indices fit in int32;
      * both contiguous.
    """
    if not abs.is_musa or not angle.is_musa:
        return False
    if abs.dtype != angle.dtype or abs.dtype != torch.float32:
        return False
    if abs.device != angle.device:
        return False
    if abs.shape != angle.shape:
        return False
    if abs.ndim == 0:
        return False

    numel = abs.numel()
    if numel == 0:
        return False

    # MTHREADS kernel indexing is int32. The interleaved output
    # contains 2 * numel FP32 elements, so max index is 2*numel-1.
    # We must have 2*numel - 1 <= _INT32_MAX.
    if numel > _INT32_MAX // 2:
        return False

    if not abs.is_contiguous() or not angle.is_contiguous():
        return False

    return True


def polar(abs, angle):
    """MUSA-specialized polar, with semantic-correct fallback."""
    logger.debug("GEMS_MTHREADS POLAR")

    if not _supported_dense_input(abs, angle):
        return default_polar(abs, angle)

    N = abs.numel()
    output = torch.empty(*abs.shape, 2, dtype=abs.dtype, device=abs.device)

    abs_flat = abs.view(-1)
    angle_flat = angle.view(-1)
    output_flat = output.view(-1)

    grid = (triton.cdiv(N, 1024),)
    with torch_device_fn.device(abs.device):
        _polar_dense_kernel[grid](
            abs_flat,
            angle_flat,
            output_flat,
            N,
            BLOCK_SIZE=1024,
            num_warps=4,
        )

    return torch.view_as_complex(output)


__all__ = ["polar"]
