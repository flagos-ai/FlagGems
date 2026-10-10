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

"""PPU addmm dispatch built on the shared optimized GEMM kernel."""

import logging

import torch

from flag_gems.ops.addmm import addmm as _generic_addmm
from flag_gems.ops.addmm import addmm_out as _generic_addmm_out
from flag_gems.utils import broadcastable_to

from .gemm_utils import _output_overlaps_inputs
from .mm import _can_use_ppu_mm, _dispatch_ppu_gemm

logger = logging.getLogger(__name__)


def _can_use_ppu_addmm(bias, mat1, mat2, out) -> bool:
    return (
        mat1.ndim == mat2.ndim == out.ndim == 2
        and mat1.shape[1] == mat2.shape[0]
        and broadcastable_to(bias.shape, out.shape)
        and bias.dtype == mat1.dtype
        and bias.device == mat1.device
        and _can_use_ppu_mm(mat1, mat2, out)
    )


def _run_ppu_addmm_dispatch(bias, mat1, mat2, out, alpha, beta):
    return _dispatch_ppu_gemm(mat1, mat2, out, bias=bias, alpha=alpha, beta=beta)


def addmm(bias, mat1, mat2, *, beta=1, alpha=1):
    logger.debug("GEMS_THEAD ADDMM")
    if mat1.ndim == 2 and mat2.ndim == 2 and mat1.shape[1] == mat2.shape[0]:
        out = torch.empty(
            (mat1.shape[0], mat2.shape[1]),
            device=mat1.device,
            dtype=mat1.dtype,
        )
        if _can_use_ppu_addmm(bias, mat1, mat2, out):
            return _run_ppu_addmm_dispatch(bias, mat1, mat2, out, alpha, beta)
    return _generic_addmm(bias, mat1, mat2, beta=beta, alpha=alpha)


def addmm_out(bias, mat1, mat2, *, beta=1, alpha=1, out=None):
    logger.debug("GEMS_THEAD ADDMM_OUT")
    if out is not None and _can_use_ppu_addmm(bias, mat1, mat2, out):
        if _output_overlaps_inputs(out, bias, mat1, mat2):
            return out.copy_(addmm(bias, mat1, mat2, beta=beta, alpha=alpha))
        return _run_ppu_addmm_dispatch(bias, mat1, mat2, out, alpha, beta)
    return _generic_addmm_out(
        bias,
        mat1,
        mat2,
        beta=beta,
        alpha=alpha,
        out=out,
    )


__all__ = ["addmm", "addmm_out"]
