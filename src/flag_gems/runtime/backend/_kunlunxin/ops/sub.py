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
from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


# Reuse the tuned CodeGenConfig from less_equal_ (same generic pointwise_dynamic
# discrete-access root cause on XPU): without it the bare pointwise_dynamic
# produces discrete access -> catastrophic latency. Block=1024, unroll_num=8,
# kunlunAutoGrid, prefer_1d_tile keep the access contiguous (dtype-balanced
# speedup 0.067 -> ~1.0, 2026-09-03 trial).
config_ = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


def sub(A, B, *, alpha=1.0):
    logger.debug("GEMS_KUNLUNXIN SUB")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        if A_is_complex and B_is_complex:
            Ar = torch.view_as_real(A)
            Br = torch.view_as_real(B)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = sub_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real).to(torch.result_type(A, B))
        elif A_is_complex and not B_is_complex:
            Ar = torch.view_as_real(A)
            if isinstance(B, torch.Tensor):
                B_casted = B.to(dtype=Ar.dtype)
                Br = torch.stack([B_casted, torch.zeros_like(B_casted)], dim=-1)
            else:
                B_tensor = torch.full_like(Ar[..., 0], fill_value=B, dtype=Ar.dtype)
                Br = torch.stack([B_tensor, torch.zeros_like(B_tensor)], dim=-1)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = sub_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real.contiguous()).to(
                torch.result_type(A, B)
            )
        else:
            Br = torch.view_as_real(B)
            if isinstance(A, torch.Tensor):
                A_casted = A.to(dtype=Br.dtype)
                Ar = torch.stack([A_casted, torch.zeros_like(A_casted)], dim=-1)
            else:
                A_tensor = torch.full_like(Br[..., 0], fill_value=A, dtype=Br.dtype)
                Ar = torch.stack([A_tensor, torch.zeros_like(A_tensor)], dim=-1)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = sub_func(Ar, Br, alpha)
            return torch.view_as_complex(out_real.contiguous()).to(
                torch.result_type(A, B)
            )
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return sub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        # Both scalar
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_KUNLUNXIN SUB_")
    if isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha, out0=A)
    else:
        return sub_func_tensor_scalar(A, B, alpha, out0=A)


def subtract_(A, B, *, alpha=1):
    """In-place subtraction: A.subtract_(B, alpha) == A -= B * alpha."""
    logger.debug("GEMS_KUNLUNXIN SUBTRACT_")
    return sub_(A, B, alpha=alpha)


def subtract(A, B, *, alpha=1):
    """Out-of-place subtraction: A.subtract(B, alpha) == A - B * alpha.

    Needed for the out-of-place variant: without this override torch.subtract
    fell back to the generic KernelGen compose (flag_gems.ops.sub.sub, no
    CodeGenConfig) -> discrete access -> catastrophic 56ms on 4096x4096 fp32.
    Reuses the tuned sub() above (config_ kernels), dtype-balanced 0.08 -> ~1.0,
    2026-09-03 trial.
    """
    logger.debug("GEMS_KUNLUNXIN SUBTRACT")
    return sub(A, B, alpha=alpha)
