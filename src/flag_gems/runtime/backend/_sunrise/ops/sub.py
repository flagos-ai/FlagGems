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

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.pointwise_dynamic import ComplexMode

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def sub_func(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_tensor_scalar(x, y, alpha):
    return x - y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def sub_func_scalar_tensor(x, y, alpha):
    return x - y * alpha


# Register complex support (elementwise)
sub_func.register_complex(mode=ComplexMode.ELEMENTWISE)
sub_func_tensor_scalar.register_complex(
    mode=ComplexMode.ELEMENTWISE, tensorize_scalars=True, fallback_target=sub_func
)
sub_func_scalar_tensor.register_complex(
    mode=ComplexMode.ELEMENTWISE, tensorize_scalars=True, fallback_target=sub_func
)


def _view_as_real_ptpu_safe(x: torch.Tensor) -> torch.Tensor:
    """`torch.view_as_real(x)` with a CPU bounce when x is on PTPU."""
    try:
        return torch.view_as_real(x)
    except NotImplementedError:
        if x.device.type != "ptpu":
            raise
        return torch.view_as_real(x.cpu()).to(x.device)


def _view_as_complex_ptpu_safe(x: torch.Tensor) -> torch.Tensor:
    """`torch.view_as_complex(x)` with a CPU bounce when x is on PTPU."""
    try:
        return torch.view_as_complex(x)
    except NotImplementedError:
        if x.device.type != "ptpu":
            raise
        return torch.view_as_complex(x.cpu()).to(x.device)


def _scalar_complex_as_real_ptpu_safe(
    scalar, complex_dtype: torch.dtype, target_shape, device: torch.device
) -> torch.Tensor:
    """Broadcast a python complex scalar to a `view_as_real`-shaped tensor."""
    cpu_scalar = torch.tensor(scalar, dtype=complex_dtype, device="cpu").expand(
        target_shape
    )
    cpu_real = torch.view_as_real(cpu_scalar).contiguous()
    if device.type == "cpu":
        return cpu_real
    return cpu_real.to(device)


def _operand_as_real_ptpu_safe(
    value, complex_dtype: torch.dtype, target_shape, device: torch.device
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value if value.is_complex() else value.to(complex_dtype)
        return _view_as_real_ptpu_safe(tensor)
    return _scalar_complex_as_real_ptpu_safe(value, complex_dtype, target_shape, device)


def _complex_sub(A, B, alpha):
    result_dtype = torch.result_type(A, B)
    shape_a = A.shape if isinstance(A, torch.Tensor) else torch.Size([])
    shape_b = B.shape if isinstance(B, torch.Tensor) else torch.Size([])
    target_shape = torch.broadcast_shapes(shape_a, shape_b)
    device = A.device if isinstance(A, torch.Tensor) else B.device

    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )

    if A_is_complex and B_is_complex:
        Ar = _operand_as_real_ptpu_safe(A, result_dtype, target_shape, device)
        Br = _operand_as_real_ptpu_safe(B, result_dtype, target_shape, device)
        common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
        Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
        out_real = sub_func(Ar, Br, alpha)
        return _view_as_complex_ptpu_safe(out_real.contiguous()).to(result_dtype)

    if A_is_complex:
        Ar = _operand_as_real_ptpu_safe(A, result_dtype, target_shape, device)
        if isinstance(B, torch.Tensor):
            Br = _operand_as_real_ptpu_safe(B, result_dtype, target_shape, device)
        else:
            Br = _scalar_complex_as_real_ptpu_safe(
                B, result_dtype, target_shape, device
            )
        common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
        Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
        out_real = sub_func(Ar, Br, alpha)
        return _view_as_complex_ptpu_safe(out_real.contiguous()).to(result_dtype)

    Br = _operand_as_real_ptpu_safe(B, result_dtype, target_shape, device)
    if isinstance(A, torch.Tensor):
        Ar = _operand_as_real_ptpu_safe(A, result_dtype, target_shape, device)
    else:
        Ar = _scalar_complex_as_real_ptpu_safe(A, result_dtype, target_shape, device)
    common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
    Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
    out_real = sub_func(Ar, Br, alpha)
    return _view_as_complex_ptpu_safe(out_real.contiguous()).to(result_dtype)


def sub(A, B, *, alpha=1):
    logger.debug("GEMS_SUNRISE SUB")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        return _complex_sub(A, B, alpha)
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return sub_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return sub_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A - B * alpha)


def sub_(A, B, *, alpha=1):
    logger.debug("GEMS_SUNRISE SUB_")
    if isinstance(B, torch.Tensor):
        return sub_func(A, B, alpha, out0=A)
    else:
        return sub_func_tensor_scalar(A, B, alpha, out0=A)
