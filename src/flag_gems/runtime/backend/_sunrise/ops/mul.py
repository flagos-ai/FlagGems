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
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import ComplexMode

logger = logging.getLogger(__name__)


config_for_broadcast = CodeGenConfig(
    8192,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=False,
    # num_warps=16
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_for_broadcast,
)
@triton.jit
def mul_func(x, y):
    return x * y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func_scalar(x, y):
    return x * y


@pointwise_dynamic(
    is_tensor=[True, True, True, True],  # ar, ai, br, bi
    num_outputs=2,
    promotion_methods=[(0, 1, 2, 3, "DEFAULT"), (0, 1, 2, 3, "DEFAULT")],
)
@triton.jit
def mul_complex_kernel(ar, ai, br, bi):
    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return real, imag


# Register complex support
mul_func.register_complex(mode=ComplexMode.CROSS, cross_kernel=mul_complex_kernel)
mul_func_scalar.register_complex(
    mode=ComplexMode.CROSS, tensorize_scalars=True, fallback_target=mul_func
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


def _complex_mul(A, B):
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
        ar, ai = Ar[..., 0], Ar[..., 1]
        br, bi = Br[..., 0], Br[..., 1]
        common_dtype = torch.promote_types(ar.dtype, br.dtype)
        ar, ai = ar.to(common_dtype), ai.to(common_dtype)
        br, bi = br.to(common_dtype), bi.to(common_dtype)
        real_out, imag_out = mul_complex_kernel(ar, ai, br, bi)
        out = torch.stack((real_out, imag_out), dim=-1)
        return _view_as_complex_ptpu_safe(out.contiguous()).to(result_dtype)

    if A_is_complex:
        Ar = _operand_as_real_ptpu_safe(A, result_dtype, target_shape, device)
        if isinstance(B, torch.Tensor):
            Br = B.unsqueeze(-1)
            out_real = mul_func(Ar, Br)
        else:
            out_real = mul_func_scalar(Ar, B)
        return _view_as_complex_ptpu_safe(out_real.contiguous()).to(result_dtype)

    Br = _operand_as_real_ptpu_safe(B, result_dtype, target_shape, device)
    if isinstance(A, torch.Tensor):
        Ar = A.unsqueeze(-1)
        out_real = mul_func(Ar, Br)
    else:
        out_real = mul_func_scalar(Br, A)
    return _view_as_complex_ptpu_safe(out_real.contiguous()).to(result_dtype)


def mul(A, B):
    logger.debug("GEMS_SUNRISE MUL")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        return _complex_mul(A, B)
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return mul_func(A, B)
    elif isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B)
    elif isinstance(B, torch.Tensor):
        return mul_func_scalar(B, A)
    else:
        # Both scalar
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_SUNRISE MUL_")
    if isinstance(B, torch.Tensor):
        return mul_func(A, B, out0=A)
    else:
        return mul_func_scalar(A, B, out0=A)
