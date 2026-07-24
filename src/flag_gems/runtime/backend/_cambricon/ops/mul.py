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

from ..utils.pointwise_dynamic import ComplexMode, pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True, True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def mul_func(x, y, inplace):
    return x * y


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def mul_func_scalar(x, y, inplace):
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


def mul(A, B):
    logger.debug("GEMS_CAMBRICON MUL")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return mul_func(A, B, False)
    elif isinstance(A, torch.Tensor):
        return mul_func_scalar(A, B, False)
    elif isinstance(B, torch.Tensor):
        return mul_func_scalar(B, A, False)
    else:
        # Both scalar
        return torch.tensor(A * B)


def mul_(A, B):
    logger.debug("GEMS_CAMBRICON MUL_")
    if not isinstance(A, torch.Tensor):
        raise TypeError("mul_ expects the first argument to be a tensor")
    dtype = torch.result_type(A, B)
    if dtype != A.dtype:
        raise RuntimeError(
            f"result type {dtype} cannot be cast to inplace dtype {A.dtype}"
        )
    if isinstance(B, torch.Tensor):
        if B.device != A.device and B.dim() == 0:
            assert B.device == torch.device("cpu"), "expect scalar tensor on cpu"
            B = B.to(A.device)
        return mul_func(A, B, True, out0=A)
    else:
        return mul_func_scalar(A, B, True, out0=A)
