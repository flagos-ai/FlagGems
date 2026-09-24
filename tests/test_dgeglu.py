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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "dgeglu", None)
except ImportError:
    TE_OP = None


def _reference_dgeglu(
    grad_output: torch.Tensor, input_tensor: torch.Tensor
) -> torch.Tensor:
    """High-precision (fp64) reference for dGeGLU backward.

    ``dGeGLU(a, b)`` with ``a = input[..., :N]``, ``b = input[..., N:]`` has
    ``grad_a = grad_output * d_gelu(a) * b`` and ``grad_b = grad_output * gelu(a)``
    where ``gelu`` uses the tanh approximation
    ``gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))``.
    TE's fp16/bf16 kernel rounds at fp16/bf16 precision, so it is not a tight
    enough reference; compute the closed form in fp64 instead.
    """
    n = input_tensor.shape[-1] // 2
    a = input_tensor[..., :n].to(torch.float64)
    b = input_tensor[..., n:].to(torch.float64)
    g = grad_output.to(torch.float64)

    sqrt_2_over_pi = 0.79788456
    tanh_inner = torch.tanh(sqrt_2_over_pi * a * (1.0 + 0.044715 * a**2))
    gelu_a = 0.5 * a * (1.0 + tanh_inner)

    sech2 = 1.0 - tanh_inner**2
    d_gelu_a = 0.5 * (1.0 + tanh_inner) + 0.5 * a * sech2 * sqrt_2_over_pi * (
        1.0 + 3.0 * 0.044715 * a**2
    )

    grad_a = g * b * d_gelu_a
    grad_b = g * gelu_a
    return torch.cat([grad_a, grad_b], dim=-1)


@pytest.mark.dgeglu
@pytest.mark.skipif(TE_OP is None, reason="'dgeglu' not found in TransformerEngine")
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dgeglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flag_gems.device
    )

    ref_out = _reference_dgeglu(
        utils.to_reference(grad_output, True),
        utils.to_reference(input_tensor, True),
    )

    with flag_gems.use_gems():
        res_out = flag_gems.dgeglu(grad_output, input_tensor)

    utils.gems_assert_close(res_out, ref_out, dtype)
