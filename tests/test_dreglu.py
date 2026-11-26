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

    TE_OP = getattr(tex, "dreglu", None)
except ImportError:
    TE_OP = None


def _reference_dreglu(
    grad_output: torch.Tensor, input_tensor: torch.Tensor
) -> torch.Tensor:
    """High-precision (fp64) reference for dReGLU backward.

    ``dReGLU(a, b)`` with ``a = input[..., :N]``, ``b = input[..., N:]`` has
    ``grad_a = grad_output * (a > 0) * b`` and ``grad_b = grad_output * relu(a)``.

    Used where the native TransformerEngine call is unavailable.
    """
    n = input_tensor.shape[-1] // 2
    a = input_tensor[..., :n].to(torch.float64)
    b = input_tensor[..., n:].to(torch.float64)
    g = grad_output.to(torch.float64)

    grad_a = g * (a > 0).to(torch.float64) * b
    grad_b = g * torch.clamp(a, min=0.0)
    return torch.cat([grad_a, grad_b], dim=-1)


@pytest.mark.dreglu
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dreglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flag_gems.device
    )

    if TE_OP is None:
        # TransformerEngine is unavailable; use the fp64 torch reference.
        ref_out = _reference_dreglu(
            utils.to_reference(grad_output, True),
            utils.to_reference(input_tensor, True),
        )
    else:
        ref_out = utils.to_reference(TE_OP(grad_output, input_tensor, None))
    res_out = flag_gems.dreglu(grad_output, input_tensor, None)
    utils.gems_assert_close(res_out, ref_out, dtype)
