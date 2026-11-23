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

    try:
        from transformer_engine.pytorch import DType as TEDType
    except ImportError:
        from transformer_engine.pytorch.cpp_extensions import DType as TEDType

    TE_OP = getattr(tex, "geglu", None)
    TORCH_TO_TE_DTYPE = {
        torch.float32: TEDType.kFloat32,
        torch.float16: TEDType.kFloat16,
        torch.bfloat16: TEDType.kBFloat16,
    }
except ImportError:
    TE_OP = None
    TORCH_TO_TE_DTYPE = {}


def _reference_geglu(
    input_tensor: torch.Tensor, quantizer=None, *te_extra_args
) -> torch.Tensor:
    """High-precision (fp64) reference for GeGLU.

    ``GeGLU(a, b)`` with ``a = input[..., :N]``, ``b = input[..., N:]`` is
    ``gelu(a) * b`` where ``gelu`` uses the tanh approximation
    ``gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * x * (1 + 0.044715 * x^2)))``.

    Takes the same arguments as TransformerEngine's entry point so it can stand
    in for it when TE is not installed.
    """
    n = input_tensor.shape[-1] // 2
    a = input_tensor[..., :n].to(torch.float64)
    b = input_tensor[..., n:].to(torch.float64)

    sqrt_2_over_pi = 0.79788456
    tanh_inner = torch.tanh(sqrt_2_over_pi * a * (1.0 + 0.044715 * a**2))
    gelu_a = 0.5 * a * (1.0 + tanh_inner)
    return gelu_a * b


if TE_OP is None:
    # TransformerEngine is not installed: bind the fp64 torch reference in
    # place of the TE entry point so the accuracy test runs instead of skipping.
    TE_OP = _reference_geglu


@pytest.mark.geglu
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_geglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if flag_gems.vendor_name == "kunlunxin":
        ref_out = TE_OP(input_tensor, None, None, TORCH_TO_TE_DTYPE[dtype])
    else:
        ref_out = TE_OP(input_tensor, None)
    ref_out = utils.to_reference(ref_out)

    res_out = flag_gems.geglu(input_tensor)
    utils.gems_assert_close(res_out, ref_out, dtype)
