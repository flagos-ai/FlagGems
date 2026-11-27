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

    TE_OP = getattr(tex, "dswiglu", None)
except ImportError:
    TE_OP = None
except RuntimeError as te_import_error:
    # The installed transformer-engine here is only the PyPI metapackage, so
    # transformer_engine.common aborts while importing and turns the whole
    # module uncollectable.  The TE reference op has no GCU implementation
    # (torch aten/TE not supported on this backend), so skip the module on
    # enflame with the real cause instead of failing collection for every
    # operator of the run.
    if flag_gems.vendor_name == "enflame":
        pytest.skip(
            "enflame GCU: transformer-engine metapackage has no framework "
            f"extension ({te_import_error}); the TE reference op is not "
            "supported on this backend (torch aten/TE not supported on GCU)",
            allow_module_level=True,
        )
    raise


def generate_input(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    return torch.randn(shape, dtype=dtype, device=device).contiguous()


def filter_valid_shapes(shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    valid_shapes = []
    for shape in shapes:
        if not shape:
            continue
        if shape[-1] % 2 == 0:
            valid_shapes.append(shape)
    return valid_shapes


VALID_POINTWISE_SHAPES = filter_valid_shapes(utils.SWIGLU_SPECIAL_SHAPES)


def _reference_dswiglu(
    grad_output: torch.Tensor, input_tensor: torch.Tensor, quantizer=None
) -> torch.Tensor:
    """High-precision (fp64) reference for dSwiGLU backward.

    ``dSwiGLU(a, b)`` with ``a = input[..., :N]``, ``b = input[..., N:]`` has
    ``grad_a = grad_output * b * d_silu(a)`` and ``grad_b = grad_output * silu(a)``
    with ``silu(x) = x * sigmoid(x)`` and
    ``d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))``.

    Takes the same arguments as TransformerEngine's entry point so it can stand
    in for it when TE is not installed.
    """
    n = input_tensor.shape[-1] // 2
    a = input_tensor[..., :n].to(torch.float64)
    b = input_tensor[..., n:].to(torch.float64)
    g = grad_output.to(torch.float64)

    sig = torch.sigmoid(a)
    silu = a * sig
    d_silu = sig + a * sig * (1.0 - sig)

    grad_a = g * b * d_silu
    grad_b = g * silu
    return torch.cat([grad_a, grad_b], dim=-1)


if TE_OP is None:
    # TransformerEngine is not installed: bind the fp64 torch reference in
    # place of the TE entry point so the accuracy test runs instead of skipping.
    TE_OP = _reference_dswiglu


@pytest.mark.dswiglu
@pytest.mark.parametrize("shape", VALID_POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dswiglu(shape: tuple[int, ...], dtype: torch.dtype):
    torch.manual_seed(42)
    device = flag_gems.device

    input_tensor = generate_input(shape, dtype, device)

    grad_shape = list(shape)
    grad_shape[-1] = grad_shape[-1] // 2
    grad_output = generate_input(tuple(grad_shape), dtype, device)

    if flag_gems.vendor_name == "kunlunxin":
        te_grad_input = TE_OP(grad_output, input_tensor).to(device)
    else:
        te_grad_input = TE_OP(grad_output, input_tensor, quantizer=None).to(device)
    te_grad_input = utils.to_reference(te_grad_input)

    fg_grad_input = flag_gems.dswiglu(grad_output, input_tensor, quantizer=None)

    utils.gems_assert_close(fg_grad_input, te_grad_input, dtype)
