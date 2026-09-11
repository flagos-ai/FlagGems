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

    TE_OP = getattr(tex, "swiglu", None)
except ImportError:
    TE_OP = None


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


@pytest.mark.swiglu
@pytest.mark.parametrize("shape", VALID_POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(TE_OP is None, reason="'swiglu' not found in TransformerEngine")
def test_swiglu(shape: tuple[int, ...], dtype: torch.dtype):
    torch.manual_seed(42)
    device = flag_gems.device

    input_tensor = generate_input(shape, dtype, device)

    te_forward = TE_OP(input_tensor, quantizer=None).to(device)
    te_forward = utils.to_reference(te_forward)

    with flag_gems.use_gems():
        fg_forward = flag_gems.swiglu(input_tensor, quantizer=None)

    utils.gems_assert_close(fg_forward, te_forward, dtype)
