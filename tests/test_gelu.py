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
from . import conftest as cfg


def _reference_input(x):
    """Input to build the reference from: CPU, float32.

    In the default (device) reference mode the reference is the backend's own
    implementation, and on Ascend ``gelu(approximate="none")`` is a coarse
    approximation -- measured against fp64 it is off by up to 1.6e-2 for
    bfloat16, 2.0e-3 for float16 and 4.7e-4 for float32, all of which exceed
    this file's 1e-4 atol, so the kernel would be reported as failing while
    being the more accurate of the two. CPU float32 gelu is exact to ~5e-7,
    which keeps the check meaningful in both reference modes.
    """
    return x.detach().cpu().to(torch.float32)


def _place_reference(ref, like):
    """Put a CPU reference where ``gems_assert_close`` expects it."""
    return ref if cfg.TO_CPU else ref.to(like.device)


@pytest.mark.gelu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(res_inp)

    ref_out = torch.nn.functional.gelu(ref_inp, approximate=approximate)
    res_out = flag_gems.gelu(res_inp, approximate=approximate)

    atol = 1e-4
    if flag_gems.vendor_name == "aipu" and dtype == torch.float16:
        atol = 1e-3
    utils.gems_assert_close(
        res_out, _place_reference(ref_out, res_inp), dtype, atol=atol
    )


@pytest.mark.gelu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn_like(res_inp)

    ref_inp = _reference_input(res_inp)
    ref_out = _reference_input(res_out)

    ref_in_grad = torch.ops.aten.gelu_backward(
        ref_out, ref_inp, approximate=approximate
    )
    res_in_grad = flag_gems.gelu_backward(res_out, res_inp, approximate=approximate)

    utils.gems_assert_close(res_in_grad, _place_reference(ref_in_grad, res_inp), dtype)


@pytest.mark.gelu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_(shape, dtype, approximate):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = _reference_input(res_inp)

    ref_out = torch.ops.aten.gelu_.default(ref_inp, approximate=approximate)
    res_out = flag_gems.gelu_(res_inp, approximate=approximate)

    utils.gems_assert_close(res_out, _place_reference(ref_out, res_inp), dtype)
