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


@pytest.mark.maximum
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_maximum(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.maximum(ref_inp1, ref_inp2)
    res_out = flag_gems.maximum(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.maximum
@pytest.mark.parametrize("cuda_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cpu_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("cpu_operand", ["self", "other"])
def test_maximum_cpu_scalar_operand(cuda_dtype, cpu_dtype, cpu_operand):
    x = torch.linspace(
        0.25, 4.0, 64, device=flag_gems.device, dtype=cuda_dtype
    ).reshape(8, 8)
    scalar = torch.tensor(1.5, device="cpu", dtype=cpu_dtype)
    inp1, inp2 = (scalar, x) if cpu_operand == "self" else (x, scalar)

    ref_out = torch.maximum(utils.to_reference(inp1), utils.to_reference(inp2))
    res_out = flag_gems.maximum(inp1, inp2)

    assert res_out.device == x.device
    assert res_out.dtype == ref_out.dtype == cuda_dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.maximum
@pytest.mark.skipif(
    not utils.fp64_is_supported, reason="float64 is not supported on this device"
)
@pytest.mark.parametrize(
    "value, scalar_dtype",
    [
        (1.0 + 2.0**-52, torch.float64),
        (2**24 + 1, torch.int64),
        (2**53 + 2**29, torch.int64),
    ],
    ids=["fp64", "int64-2^24", "int64-2^53"],
)
@pytest.mark.parametrize("cpu_operand", ["self", "other"])
def test_maximum_cpu_scalar_precision(value, scalar_dtype, cpu_operand):
    inp = torch.ones((2, 3), dtype=torch.float64, device=flag_gems.device)
    # None of these values survives a trip through fp32.
    scalar = torch.tensor(value, dtype=scalar_dtype)
    inp1, inp2 = (scalar, inp) if cpu_operand == "self" else (inp, scalar)

    ref_out = torch.maximum(utils.to_reference(inp1), utils.to_reference(inp2))
    res_out = flag_gems.maximum(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.maximum
@pytest.mark.parametrize("cpu_operand", ["self", "other"])
def test_maximum_rejects_non_scalar_cpu_operand(cpu_operand):
    inp = torch.randn((2, 19, 7), device=flag_gems.device)
    cpu_tensor = torch.randn(7)
    inp1, inp2 = (cpu_tensor, inp) if cpu_operand == "self" else (inp, cpu_tensor)

    with pytest.raises(RuntimeError, match="same device"):
        torch.maximum(inp1, inp2)
    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.maximum(inp1, inp2)
