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

if cfg.QUICK_MODE:
    ROUND_DECIMALS_SHAPES = [(2, 3)]
    ROUND_HALF_SHAPES = [(2, 3)]
    ROUND_DECIMALS = [0]
else:
    ROUND_DECIMALS_SHAPES = [(2, 3), (128, 256), (4, 8, 16)]
    ROUND_HALF_SHAPES = [(2, 3), (4, 8)]
    ROUND_DECIMALS = [-2, -1, 0, 1, 2]


@pytest.mark.round
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.round(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.round(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.round_(ref_inp)
    with flag_gems.use_gems():
        res_out = inp.round_()

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="round-half-to-even boundary fix is specific to the Ascend kernel",
)
@pytest.mark.round_
@pytest.mark.parametrize(
    "value",
    # Half-integers around +-2**22 and +-2**23: fp32 still represents these
    # exactly, so round-half-to-even must apply (regression test for a
    # magic-constant implementation that returned them unchanged).
    [
        2**22 - 0.5,
        2**22 + 0.5,
        2**22 + 1.5,
        2**23 - 0.5,
        -(2**22) - 0.5,
        -(2**22) - 1.5,
        -(2**23) + 0.5,
    ],
)
def test_round__large_half_integers(value):
    from flag_gems.runtime.backend._ascend.ops import round_ as ascend_round_

    inp = torch.tensor([value], dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.round_(ref_inp)
    res_out = ascend_round_(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="integral fast path is specific to the Ascend kernel",
)
@pytest.mark.round_
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_round__integral_unchanged(dtype):
    # round_ must leave integral tensors bit-identical; routing them through
    # fp32 would corrupt values above 2**24 (checked for the wider dtypes,
    # where such values are representable).
    values = [0, 1, -1]
    if torch.iinfo(dtype).max > 2**24 + 3:
        values += [2**24 + 3, -(2**24) - 3]
    from flag_gems.runtime.backend._ascend.ops import round_ as ascend_round_

    inp = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    expected = utils.to_reference(inp.clone())

    res_out = ascend_round_(inp)

    utils.gems_assert_equal(res_out, expected)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="empty-input handling is specific to the Ascend kernel",
)
@pytest.mark.round_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round__empty(dtype):
    from flag_gems.runtime.backend._ascend.ops import round_ as ascend_round_

    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.round_(ref_inp)
    res_out = ascend_round_(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="contiguity guard is specific to the Ascend kernel",
)
@pytest.mark.round_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round__non_contiguous_rejected(dtype):
    # A strided view would be traversed linearly by the kernel and corrupt
    # neighbouring elements, so it must be rejected rather than silently wrong.
    from flag_gems.runtime.backend._ascend.ops import round_ as ascend_round_

    base = torch.randn(16, dtype=dtype, device=flag_gems.device)
    view = base[::2]
    assert not view.is_contiguous()

    with pytest.raises(ValueError):
        ascend_round_(view)


@pytest.mark.round_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.round(ref_inp, out=ref_out)
    with flag_gems.use_gems():
        torch.round(inp, out=out)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.round
@pytest.mark.parametrize("shape", ROUND_DECIMALS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("decimals", ROUND_DECIMALS)
def test_round_decimals(shape, dtype, decimals):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 100

    # When demical≠0 and input is float16/bfloat16,
    # compute result is difference between CUDA and CPU in Pytorch itself because of precision error
    # so compare the result between FlagGems version and Pytorch CUDA version
    ref_out = torch.round(inp, decimals=decimals)
    ref_out = ref_out.to("cpu")

    with flag_gems.use_gems():
        res_out = torch.round(inp, decimals=decimals)
        res_out = res_out.to("cpu")

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round
@pytest.mark.parametrize("shape", ROUND_HALF_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_half_to_even(shape, dtype):
    # Test round half to even: 2.5->2, 3.5->4, -2.5->-2, -3.5->-4
    inp = torch.tensor(
        [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.round(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.round(inp)

    utils.gems_assert_equal(res_out, ref_out)
