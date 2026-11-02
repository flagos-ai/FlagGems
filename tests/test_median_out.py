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

# A mix of parities and ranks: for an even element count ``median`` returns the
# lower of the two middle elements, and the flattened case (4, 8, 16, 8) is the
# one the fast path was sized for.
MEDIAN_OUT_SHAPES = [(1,), (7, 5), (3, 17), (2, 3, 5), (4, 8, 16, 8)]


@pytest.mark.median_out
@pytest.mark.parametrize("shape", MEDIAN_OUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_median_out(shape, dtype):
    if dtype in utils.ALL_INT_DTYPES:
        # Narrow range on purpose: integer inputs collide often, which is where
        # a wrong lower-middle rule shows up.
        inp = torch.randint(-8, 9, shape).to(dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randn(shape, device=flag_gems.device).to(dtype)
    ref_inp = utils.to_reference(inp)

    out = torch.empty((), dtype=dtype, device=flag_gems.device)
    ref_out = torch.empty((), dtype=dtype, device=ref_inp.device)

    ref_result = torch.ops.aten.median.out(ref_inp, out=ref_out)
    result = flag_gems.median_out(inp, out=out)

    assert result is out
    utils.gems_assert_equal(result, ref_result)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.median_out
@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 2.0, 3.0, 4.0], 2.0),
        ([4.0, 3.0, 2.0, 1.0], 2.0),
        ([2.0, 2.0, 3.0, 4.0], 2.0),
        ([1.0, 2.0, 2.0, 2.0], 2.0),
        ([5.0, 5.0, 5.0, 5.0], 5.0),
    ],
)
def test_median_out_even_count_picks_lower_middle(values, expected):
    inp = torch.tensor(values, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty((), dtype=inp.dtype, device=flag_gems.device)

    result = flag_gems.median_out(inp, out=out)

    assert result is out
    # Build the expectation on the reference device: gems_assert_equal expects
    # `ref` to live there when running with --ref=cpu.
    ref = torch.full_like(utils.to_reference(out), expected)
    utils.gems_assert_equal(out, ref)


@pytest.mark.median_out
def test_median_out_rejects_bad_out_dtype():
    inp = torch.randn((7,), dtype=torch.float32, device=flag_gems.device)
    bad_out = torch.empty((), dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.median_out(inp, out=bad_out)


@pytest.mark.median_out
def test_median_out_rejects_cpu_out():
    if torch.device(flag_gems.device).type != "cuda":
        pytest.skip("mixed-device out= only applies when the accelerator is not cpu")

    inp = torch.randn((7,), dtype=torch.float32, device=flag_gems.device)
    cpu_out = torch.empty((), dtype=inp.dtype, device="cpu")

    with pytest.raises(RuntimeError):
        flag_gems.median_out(inp, out=cpu_out)
