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

RANDOM_INT_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

RANDOM_SHAPES = [(1,), (20, 320, 15)]


@pytest.mark.random_
@pytest.mark.parametrize("shape", RANDOM_SHAPES)
@pytest.mark.parametrize("dtype", RANDOM_INT_DTYPES)
def test_random_(shape, dtype):
    x = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = x.random_()
    assert res_out is x
    if dtype == torch.bool:
        assert ((x == 0) | (x == 1)).all()
    else:
        info = torch.iinfo(dtype)
        # Default range is [0, dtype_max), matching aten.
        assert (x >= 0).all()
        assert (x <= info.max).all()


@pytest.mark.random_
@pytest.mark.parametrize("shape", RANDOM_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64],
)
@pytest.mark.parametrize("from_to", [(3, 7), (-10, 10), (100, 200)])
def test_random_from(shape, dtype, from_to):
    from_, to = from_to
    if dtype == torch.int8 and not (-128 <= from_ and to <= 128):
        pytest.skip("range not representable in int8")
    if dtype == torch.uint8 and not (0 <= from_ and to <= 256):
        pytest.skip("range not representable in uint8")
    x = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = x.random_(from_, to)
    assert res_out is x
    assert (x >= from_).all()
    assert (x < to).all()
