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

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="CUDA-only op; no CPU reference implementation"
)


# (component sizes along the ragged dim, dense dim) configurations.
NESTED_CONFIGS = [
    ([5, 7, 4], 3),
    ([16, 8, 12, 20], 8),
    ([64, 128, 256], 16),
    ([3, 5, 2, 7, 4], 6),
]


@pytest.mark.nested_select_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("sizes, dense_dim", NESTED_CONFIGS)
@pytest.mark.parametrize("index", [0, 1, -1])
def test_nested_select_backward(dtype, sizes, dense_dim, index):
    comps = [
        torch.randn(s, dense_dim, dtype=dtype, device=flag_gems.device) for s in sizes
    ]
    self_nt = torch.nested.nested_tensor(
        comps, layout=torch.jagged, device=flag_gems.device
    )

    nidx = index if index >= 0 else len(sizes) + index
    grad = torch.randn(sizes[nidx], dense_dim, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten._nested_select_backward(
        utils.to_reference(grad), self_nt, 0, index
    )
    res_out = flag_gems._nested_select_backward(grad, self_nt, 0, index)

    assert res_out.is_nested
    assert torch.equal(res_out.offsets(), ref_out.offsets())
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)


@pytest.mark.nested_select_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_select_backward_1d_components(dtype):
    sizes = [5, 7, 4, 3]
    comps = [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in sizes]
    self_nt = torch.nested.nested_tensor(
        comps, layout=torch.jagged, device=flag_gems.device
    )

    grad = torch.randn(sizes[1], dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten._nested_select_backward(
        utils.to_reference(grad), self_nt, 0, 1
    )
    res_out = flag_gems._nested_select_backward(grad, self_nt, 0, 1)

    assert res_out.is_nested
    assert torch.equal(res_out.offsets(), ref_out.offsets())
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
