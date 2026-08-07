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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

# ``_assert_tensor_metadata`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._assert_tensor_metadata`` and
# ``-m _assert_tensor_metadata`` both work.
setattr(
    pytest.mark,
    "_assert_tensor_metadata",
    MarkDecorator(
        Mark("_assert_tensor_metadata", (), {}, _ispytest=True), _ispytest=True
    ),
)


@pytest.mark._assert_tensor_metadata
@pytest.mark.parametrize("shape", [(2, 3), (4, 5, 6), (128,)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_accuracy__assert_tensor_metadata(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Matching metadata: should return None both in torch and gems.
    ref_out = torch._assert_tensor_metadata(
        inp, size=list(shape), stride=list(inp.stride()), dtype=dtype
    )
    with flag_gems.use_gems():
        res_out = torch._assert_tensor_metadata(
            inp, size=list(shape), stride=list(inp.stride()), dtype=dtype
        )
    assert ref_out is None
    assert res_out is None


@pytest.mark._assert_tensor_metadata
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test__assert_tensor_metadata_size_mismatch(dtype):
    inp = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch._assert_tensor_metadata(inp, size=[5, 5])


@pytest.mark._assert_tensor_metadata
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test__assert_tensor_metadata_dtype_mismatch(dtype):
    other = torch.float16 if dtype == torch.float32 else torch.float32
    inp = torch.randn((3, 4), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch._assert_tensor_metadata(inp, dtype=other)


@pytest.mark._assert_tensor_metadata
def test__assert_tensor_metadata_stride_mismatch():
    inp = torch.randn((3, 4), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch._assert_tensor_metadata(inp, stride=[1, 1])


@pytest.mark._assert_tensor_metadata
def test__assert_tensor_metadata_none_args():
    # All optional args are None: nothing is checked, returns None.
    inp = torch.randn((3, 4), dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch._assert_tensor_metadata(inp)
    assert res_out is None
