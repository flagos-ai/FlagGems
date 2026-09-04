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

FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.FLOAT_DTYPES
SHAPES = [(0,), (17,), (4096,), (4097,), (128, 256), (65536,), (65537,)]
if not cfg.QUICK_MODE:
    SHAPES += [(1024, 1024)]


def _make_args(shape, dtype, threshold=0.5):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < threshold
    grad = torch.randn(int(mask.sum()), dtype=dtype, device=flag_gems.device)
    return grad, inp, mask


def _assert_matches_reference(grad, inp, mask):
    ref = torch.ops.aten.masked_select_backward.default(
        utils.to_reference(grad),
        utils.to_reference(inp),
        utils.to_reference(mask),
    )
    result = flag_gems.masked_select_backward(grad, inp, mask)
    utils.gems_assert_equal(result, ref)
    assert result.stride() == ref.stride()


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_select_backward(shape, dtype):
    _assert_matches_reference(*_make_args(shape, dtype))


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_accuracy_masked_select_backward_mask_extremes(threshold):
    grad, inp, mask = _make_args((8192,), torch.float32, threshold)
    # ATen accepts a source longer than the number of selected mask entries.
    grad = torch.cat((grad, torch.randn(3, device=flag_gems.device)))
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int32, torch.float64, torch.complex64]
)
def test_accuracy_masked_select_backward_extended_dtypes(dtype):
    shape = (73, 71)
    mask = torch.rand(shape, device=flag_gems.device) < 0.4
    if dtype == torch.bool:
        inp = torch.rand(shape, device=flag_gems.device) < 0.5
        grad = torch.rand(int(mask.sum()), device=flag_gems.device) < 0.5
    elif dtype.is_complex:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        grad = torch.randn(int(mask.sum()), dtype=dtype, device=flag_gems.device)
    elif dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        grad = torch.randn(int(mask.sum()), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-5, 6, shape, dtype=dtype, device=flag_gems.device)
        grad = torch.randint(
            -5, 6, (int(mask.sum()),), dtype=dtype, device=flag_gems.device
        )
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("case", ["input", "mask"])
def test_accuracy_masked_select_backward_broadcast(case):
    inp_shape, mask_shape = ((1, 7), (5, 7)) if case == "input" else ((5, 7), (1, 7))
    inp = torch.randn(inp_shape, device=flag_gems.device)
    mask = torch.rand(mask_shape, device=flag_gems.device) < 0.5
    count = int(torch.broadcast_tensors(inp, mask)[1].sum())
    grad = torch.randn(count, device=flag_gems.device)
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("layout", ["transpose", "channels_last"])
def test_accuracy_masked_select_backward_preserves_layout(layout):
    if layout == "transpose":
        inp = torch.randn((11, 7), device=flag_gems.device).T
    else:
        inp = torch.randn((2, 3, 5, 7), device=flag_gems.device).contiguous(
            memory_format=torch.channels_last
        )
    mask = (torch.rand_like(inp) < 0.5).contiguous()
    grad = torch.randn(int(mask.sum()), device=flag_gems.device)
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
def test_accuracy_masked_select_backward_noncontiguous_inputs():
    inp = torch.randn((67, 65), device=flag_gems.device)
    mask = (torch.rand((65, 67), device=flag_gems.device) < 0.5).T
    grad_storage = torch.randn(int(mask.sum()) * 2, device=flag_gems.device)
    grad = grad_storage[::2]
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
def test_masked_select_backward_errors():
    inp = torch.randn((8,), device=flag_gems.device)
    grad = torch.randn((4,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.masked_select_backward(grad, inp, torch.ones_like(inp))
    with pytest.raises(RuntimeError):
        flag_gems.masked_select_backward(
            grad.to(torch.float64), inp, torch.ones_like(inp, dtype=torch.bool)
        )
