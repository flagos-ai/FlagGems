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
BOUNDARY_SHAPES = [(32769,), (262144,), (262145,)]
if not cfg.QUICK_MODE:
    SHAPES += [(1024, 1024)]


def _make_args(shape, dtype, threshold=0.5):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = torch.rand(shape, device=flag_gems.device) < threshold
    grad = torch.randn(int(mask.sum()), dtype=dtype, device=flag_gems.device)
    return grad, inp, mask


def _wide_to_device(value):
    return (
        value.contiguous()
        .view(torch.uint8)
        .to(flag_gems.device)
        .view(value.dtype)
        .reshape(value.shape)
    )


def _reference_input(value):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and utils.TO_CPU
        and value.dtype == torch.float64
    ):
        return (
            value.contiguous()
            .view(torch.uint8)
            .cpu()
            .view(value.dtype)
            .reshape(value.shape)
        )
    return utils.to_reference(value)


def _assert_matches_reference(grad, inp, mask):
    ref = torch.ops.aten.masked_select_backward.default(
        _reference_input(grad),
        _reference_input(inp),
        _reference_input(mask),
    )
    result = flag_gems.masked_select_backward(grad, inp, mask)
    assert result.device == inp.device and result.dtype == inp.dtype
    assert result.shape == ref.shape
    assert result.stride() == ref.stride()
    if (
        flag_gems.vendor_name == "kunlunxin"
        and utils.TO_CPU
        and inp.dtype == torch.float64
    ):
        utils.gems_assert_equal(_reference_input(result), ref)
    else:
        utils.gems_assert_equal(result, ref)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_masked_select_backward(shape, dtype):
    _assert_matches_reference(*_make_args(shape, dtype))


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("shape", BOUNDARY_SHAPES)
def test_accuracy_masked_select_backward_kernel_boundaries(shape):
    _assert_matches_reference(*_make_args(shape, torch.float32))


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("shape", [(32769,), (262145,)])
@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_accuracy_masked_select_backward_mask_extremes(shape, threshold):
    torch.manual_seed(0)
    grad, inp, mask = _make_args(shape, torch.float32, threshold)
    # ATen accepts a source longer than the number of selected mask entries.
    grad = torch.cat((grad, torch.randn(3, device=flag_gems.device)))
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int32, torch.float64, torch.complex64]
)
def test_accuracy_masked_select_backward_extended_dtypes(dtype):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype in (torch.bool, torch.float64, torch.complex64)
        or flag_gems.vendor_name == "ascend"
        and dtype.is_complex
    ) and not utils.TO_CPU:
        pytest.skip("backend native masked_select_backward reference lacks this dtype")
    torch.manual_seed(0)
    shape = (73, 71)
    mask = torch.rand(shape, device=flag_gems.device) < 0.4
    if dtype == torch.bool:
        inp = torch.rand(shape, device=flag_gems.device) < 0.5
        grad = torch.rand(int(mask.sum()), device=flag_gems.device) < 0.5
    elif dtype.is_complex:
        factory_device = (
            "cpu" if flag_gems.vendor_name == "ascend" else flag_gems.device
        )
        inp = torch.randn(shape, dtype=dtype, device=factory_device).to(
            flag_gems.device
        )
        grad = torch.randn(int(mask.sum()), dtype=dtype, device=factory_device).to(
            flag_gems.device
        )
    elif flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        inp = _wide_to_device(torch.randn(shape, dtype=dtype, device="cpu"))
        grad = _wide_to_device(torch.randn(int(mask.sum()), dtype=dtype, device="cpu"))
    elif dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        grad = torch.randn(int(mask.sum()), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-5, 6, shape, dtype=dtype, device=flag_gems.device)
        grad = torch.randint(
            -5, 6, (int(mask.sum()),), dtype=dtype, device=flag_gems.device
        )
    assert inp.dtype == grad.dtype == dtype
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("case", ["input", "mask"])
def test_accuracy_masked_select_backward_broadcast(case):
    torch.manual_seed(0)
    inp_shape, mask_shape = ((1, 7), (5, 7)) if case == "input" else ((5, 7), (1, 7))
    inp = torch.randn(inp_shape, device=flag_gems.device)
    mask = torch.rand(mask_shape, device=flag_gems.device) < 0.5
    count = int(torch.broadcast_tensors(inp, mask)[1].sum())
    grad = torch.randn(count, device=flag_gems.device)
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("layout", ["transpose", "channels_last"])
def test_accuracy_masked_select_backward_preserves_layout(layout):
    torch.manual_seed(0)
    if layout == "transpose":
        inp = torch.randn((11, 7), device=flag_gems.device).T
    else:
        factory_device = (
            "cpu" if flag_gems.vendor_name == "ascend" else flag_gems.device
        )
        inp = (
            torch.randn((2, 3, 5, 7), device=factory_device)
            .contiguous(memory_format=torch.channels_last)
            .to(flag_gems.device)
        )
    if flag_gems.vendor_name == "ascend" and layout == "channels_last":
        mask = (torch.rand(inp.shape, device=flag_gems.device) < 0.5).contiguous()
    else:
        mask = (torch.rand_like(inp) < 0.5).contiguous()
    grad = torch.randn(int(mask.sum()), device=flag_gems.device)
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize(
    "input_shape,mask_shape",
    [
        ((), (3, 5)),
        ((2, 1, 5), (3, 1)),
        ((0, 3), (1, 3)),
        ((2, 3), ()),
    ],
)
def test_masked_select_backward_broadcast_metadata(input_shape, mask_shape):
    torch.manual_seed(0)
    inp = torch.randn(input_shape, device=flag_gems.device)
    mask = torch.rand(mask_shape, device=flag_gems.device) < 0.5
    count = int(torch.broadcast_tensors(inp, mask)[1].sum())
    grad = torch.randn(count, device=flag_gems.device)
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
def test_masked_select_backward_broadcast_nonzero_storage_offset():
    torch.manual_seed(0)
    inp = torch.randn((4, 5), device=flag_gems.device)[1:2]
    mask = (torch.rand((5, 4), device=flag_gems.device) < 0.5)[:, 1:].T
    grad = torch.randn(int(mask.sum()), device=flag_gems.device)
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
def test_masked_select_backward_incompatible_broadcast():
    torch.manual_seed(0)
    inp = torch.randn((2, 3), device=flag_gems.device)
    mask = torch.ones((4, 3), dtype=torch.bool, device=flag_gems.device)
    grad = torch.ones(12, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.ops.aten.masked_select_backward(
            utils.to_reference(grad), utils.to_reference(inp), utils.to_reference(mask)
        )
    with pytest.raises(RuntimeError):
        flag_gems.masked_select_backward(grad, inp, mask)


@pytest.mark.masked_select_backward
def test_accuracy_masked_select_backward_noncontiguous_inputs():
    torch.manual_seed(0)
    inp = torch.randn((67, 65), device=flag_gems.device)
    mask = (torch.rand((65, 67), device=flag_gems.device) < 0.5).T
    grad_storage = torch.randn(int(mask.sum()) * 2, device=flag_gems.device)
    grad = grad_storage[::2]
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
def test_masked_select_backward_errors():
    torch.manual_seed(0)
    inp = torch.randn((8,), device=flag_gems.device)
    grad = torch.randn((4,), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.masked_select_backward(grad, inp, torch.ones_like(inp))
    mismatch_dtype = (
        torch.float16
        if flag_gems.vendor_name in ("kunlunxin", "ascend")
        else torch.float64
    )
    bad_grad = grad.to(mismatch_dtype)
    assert bad_grad.dtype == mismatch_dtype and bad_grad.dtype != inp.dtype
    with pytest.raises(RuntimeError):
        flag_gems.masked_select_backward(
            bad_grad, inp, torch.ones_like(inp, dtype=torch.bool)
        )

    mask = torch.ones_like(inp, dtype=torch.bool)
    error = "Number of elements of source < number of ones in mask"
    with pytest.raises(RuntimeError, match=error):
        flag_gems.masked_select_backward(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("size", [17, 65537, 262145])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_masked_select_backward_large_integer_values(size, dtype):
    inp = torch.zeros(size, dtype=dtype, device=flag_gems.device)
    mask = torch.arange(size, device=inp.device) % 3 != 0
    count = int(mask.sum())
    base = 2**25 + 1 if dtype == torch.int32 else 2**60 + 1
    grad = torch.arange(count, dtype=dtype, device=inp.device) + base
    grad[::2] = -grad[::2]
    _assert_matches_reference(grad, inp, mask)


@pytest.mark.masked_select_backward
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_masked_select_backward_conjugated_grad(dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.complex128:
        pytest.skip("Kunlunxin input storage cannot preserve complex128 dtype")
    if flag_gems.vendor_name in ("ascend", "kunlunxin") and not utils.TO_CPU:
        pytest.skip(
            "backend native masked_select_backward reference lacks complex support"
        )
    torch.manual_seed(0)
    factory_device = "cpu" if flag_gems.vendor_name == "ascend" else flag_gems.device
    inp = torch.randn(73, dtype=dtype, device=factory_device).to(flag_gems.device)
    mask = torch.arange(73, device=inp.device) % 2 == 0
    grad = (
        torch.randn(int(mask.sum()), dtype=dtype, device=factory_device)
        .to(inp.device)
        .conj()
    )
    assert grad.is_conj() and grad.is_contiguous()
    _assert_matches_reference(grad, inp, mask)
