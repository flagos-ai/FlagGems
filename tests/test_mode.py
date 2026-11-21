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
from flag_gems.ops.mode import mode_backward

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


MTHREADS_NATIVE_MODE_SKIP = pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason=(
        "MThreads native torch.mode raises 'MUSA error: misaligned address' "
        "during repeated benchmarking for FP16/BF16/INT16 at shapes "
        "(64, 64), (256, 256), and (1024, 1024); skip these dtypes "
        "pending a native backend fix."
    ),
)


def _mode_dtype_params(dtypes):
    return [
        (
            pytest.param(dtype, marks=MTHREADS_NATIVE_MODE_SKIP)
            if dtype in (torch.float16, torch.bfloat16, torch.int16)
            else dtype
        )
        for dtype in dtypes
    ]


def _assert_mode_matches(inp, dim, keepdim):
    normalized_dim = dim % inp.ndim
    ref_inp = inp.cpu()
    ref_out_value, _ = torch.mode(ref_inp, dim=dim, keepdim=keepdim)
    res_out_value, res_out_index = flag_gems.mode(inp, dim=dim, keepdim=keepdim)

    assert res_out_value.dtype == inp.dtype
    assert res_out_index.dtype == torch.int64
    assert res_out_value.shape == ref_out_value.shape
    assert res_out_index.shape == ref_out_value.shape
    assert torch.all(
        (res_out_index.cpu() >= 0) & (res_out_index.cpu() < inp.shape[dim])
    )
    utils.gems_assert_equal(res_out_value.cpu(), ref_out_value)
    gather_idx = res_out_index.cpu().reshape(
        list(ref_inp.shape[:normalized_dim])
        + [1]
        + list(ref_inp.shape[normalized_dim + 1 :])
    )
    values_at_index = ref_inp.gather(normalized_dim, gather_idx).reshape(
        res_out_index.shape
    )
    utils.gems_assert_equal(values_at_index, ref_out_value)


@pytest.mark.mode
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize(
    "dtype",
    _mode_dtype_params(FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]),
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mode(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        low, high = (
            (0, 256)
            if dtype == torch.uint8
            else ((-128, 128) if dtype == torch.int8 else (-100, 100))
        )
        inp = torch.randint(low, high, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

    _assert_mode_matches(inp, dim, keepdim)


@pytest.mark.mode
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int8, torch.uint8])
@pytest.mark.parametrize(
    "data, dim, keepdim",
    [
        (
            [
                [1, 2, 2, 3],
                [4, 4, 4, 5],
                [6, 7, 7, 7],
            ],
            1,
            False,
        ),
        (
            [
                [1, 5, 2],
                [1, 5, 3],
                [4, 6, 3],
                [1, 7, 3],
            ],
            0,
            True,
        ),
    ],
)
def test_mode_repeated_values(data, dim, keepdim, dtype):
    inp = torch.tensor(data, dtype=dtype, device=flag_gems.device)

    _assert_mode_matches(inp, dim, keepdim)


@pytest.mark.mode
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("case", ["unique", "tie", "equal"])
@pytest.mark.parametrize("repeat_factor", [1, 3, 103, 1639])
def test_mode_byte_boundaries(dtype, dim, keepdim, case, repeat_factor):
    low, high = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    middle = -1 if dtype == torch.int8 else 128
    if case == "unique":
        data = [[low, high, high, middle, high], [middle, low, low, high, low]]
    elif case == "tie":
        data = [[high, low, high, low, middle], [middle, high, middle, high, low]]
    else:
        data = [[low] * 5, [high] * 5]
    # Preserve the unique/tied frequencies while crossing reduction block sizes.
    inp = torch.tensor(data, dtype=dtype).repeat(1, repeat_factor).to(flag_gems.device)
    if dim == 0:
        inp = inp.t()
    _assert_mode_matches(inp, dim, keepdim)


@pytest.mark.mode
@pytest.mark.parametrize(
    "dtype",
    _mode_dtype_params(
        [
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.int16,
            torch.int32,
            torch.int64,
        ]
    ),
)
@pytest.mark.parametrize("width", [63, 129, 1025, 4097])
@pytest.mark.parametrize("dim", [0, -1])
def test_mode_run_boundaries(dtype, width, dim):
    # Tied runs, extrema, padding and runs spanning the scan tile boundary.
    if dtype.is_floating_point:
        low, high = -float("inf"), float("inf")
    else:
        low, high = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    data = torch.full((3, width), high, dtype=dtype)
    data[0, : width // 2] = low
    data[1, ::2] = low
    data[2, :] = low
    inp = data.to(flag_gems.device)
    if dim == 0:
        inp = inp.t()
    _assert_mode_matches(inp, dim, False)


@pytest.mark.mode
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32])
@pytest.mark.parametrize("width", [129, 513, 4097])
def test_mode_adjacent_values(dtype, width):
    generator = torch.Generator().manual_seed(42)
    if dtype == torch.int32:
        inp = torch.randint(
            -(2**31), 2**31, (7, width), dtype=dtype, generator=generator
        )
        low, high = 2**24, 2**24 + 1
    else:
        inp = torch.randn((7, width), generator=generator)
        low = 1.0
        high = torch.nextafter(torch.tensor(low), torch.tensor(float("inf"))).item()
    # Adjacent keys must remain distinct, including integers beyond FP32 precision.
    inp[0, ::2] = low
    inp[0, 1::2] = high
    inp[1] = high
    inp[2, ::2] = 0
    inp[2, 1::2] = -0.0
    _assert_mode_matches(inp.to(flag_gems.device), -1, False)


@pytest.mark.mode
@pytest.mark.parametrize(
    "dtype",
    _mode_dtype_params(
        [torch.float16, torch.bfloat16, torch.int16, torch.int8, torch.uint8]
    ),
)
@pytest.mark.parametrize("dim", [0, -1])
def test_mode_many_rows(dtype, dim):
    # Independent rows must not exhaust the launch grid for short reductions.
    data = torch.arange(4097, dtype=torch.int32).remainder(17).to(dtype)
    inp = data[:, None].expand(-1, 65).contiguous()
    inp[:, -1] = -1
    if dim == 0:
        inp = inp.t()
    _assert_mode_matches(inp.to(flag_gems.device), dim, True)


@pytest.mark.mode
@pytest.mark.parametrize("rows", [4095, 4096, 4097, 8195])
@pytest.mark.parametrize("dim", [0, -1])
def test_mode_histogram_launch_boundary(rows, dim):
    # Cross the legacy 16-bucket grid bound and exercise the INT32 radix path.
    inp = torch.tensor([3, 1, 3, 1, 2], dtype=torch.int32).repeat(rows, 13)
    if dim == 0:
        inp = inp.t()
    _assert_mode_matches(inp.to(flag_gems.device), dim, False)


@pytest.mark.mode
@pytest.mark.parametrize("rows", [255, 256, 257, 513, 65536])
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
def test_mode_byte_histogram_launch_boundary(rows, dtype):
    inp = torch.tensor([3, 1, 3, 1, 2], dtype=dtype).repeat(rows, 1)
    _assert_mode_matches(inp.to(flag_gems.device), -1, False)


@pytest.mark.mode
@pytest.mark.parametrize("rows", [65534, 65535, 65536, 98307])
def test_mode_small_rows_launch_boundary(rows):
    inp = torch.tensor([3, 1, 3, 1, 2], dtype=torch.float32).repeat(rows, 1)
    _assert_mode_matches(inp.to(flag_gems.device), -1, False)


@pytest.mark.mode
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend backend override")
@pytest.mark.parametrize("rows", [255, 256, 257])
def test_mode_histogram16_batch_boundary(rows):
    # Cross the 256-row histogram workspace batch and check the partial tail.
    row_values = torch.arange(rows, dtype=torch.int16).remainder(17)
    inp = row_values[:, None].expand(-1, 257).contiguous()
    inp[:, -1] = -1
    _assert_mode_matches(inp.to(flag_gems.device), -1, False)


@pytest.mark.mode
@pytest.mark.parametrize("shape", [(4096, 4096), (1024, 65536)])
def test_mode_large_radix_launch_boundary(shape):
    # Keep large logical tile counts on the INT32 radix path on Ascend.
    row = torch.arange(shape[1], dtype=torch.int32) % 17
    inp = row.expand(shape).contiguous().to(flag_gems.device)
    _assert_mode_matches(inp, -1, False)


@pytest.mark.mode
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend backend override")
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int8, torch.uint8])
def test_mode_ascend_dispatch(dtype):
    assert flag_gems.mode.__module__.endswith("_ascend.ops.mode")
    inp = torch.tensor([[3, 1, 3, 1, 2]], dtype=dtype, device=flag_gems.device)
    _assert_mode_matches(inp, -1, True)


# ---------------------------------------------------------------------------
# Backward
#
# `torch.mode` has no independent ATen backward schema: `derivatives.yaml`
# generates the values gradient as `value_selecting_reduction_backward`, which
# scatters the upstream gradient onto the int64 mode indices the forward
# returned and zeroes the rest. `indices` is an integer output and carries no
# gradient. The tests below cover that contract through the public
# `flag_gems.mode` autograd entry and through the `mode_backward` helper, with
# the `mode_backward` marker.
# ---------------------------------------------------------------------------


def _distinct_mode_input(shape, dim, dtype):
    """Strictly increasing integers along ``dim``.

    Every value is distinct and exactly representable in FP16, FP32 and BF16 up
    to the ranks used below, so the mode is the minimum and its index is unique.
    That keeps the forward's tie-breaking out of the comparison.
    """
    n_reduce = shape[dim]
    offset = torch.arange(n_reduce, device=flag_gems.device).reshape(
        [1] * dim + [n_reduce] + [1] * (len(shape) - dim - 1)
    )
    return offset.expand(shape).contiguous().to(dtype)


def _distinct_shape_for(dtype):
    """Reduction extent whose integer values are still exact in ``dtype``."""
    if dtype == torch.bfloat16:
        return (128, 256)
    if dtype == torch.float16:
        return (512, 1024)
    return (1024, 4096)


QUICK_BACKWARD_SHAPES = [(1, 2), (64, 64)]


@pytest.mark.mode_backward
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mode_backward_autograd(dtype, dim, keepdim):
    """Autograd through `flag_gems.mode` matches native `torch.mode`."""
    shape = (
        _distinct_shape_for(dtype) if not cfg.QUICK_MODE else QUICK_BACKWARD_SHAPES[1]
    )
    if dim >= len(shape):
        pytest.skip("dim outside this rank")
    inp = _distinct_mode_input(shape, dim, dtype)
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    inp = inp.requires_grad_(True)

    res_values, res_indices = flag_gems.mode(inp, dim=dim, keepdim=keepdim)
    ref_values, ref_indices = torch.mode(ref_inp, dim=dim, keepdim=keepdim)

    assert not res_indices.requires_grad
    assert res_indices.dtype == torch.int64
    utils.gems_assert_close(res_values, ref_values, dtype)
    utils.gems_assert_equal(res_indices, ref_indices)

    out_grad = torch.randn(res_values.shape, dtype=dtype, device=flag_gems.device)
    (res_grad,) = torch.autograd.grad(res_values, inp, out_grad)
    (ref_grad,) = torch.autograd.grad(
        ref_values, ref_inp, utils.to_reference(out_grad, True)
    )

    assert res_grad.shape == inp.shape
    assert res_grad.dtype == inp.dtype
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.mode_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_mode_backward_matches_value_selecting_helper(dtype, dim, keepdim):
    """`mode_backward` is exactly `aten::value_selecting_reduction_backward`."""
    shape = (9, 200, 2560) if not cfg.QUICK_MODE else (2, 3, 5)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    values, indices = flag_gems.mode(inp, dim=dim, keepdim=keepdim)
    out_grad = torch.randn_like(values)

    res_grad = mode_backward(out_grad, dim, indices, list(shape), keepdim)
    ref_grad = torch.ops.aten.value_selecting_reduction_backward(
        out_grad, dim, indices, list(shape), keepdim
    )
    assert res_grad.shape == inp.shape
    assert res_grad.dtype == inp.dtype
    assert res_grad.stride() == ref_grad.stride()
    assert torch.equal(res_grad, ref_grad)


@pytest.mark.mode_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
def test_mode_backward_gradient_attribution(dtype, keepdim):
    """Only the positions the forward selected receive gradient."""
    shape = (300, 257)
    inp = _distinct_mode_input(shape, 1, dtype).requires_grad_(True)
    values, indices = flag_gems.mode(inp, dim=1, keepdim=keepdim)
    out_grad = torch.randn_like(values)
    (grad,) = torch.autograd.grad(values, inp, out_grad)

    assert torch.count_nonzero(grad) == indices.numel()
    index = indices if keepdim else indices.unsqueeze(1)
    upstream = out_grad if keepdim else out_grad.unsqueeze(1)
    expected = torch.zeros(shape, dtype=dtype, device=flag_gems.device).scatter(
        1, index, upstream
    )
    assert torch.equal(grad, expected)


@pytest.mark.mode_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mode_backward_repeated_values(dtype):
    """Repeated mode values scatter onto the index the forward reported."""
    data = [[3, 3, 3, 0, 0], [0, 3, 3, 3, 0], [0, 1, 3, 3, 3], [1, 2, 3, 4, 5]]
    inp = torch.tensor(data, dtype=dtype, device=flag_gems.device)
    values, indices = flag_gems.mode(inp, dim=1)
    out_grad = torch.randn_like(values)
    grad = mode_backward(out_grad, 1, indices, list(inp.shape), False)

    # The candidate gradient places `out_grad` exactly at the reported index.
    expected = torch.zeros_like(inp).scatter(
        1, indices.unsqueeze(1), out_grad.unsqueeze(1)
    )
    assert torch.equal(grad, expected)
    # the reported index must actually hold a mode value
    gathered = inp.gather(1, indices.unsqueeze(1)).squeeze(1)
    assert torch.equal(gathered, values)
    utils.gems_assert_close(
        values, torch.mode(utils.to_reference(inp, True), dim=1).values, dtype
    )


@pytest.mark.mode_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mode_backward_expanded_grad(dtype):
    """A broadcast upstream gradient is expanded before the scatter."""
    inp = _distinct_mode_input((7, 13), 1, dtype).requires_grad_(True)
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    res_values, _ = flag_gems.mode(inp, dim=1)
    ref_values, _ = torch.mode(ref_inp, dim=1)
    out_grad = torch.randn(1, dtype=dtype, device=flag_gems.device).expand(7)

    (res_grad,) = torch.autograd.grad(res_values, inp, out_grad)
    (ref_grad,) = torch.autograd.grad(
        ref_values, ref_inp, utils.to_reference(out_grad, True)
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.mode_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mode_backward_dim0_noncontiguous(dtype):
    """`dim=0` on a transposed input writes the standard-layout gradient."""
    # Distinct values along the reduced axis keep the mode index unique while
    # the transposed view keeps the input non-contiguous.
    inp = _distinct_mode_input((96, 64), 1, dtype).t().requires_grad_(True)
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    res_values, _ = flag_gems.mode(inp, dim=0)
    ref_values, _ = torch.mode(ref_inp, dim=0)
    out_grad = torch.randn(res_values.shape, dtype=dtype, device=flag_gems.device)
    (res_grad,) = torch.autograd.grad(res_values, inp, out_grad)
    (ref_grad,) = torch.autograd.grad(
        ref_values, ref_inp, utils.to_reference(out_grad, True)
    )
    assert res_grad.is_contiguous()
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.mode_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mode_backward_empty_rows(dtype):
    """Zero outer elements produce an empty gradient of the input shape."""
    inp = torch.randn((0, 5), dtype=dtype, device=flag_gems.device).requires_grad_(True)
    values, indices = flag_gems.mode(inp, dim=1)
    grad = mode_backward(torch.randn_like(values), 1, indices, list(inp.shape), False)
    assert grad.shape == inp.shape
    assert grad.dtype == dtype
    assert grad.numel() == 0
