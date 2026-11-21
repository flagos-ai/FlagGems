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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [1]
    SHAPES = [(1, 256)]
    BACKWARD_SHAPES = [(1, 256)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    SHAPES = [(1, 256), (4096, 256), (200, 2560, 3), (1, 0, 128, 512)]
    BACKWARD_SHAPES = [(1, 256), (4096, 256), (200, 2560, 3)]

random.seed(time.time() // 100)


# Issue 2852: This fails at (1, 2) (200, 40999, 3)
@pytest.mark.softmax
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax(shape, dtype, dim, neg_inf):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    res_out = flag_gems.softmax(inp, dim)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.softmax_out
@pytest.mark.parametrize(
    "shape",
    (
        [(1, 256)]
        if cfg.QUICK_MODE
        else [(1, 256), (4096, 256), (200, 2560, 3), (1, 0, 128, 512)]
    ),
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_out(shape, dtype, dim, neg_inf):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._softmax.out(ref_inp, dim, False, out=ref_out)

    res_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    flag_gems.softmax_out(inp, dim, False, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.softmax_backward_out
@pytest.mark.parametrize(
    "shape", [(1, 256)] if cfg.QUICK_MODE else [(1, 256), (4096, 256), (200, 2560, 3)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_backward_out(shape, dtype, dim, neg_inf):
    if shape[dim] == 1 and flag_gems.vendor_name == "kunlunxin":
        pytest.skip(
            "Issue #2851 _softmax_backward_data short-circuits to zero when reduction dim "
            "is 1, while the Triton kernel computes normally with synthetic inputs, "
            "causing a mismatch that does not reflect a real correctness issue."
        )
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        res_grad = torch.where(res_grad < 0.0, float("-inf"), res_grad)
    res_out = torch.randn_like(res_grad)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.empty(shape, dtype=ref_grad.dtype, device=ref_grad.device)
    torch.ops.aten._softmax_backward_data.out(
        ref_grad, ref_out, dim, ref_grad.dtype, grad_input=ref_in_grad
    )

    res_in_grad = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    flag_gems.softmax_backward_out(
        res_grad, res_out, dim, dtype, grad_input=res_in_grad
    )
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )


@pytest.mark.softmax_backward
@pytest.mark.softmax_backward_data
@pytest.mark.parametrize("shape", BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_backward(shape, dtype, dim, neg_inf):
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        res_grad = torch.where(res_grad < 0.0, float("-inf"), res_grad)
    res_out = torch.randn_like(res_grad)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.ops.aten._softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    res_in_grad = flag_gems.softmax_backward(res_grad, res_out, dim, dtype)

    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )


# ------------------------ backward -------------------------------
# The four core backward shapes put the reduction on dim 0 of a flat tensor or
# on dim 1 of a multi-dimensional one, so the row-dominant ``(1048576,)`` case
# reduces a whole row inside a single CTA.  The extra shapes below pin the
# split-reduction path, the strided path, the empty reduction, a single
# element and the ``half_to_float`` forward.
SPLIT_SHAPES = (
    [(1, 65536)]
    if cfg.QUICK_MODE
    else [(1048576,), (1, 65536), (1, 1048576), (4, 65536), (64, 512, 512)]
)


@pytest.mark.softmax_backward
@pytest.mark.softmax_backward_data
@pytest.mark.parametrize("shape", SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_softmax_backward_split_reduction(shape, dtype):
    """Row-dominant shapes split one reduction row across CTAs."""
    dim = 0 if len(shape) == 1 else 1
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)
    ref_in_grad = torch.ops.aten._softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    res_in_grad = flag_gems.softmax_backward(res_grad, res_out, dim, dtype)

    assert res_in_grad.shape == res_grad.shape
    assert res_in_grad.dtype == dtype
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )


@pytest.mark.softmax_backward
@pytest.mark.softmax_backward_data
@pytest.mark.parametrize("dim", [0, -1, -3])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_softmax_backward_dim_and_empty(dtype, dim):
    """Negative dims follow ATen normalisation; an empty reduction returns empty."""
    shape = (4, 5, 6)
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)
    ref_in_grad = torch.ops.aten._softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    res_in_grad = flag_gems.softmax_backward(res_grad, res_out, dim, dtype)
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )

    # An empty reduction returns an empty tensor instead of dividing by a zero
    # reduction length. The reference stays on the device because a CPU ATen
    # kernel would report a descriptor-level mismatch for the empty result.
    empty = torch.empty((3, 0, 5), dtype=dtype, device=flag_gems.device)
    res_empty = flag_gems.softmax_backward(empty, empty, 1, dtype)
    assert res_empty.shape == (3, 0, 5)
    assert res_empty.dtype == dtype
    assert res_empty.numel() == 0


@pytest.mark.softmax_backward
@pytest.mark.softmax_backward_data
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_softmax_backward_consumes_forward_output(dtype):
    """Backward of a real softmax output equals the ATen saved-output kernel."""
    shape = (64, 512, 512) if not cfg.QUICK_MODE else (64, 256)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    saved_out = flag_gems.softmax(inp, 1)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad = utils.to_reference(grad, True)
    ref_out = utils.to_reference(saved_out, True)
    ref_in_grad = torch.ops.aten._softmax_backward_data(
        ref_grad, ref_out, 1, ref_grad.dtype
    )
    res_in_grad = flag_gems.softmax_backward(grad, saved_out, 1, dtype)

    assert res_in_grad.shape == inp.shape
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[1], equal_nan=True
    )


@pytest.mark.softmax_backward
@pytest.mark.softmax_backward_data
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_softmax_backward_half_to_float(dtype):
    """A ``half_to_float`` forward keeps the input dtype for the gradient."""
    if dtype != torch.float16:
        pytest.skip(
            "half_to_float widening is only accepted by ATen for float16 input_dtype"
        )
    shape = (32, 512)
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    # Device-to-device on purpose: the ATen backward rejects a float32
    # grad_output with a bfloat16 `input_dtype` ("expected input and grad types
    # to match"), so this mixed descriptor cannot be expressed through a
    # CPU reference and is checked against the native kernel directly.
    ref_in_grad = torch.ops.aten._softmax_backward_data(
        res_grad.float(), res_out, 1, dtype
    )
    res_in_grad = flag_gems.softmax_backward(res_grad, res_out, 1, dtype)

    assert res_in_grad.dtype == dtype
    # Keep both sides on the device: `gems_assert_close` compares descriptors
    # too, and `--ref cpu` cannot host the float32 grad_output / half
    # input_dtype combination this test is about.
    torch.testing.assert_close(
        res_in_grad, ref_in_grad, atol=1e-3, rtol=1e-2, equal_nan=True
    )


@pytest.mark.softmax_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_softmax_backward_kernel_attribution(dtype):
    """The row-dominant core shape must run the split kernels, not one CTA/row."""
    from torch.profiler import ProfilerActivity, profile

    shape = (1048576,)
    saved_out = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.softmax_backward(grad, saved_out, 0, dtype)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = {
        event.key for event in prof.key_averages() if event.device_time_total > 0
    }
    assert [
        name for name in launches if "softmax_backward_split_reduce" in name
    ], launches
    assert [
        name for name in launches if "softmax_backward_split_apply" in name
    ], launches
    # The one-CTA-per-row fallback must not run for this shape.
    assert not [
        name for name in launches if name.endswith("softmax_backward_kernel_inner")
    ], launches


@pytest.mark.softmax_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_softmax_backward_out_and_strided_grad(dtype):
    """`out=` keeps the caller buffer and a strided grad_output is accepted."""
    shape = (64, 512, 512)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # A real softmax output keeps every element in [0, 1], which bounds the
    # rounding difference between the float32 accumulation and ATen.
    saved_out = torch.nn.functional.softmax(inp, dim=1)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    strided_grad = grad.transpose(0, 1).contiguous().transpose(0, 1)

    ref_grad = utils.to_reference(strided_grad, True)
    ref_out = utils.to_reference(saved_out, True)
    # `to_reference` keeps the transposed strides, so the CPU ATen kernel also
    # exercises a non-contiguous upstream gradient.
    assert not ref_grad.is_contiguous()
    ref = torch.ops.aten._softmax_backward_data(ref_grad, ref_out, 1, ref_grad.dtype)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    result = flag_gems.softmax_backward_out(
        strided_grad, saved_out, 1, dtype, grad_input=out
    )
    assert result.data_ptr() == out.data_ptr()
    utils.gems_assert_close(result, ref, dtype, reduce_dim=shape[1], equal_nan=True)
