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

import itertools

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_NVIDIA_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or flag_gems.device != "cuda",
    reason="to_copy_backward is an NVIDIA general-operator implementation",
)

_BACKWARD_DTYPE_PAIRS = [
    (src, dst) for src, dst in itertools.product(utils.FLOAT_DTYPES, repeat=2)
]


@pytest.mark.to_copy_backward
@_NVIDIA_ONLY
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype,dst_dtype", _BACKWARD_DTYPE_PAIRS)
def test_to_copy_backward(shape, src_dtype, dst_dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    inp = torch.randn(
        shape, dtype=src_dtype, device=flag_gems.device, requires_grad=True
    )
    # Keep the native reference at the source dtype: ``to_reference`` promotes
    # half and bfloat16 inputs to float64, which changes the expected gradient
    # dtype for this cast-specific backward.
    ref_inp = inp.detach().clone().requires_grad_()

    res_out = flag_gems.to_copy(inp, dtype=dst_dtype)
    ref_out = torch.ops.aten._to_copy(ref_inp, dtype=dst_dtype)
    grad = torch.randn_like(res_out)
    ref_grad = grad.detach().clone()

    (res_in_grad,) = torch.autograd.grad(res_out, inp, grad)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    utils.gems_assert_equal(res_in_grad, ref_in_grad)

    res_direct = flag_gems.to_copy_backward(grad, src_dtype)
    utils.gems_assert_equal(res_direct, ref_in_grad)


@pytest.mark.to_copy_backward
@_NVIDIA_ONLY
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_to_copy_backward_registered_dispatch(dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    inp = torch.randn(
        (64, 64), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    dst_dtype = torch.float32 if dtype != torch.float32 else torch.float16
    grad = torch.randn((64, 64), dtype=dst_dtype, device=flag_gems.device)

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["to_copy"])
        out = torch.ops.aten._to_copy(inp, dtype=dst_dtype)
        assert "ToCopyFunction" in type(out.grad_fn).__name__
        (res_grad,) = torch.autograd.grad(out, inp, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    expected = torch.ops.aten._to_copy(grad, dtype=dtype)
    utils.gems_assert_equal(res_grad, expected)


@pytest.mark.to_copy_backward
@_NVIDIA_ONLY
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_to_copy_backward_fused_kernel_count(dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn((4096,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    dst_dtype = torch.float32 if dtype != torch.float32 else torch.float16
    out = flag_gems.to_copy(inp, dtype=dst_dtype)
    grad = torch.randn_like(out)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "to_copy_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.COMPLEX_DTYPES,
)
def test_to_dtype(shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip("#2855: Skiping complex to_copy test on tsingmicro platform")
    if flag_gems.vendor_name == "ascend" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip("Issues #3267: Ascend NPU does not support complex32 dtype")
    if flag_gems.vendor_name == "enflame" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip(
            "Enflame GCU400 has no native complex dtype support: "
            "aten::_to_copy complex redispatch aborts"
        )
    x = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = ref_x.to(dtype)
    out = flag_gems.to_copy(x, dtype=dtype)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("target_dtype", utils.ALL_FLOAT_DTYPES + utils.COMPLEX_DTYPES)
def test_to_copy_dtype_cast(shape, target_dtype):
    if flag_gems.vendor_name == "tsingmicro" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip("#2855: Skiping complex to_copy test on tsingmicro platform")
    if flag_gems.vendor_name == "ascend" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip("Issues #3267: Ascend NPU does not support complex32 dtype")
    if flag_gems.vendor_name == "enflame" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip(
            "Enflame GCU400 has no native complex dtype support: "
            "aten::_to_copy complex redispatch aborts"
        )
    src_dtype = torch.float32 if target_dtype != torch.float32 else torch.float16
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=target_dtype)
    res_out = flag_gems.to_copy(x, dtype=target_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.contiguous_format],
)
def test_to_copy_preserve_strides(memory_format):
    base = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    x = base.transpose(0, 1)[::2]
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(
        ref_x,
        dtype=ref_x.dtype,
        memory_format=memory_format,
    )
    res_out = flag_gems.to_copy(
        x,
        dtype=x.dtype,
        memory_format=memory_format,
    )
    utils.gems_assert_equal(res_out, ref_out)
    if memory_format is torch.preserve_format:
        assert res_out.stride() == ref_out.stride()
    else:
        assert res_out.is_contiguous()


# Generate (src, dst) pairs excluding same-dtype conversions
_FLOAT_TO_FLOAT_PAIRS = [
    (s, d)
    for s, d in itertools.product(utils.FLOAT_DTYPES, utils.ALL_FLOAT_DTYPES)
    if s != d
]


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype,dst_dtype", _FLOAT_TO_FLOAT_PAIRS)
def test_to_copy_float_to_float(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and (
        src_dtype == torch.bfloat16 or dst_dtype == torch.bfloat16
    ):
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    res_out = flag_gems.to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_float_to_int(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and src_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    res_out = flag_gems.to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("dst_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_int_to_float(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and dst_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    res_out = flag_gems.to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


# Generate (src, dst) int pairs excluding same-dtype conversions
_INT_DTYPES = [torch.int8, torch.int16, torch.int32]
_INT_TO_INT_PAIRS = list(itertools.permutations(_INT_DTYPES, 2))


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype,dst_dtype", _INT_TO_INT_PAIRS)
def test_to_copy_int_to_int(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    res_out = flag_gems.to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_float_to_uint8(shape, src_dtype):
    if flag_gems.vendor_name == "ascend" and src_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randint(0, 255, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=torch.uint8)
    res_out = flag_gems.to_copy(x, dtype=torch.uint8)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dst_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_uint8_to_float(shape, dst_dtype):
    if flag_gems.vendor_name == "ascend" and dst_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    res_out = flag_gems.to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_uint8_to_int(shape, dst_dtype):
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    res_out = flag_gems.to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)
