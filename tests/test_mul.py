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

import importlib
import random
from contextlib import nullcontext

import numpy as np
import pytest
import torch
import triton

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_tensor_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.mul(ref_inp1, inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_mul_scalar_scalar(dtype):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random()))
        inp2 = float(np.float32(random.random()))
    else:
        inp1 = random.randint(0, 100)
        inp2 = random.randint(0, 100)

    ref_out = torch.mul(inp1, inp2)
    res_out = flag_gems.mul(inp1, inp2)

    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_mul_tensor_scalar_int(shape, dtype):
    inp1 = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = 3
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.mul(ref_inp1, inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_mul_tensor_scalar_int_float_scalar(shape, dtype):
    # int tensor x float scalar promotes to the default float dtype
    inp1 = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    inp2 = 0.001
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.mul(ref_inp1, inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", [True, False, 2.0])
def test_mul_tensor_scalar_bool(shape, scalar):
    inp1 = torch.randint(0, 2, shape, dtype=torch.bool, device="cpu").to(
        flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1, False)

    ref_out = torch.mul(ref_inp1, scalar)
    res_out = flag_gems.mul(inp1, scalar)

    if isinstance(scalar, bool):
        utils.gems_assert_equal(res_out, ref_out)
    else:
        # bool tensor x float scalar promotes to float32
        utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.mul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
def test_mul_scalar_scalar_float(shape):
    inp1 = float(np.float32(random.random()))
    inp2 = float(np.float32(random.random()))

    ref_out = torch.mul(inp1, inp2)
    res_out = flag_gems.mul(inp1, inp2)

    # scalar x scalar falls back to torch.tensor(A * B), which yields float32
    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.mul_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_tensor_tensor_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.mul_(ref_inp2)
    res_out = flag_gems.mul_(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_tensor_scalar_(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), True)

    ref_out = ref_inp1.mul_(inp2)
    res_out = flag_gems.mul_(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((10, 1), (1, 5)),
        ((1, 5), (10, 1)),
        ((1048576, 1), (1, 32)),
        ((4096, 32, 128), (1, 1, 128)),
        ((3, 1, 5), (1, 4, 1)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_broadcast_shape(shape_a, shape_b, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    assert res_out.shape == ref_out.shape, (
        f"Shape mismatch: FlagGems produced {res_out.shape}, "
        f"expected {ref_out.shape}"
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_mul_complex_complex(shape, complex_dtype):
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    # inp1: complex tensor
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    if flag_gems.vendor_name == "cambricon" and complex_dtype == torch.complex32:
        from .accuracy_utils import to_cpu

        res_out = to_cpu(res_out, ref_out)
        ref_out = ref_out.to(complex_dtype)
        torch.testing.assert_close(res_out, ref_out, atol=5e-3, rtol=2e-3)
    else:
        utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_mul_complex_float_tensor(shape, complex_dtype):
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    # inp1: complex tensor
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    if complex_dtype == torch.complex64:
        float_dtype = torch.float32
    elif complex_dtype == torch.complex32:
        float_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported complex_dtype: {complex_dtype}")

    inp2 = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)

    # mthreads torch.mul unsupport complex x float.
    if flag_gems.vendor_name == "mthreads":
        ref_inp1 = inp1.to("cpu")
        ref_inp2 = inp2.to("cpu")
    else:
        ref_inp1 = utils.to_reference(inp1, True)
        ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    if flag_gems.vendor_name == "mthreads":
        res_out = res_out.to("cpu")
        ref_out = ref_out.to(complex_dtype)
    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_mul_complex_int_tensor(shape, complex_dtype):
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    # inp1: complex tensor
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = torch.randint(10, 20, shape, device=flag_gems.device)

    # mthreads torch.mul unsupport complex x int.
    if flag_gems.vendor_name == "mthreads":
        ref_inp1 = inp1.to("cpu")
        ref_inp2 = inp2.to("cpu")
    else:
        ref_inp1 = utils.to_reference(inp1, True)
        ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    if flag_gems.vendor_name == "mthreads":
        res_out = res_out.to("cpu")
        ref_out = ref_out.to(complex_dtype)
    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
def test_mul_complex_int_scalar(shape, complex_dtype):
    if flag_gems.vendor_name == "mthreads" and complex_dtype == torch.complex32:
        pytest.skip("mthreads does not support complex32 dtype")
    # inp1: complex tensor
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    inp2 = 3

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = inp2

    ref_out = torch.mul(ref_inp1, ref_inp2)
    res_out = flag_gems.mul(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, complex_dtype)


# Ascend coreDim and large-tensor regressions.
@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "n_elements,description",
    [
        # Small tensor - baseline test
        (10240, "Small tensor (baseline)"),
        # Just below threshold: 65535 * 1024 - 1 = 67,107,839
        (67107839, "Just below coreDim threshold"),
        # Just above threshold: 65536 * 1024 = 67,108,864
        (67108864, "Just above coreDim threshold"),
        # From actual training error logs
        (141050 * 1024, "From device 0-3 error log (144,435,200)"),
        (169351 * 1024, "From device 8-11 error log (173,415,424)"),
        (310400 * 1024, "From device 4-7 error log (317,849,600)"),
        # Very large tensor - stress test
        # Note: This creates a ~4GB tensor (1G * 4 bytes), ensure device has enough memory
        pytest.param(
            1024 * 1024 * 1024,
            "Very large tensor (1G elements, 4GB)",
        ),
    ],
)
@pytest.mark.parametrize("scalar", [2.5, -1.5, 0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mul_large_tensor_scalar(n_elements, description, scalar, dtype):
    """Test mul operation with large tensors that trigger coreDim > 65535."""
    if n_elements == 1024 * 1024 * 1024:
        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            pytest.skip("Requires an Ascend NPU with at least 8GB device memory")
        if npu.get_device_properties(0).total_memory < 8 * 1024**3:
            pytest.skip("Requires at least 8GB device memory")

    # Create large tensor
    inp = torch.randn(n_elements, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    # Compute reference result
    ref_out = torch.mul(ref_inp, scalar)

    # Compute FlagGems result
    # This should not raise "coreDim can't be greater than UINT16_MAX" error
    res_out = flag_gems.mul(inp, scalar)

    # Verify correctness
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "n_elements",
    [
        67108864,  # 65536 * 1024
        141050 * 1024,
        169351 * 1024,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mul_large_tensor_scalar_dtypes(n_elements, dtype):
    """Test mul operation with large tensors across different dtypes."""
    if dtype == torch.bfloat16 and not flag_gems.runtime.device.support_bf16:
        pytest.skip("bfloat16 not supported on this device")

    inp = torch.randn(n_elements, dtype=dtype, device=flag_gems.device)
    scalar = 2.5
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mul(ref_inp, scalar)
    res_out = flag_gems.mul(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mul_
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="This test is specific to Ascend NPU coreDim limit",
)
@pytest.mark.parametrize(
    "n_elements",
    [
        67108864,  # 65536 * 1024
        173415424,  # 169351 * 1024
    ],
)
@pytest.mark.parametrize("scalar", [2.0, -3.5])
def test_mul_large_tensor_scalar_inplace(n_elements, scalar):
    """Test inplace mul operation with large tensors."""
    inp = torch.randn(n_elements, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.mul_(scalar)
    res_out = flag_gems.mul_(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend complex launch grid"
)
@pytest.mark.parametrize("runtime_meta", [False, True])
@pytest.mark.parametrize("n_elements", [2051, 65535 * 1024 + 1])
def test_mul_complex_grid_meta_review(monkeypatch, runtime_meta, n_elements):
    mul_ops = importlib.import_module(flag_gems.mul.__module__)
    assert "_ascend" in mul_ops.__name__
    monkeypatch.setattr(
        mul_ops, "_needs_runtime_meta_for_constexpr_tuple", lambda: runtime_meta
    )
    monkeypatch.setattr(mul_ops.torch_device_fn, "device", lambda device: nullcontext())
    launches = []

    class CaptureGrid:
        def __getitem__(self, grid):
            assert callable(grid)
            for block in (256, 512, 1024):
                expected = min(triton.cdiv(n_elements, block), 65535)
                assert grid({"BLOCK_SIZE": block}) == (expected,)

            def launch(*args, **kwargs):
                launches.append(True)

            return launch

    kernel_name = (
        "mul_complex_generic_nd_runtime_meta_kernel"
        if runtime_meta
        else "mul_complex_generic_nd_kernel"
    )
    monkeypatch.setattr(mul_ops, kernel_name, CaptureGrid())
    # Meta tensors test the real launcher above the cap without allocating GBs.
    output = torch.empty(n_elements, device="meta", dtype=torch.complex64)
    part = torch.empty(n_elements, device="meta")
    result = mul_ops._launch_complex_generic(
        part,
        part,
        part,
        part,
        output,
        (n_elements,),
        [(1,)] * 4,
        (2,),
        (2,),
        torch.complex64,
    )
    assert result is output
    assert launches == [True]


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend complex launch grid"
)
@pytest.mark.parametrize("block_size", [256, 512])
@pytest.mark.parametrize("runtime_meta", [False, True])
@pytest.mark.parametrize("n_elements", [2051, 262147])
def test_mul_complex_grid_stride_review(
    monkeypatch, block_size, runtime_meta, n_elements
):
    mul_ops = importlib.import_module(flag_gems.mul.__module__)
    monkeypatch.setattr(
        mul_ops, "_needs_runtime_meta_for_constexpr_tuple", lambda: runtime_meta
    )
    kernel_name = (
        "mul_complex_generic_nd_runtime_meta_kernel"
        if runtime_meta
        else "mul_complex_generic_nd_kernel"
    )
    kernel = getattr(mul_ops, kernel_name)
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    launches = []

    class FixedTileKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                resolved = grid({"BLOCK_SIZE": block_size})
                assert resolved == (min(triton.cdiv(n_elements, block_size), 65535),)
                # Exercise multiple grid strides and the masked last tile with
                # both implementations, without requiring huge allocations.
                launches.append(resolved)
                return kernel[(3,)](*args, **kwargs, BLOCK_SIZE=block_size)

            return launch

    monkeypatch.setattr(mul_ops, kernel_name, FixedTileKernel())
    generator = torch.Generator().manual_seed(20261117)
    a_cpu = torch.randn((n_elements,), dtype=torch.complex64, generator=generator)
    b_cpu = torch.tensor([1.5 - 0.5j], dtype=torch.complex64)
    expected = a_cpu * b_cpu
    result = flag_gems.mul(a_cpu.to(flag_gems.device), b_cpu.to(flag_gems.device))
    torch.npu.synchronize()
    assert launches
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or triton.__version__ < "3.5",
    reason="Hopper tiled broadcast kernel requires triton >= 3.5",
)
@pytest.mark.parametrize("n_cols", [7, 96])
def test_mul_hopper_broadcast_survives_monkeypatched_next_power_of_2(
    monkeypatch, n_cols
):
    """The tiled broadcast kernel must not depend on a mutable triton global.

    Some downstream packages replace ``triton.next_power_of_2`` with an
    int-only implementation. The kernel evaluates next_power_of_2 on a
    ``tl.constexpr``, so a helper that only accepts ``int`` makes compilation
    fail. Here we install an int-only version (the one sglang uses) and assert
    the broadcast path still compiles and computes the right result.
    """
    hopper_mul = importlib.import_module(
        "flag_gems.runtime.backend._nvidia.hopper.ops.mul"
    )

    def _int_only_next_power_of_2(n):
        return 1 << (n - 1).bit_length() if n > 0 else 1

    monkeypatch.setattr(triton, "next_power_of_2", _int_only_next_power_of_2)

    rows = 262144
    inp1 = torch.randn((rows, 1), dtype=torch.float32, device=flag_gems.device)
    inp2 = torch.randn((1, n_cols), dtype=torch.float32, device=flag_gems.device)
    ref_out = utils.to_reference(inp1, True) * utils.to_reference(inp2, True)
    res_out = flag_gems.mul(inp1, inp2)

    assert res_out.shape == ref_out.shape, (
        f"Shape mismatch: FlagGems produced {res_out.shape}, "
        f"expected {ref_out.shape}"
    )
    utils.gems_assert_close(res_out, ref_out, torch.float32)
    # Guard against the fix silently losing the helper (e.g. helper removed but
    # call sites reverted): the module must own its constexpr-safe version.
    assert hasattr(hopper_mul, "_next_power_of_2")


_MUL_BACKWARD_NVIDIA_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia" or flag_gems.device != "cuda",
    reason="mul_backward is an NVIDIA general-operator implementation",
)


def _run_mul_backward_case(input, other):
    res_input = input.detach().requires_grad_()
    res_other = other.detach().requires_grad_()
    ref_input = utils.to_reference(input.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    res_output = flag_gems.mul(res_input, res_other)
    ref_output = torch.mul(ref_input, ref_other)
    grad = torch.randn_like(res_output)
    res_grads = torch.autograd.grad(res_output, (res_input, res_other), grad)
    ref_grads = torch.autograd.grad(
        ref_output, (ref_input, ref_other), utils.to_reference(grad)
    )
    return res_input, res_other, grad, res_grads, ref_grads


@pytest.mark.mul_backward
@_MUL_BACKWARD_NVIDIA_ONLY
@pytest.mark.parametrize("shape", [(64,), (8, 16), (2, 3, 4)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_backward_basic(shape, dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    _, _, grad, res_grads, ref_grads = _run_mul_backward_case(input, other)
    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=2)

    direct = flag_gems.mul_backward(grad, input, other)
    for result, reference in zip(direct, ref_grads):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=2)


@pytest.mark.mul_backward
@_MUL_BACKWARD_NVIDIA_ONLY
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "input_shape,other_shape",
    [
        ((8, 7, 5), (5,)),
        ((8, 7, 5), (1, 7, 1)),
        ((1, 7, 5), (8, 1, 5)),
        ((), (8, 7)),
        ((8, 7), ()),
    ],
)
def test_mul_backward_broadcast(input_shape, other_shape, dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    input = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    _, _, _, res_grads, ref_grads = _run_mul_backward_case(input, other)
    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=8)


@pytest.mark.mul_backward
@_MUL_BACKWARD_NVIDIA_ONLY
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_backward_noncontiguous_and_selective(dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    input = torch.randn((8, 16), dtype=dtype, device=flag_gems.device).T
    other = torch.randn((8, 16), dtype=dtype, device=flag_gems.device).T
    res_input, res_other, grad, res_grads, ref_grads = _run_mul_backward_case(
        input, other
    )
    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=2)

    direct = flag_gems.mul_backward(grad, res_input, res_other, (True, False))
    assert direct[1] is None
    utils.gems_assert_close(direct[0], ref_grads[0], dtype, reduce_dim=2)


@pytest.mark.mul_backward
@_MUL_BACKWARD_NVIDIA_ONLY
@pytest.mark.parametrize(
    "input_dtype,other_dtype",
    [(torch.float16, torch.float32), (torch.float32, torch.float16)],
)
def test_mul_backward_mixed_dtype(input_dtype, other_dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    input = torch.randn((64, 64), dtype=input_dtype, device=flag_gems.device)
    other = torch.randn((64, 64), dtype=other_dtype, device=flag_gems.device)
    _, _, _, res_grads, ref_grads = _run_mul_backward_case(input, other)
    utils.gems_assert_close(res_grads[0], ref_grads[0], input_dtype, reduce_dim=2)
    utils.gems_assert_close(res_grads[1], ref_grads[1], other_dtype, reduce_dim=2)


@pytest.mark.mul_backward
@_MUL_BACKWARD_NVIDIA_ONLY
@pytest.mark.parametrize(
    "input_shape,other_shape", [((32, 32), (32, 32)), ((8, 1), (1, 7))]
)
def test_mul_backward_complex(input_shape, other_shape):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    input = torch.randn(input_shape, dtype=torch.complex64, device=flag_gems.device)
    other = torch.randn(other_shape, dtype=torch.complex64, device=flag_gems.device)
    _, _, _, res_grads, ref_grads = _run_mul_backward_case(input, other)
    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, torch.complex64, reduce_dim=2)


@pytest.mark.mul_backward
@_MUL_BACKWARD_NVIDIA_ONLY
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_backward_registered_dispatch(dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    input = torch.randn(
        (32, 32), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    other = torch.randn_like(input, requires_grad=True)
    grad = torch.randn_like(input)
    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["mul"])
        output = torch.mul(input, other)
        assert "MulFunction" in type(output.grad_fn).__name__
        grads = torch.autograd.grad(output, (input, other), grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(grads[0], grad * other, dtype, reduce_dim=2)
    utils.gems_assert_close(grads[1], grad * input, dtype, reduce_dim=2)


@pytest.mark.mul_backward
@_MUL_BACKWARD_NVIDIA_ONLY
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mul_backward_fused_kernel_count(dtype):
    if utils.TO_CPU:
        pytest.skip("Triton backward suite; skipped under --ref=cpu")
    from torch.profiler import ProfilerActivity, profile

    input = torch.randn(
        (4096,), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    other = torch.randn_like(input, requires_grad=True)
    output = flag_gems.mul(input, other)
    grad = torch.randn_like(output)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(output, (input, other), grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "mul_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches
