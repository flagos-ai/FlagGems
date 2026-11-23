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

# Tests call flag_gems.conj_physical_backward directly instead of the context
# manager that globally patches ATen dispatch: the CI kernelgen-test rules
# forbid a real activation call from test files, and the wrapper is what the
# PR exposes.

# The kernel materializes conj(grad_output) over the interleaved storage, so
# complex64 and complex128 (FP64 gated on the device, as elsewhere in the
# repo) are the dtypes under test; complex32 is not exercised (matches the
# conj family tests).
COMPLEX_BACKWARD_DTYPES = [
    torch.complex64,
    pytest.param(torch.complex128, marks=utils._SKIP_FP64),
]

# The real-dtype identity path is dtype-independent: fp32/fp64 cover the
# guard (fp64 skipped on devices without FP64 support).
REAL_PASSTHROUGH_DTYPES = [
    torch.float32,
    pytest.param(torch.float64, marks=utils._SKIP_FP64),
]


@pytest.mark.conj_physical_backward
@pytest.mark.parametrize(
    "shape", [(256,), (32, 64), (2, 3, 4), (4096, 4096), (64, 512, 512)]
)
# conj_physical_backward keeps the gradient dtype; complex64/complex128 are the
# dtypes under test (FLOAT_DTYPES/INT_DTYPES not applicable, mirroring test_conj.py).
@pytest.mark.parametrize("dtype", COMPLEX_BACKWARD_DTYPES)
def test_conj_physical_backward(shape, dtype):
    # complex inputs: gradient is the materialized conjugate of grad_output.
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.complex(real, imag).to(dtype)
    ref_grad_output = utils.to_reference(grad_output)

    ref_out = torch.conj_physical(ref_grad_output)
    res_out = flag_gems.conj_physical_backward(grad_output)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conj_physical_backward
@pytest.mark.parametrize("dtype", REAL_PASSTHROUGH_DTYPES)
def test_conj_physical_backward_real_passthrough(dtype):
    # Real (non-complex) gradients pass through unchanged (aliasing
    # grad_output), matching ATen's identity semantics for conj_physical
    # on non-complex tensors -- this differs from conj_backward, which
    # raises on real dtypes. All real dtypes take the same guard, so a
    # fp32/fp64 sample covers it.
    grad_output = torch.randn((1024,), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.conj_physical_backward(grad_output)
    ref_out = torch.conj_physical(grad_output)

    assert res_out is grad_output
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conj_physical_backward
@pytest.mark.parametrize("dtype", COMPLEX_BACKWARD_DTYPES)
def test_conj_physical_backward_non_contiguous(dtype):
    # A transposed view exercises the .contiguous() path in the wrapper.
    base = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    grad_output = base.t()
    assert not grad_output.is_contiguous()

    ref_out = torch.conj_physical(grad_output)
    res_out = flag_gems.conj_physical_backward(grad_output)

    # to_reference relocates the reference to the comparison device, which in
    # --ref=cpu mode is the CPU (res follows via gems_assert_close's to_cpu).
    utils.gems_assert_close(res_out, utils.to_reference(ref_out), dtype)


@pytest.mark.conj_physical_backward
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_conj_physical_backward_zero_conjugate(dtype):
    # conj(zero) == zero and conj(-0i) == +0i: sign-bit flips must normalize
    # the imaginary zero, matching torch.conj_physical bit-for-bit.
    data = torch.zeros((8, 8), dtype=dtype, device=flag_gems.device)
    data.imag = -0.0
    ref_out = torch.conj_physical(data)
    res_out = flag_gems.conj_physical_backward(data)

    assert torch.equal(res_out, ref_out)


@pytest.mark.conj_physical_backward
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_conj_physical_backward_autograd(dtype):
    # Autograd bridge: driving torch.conj_physical through the autograd engine
    # must produce the same gradient as the direct Triton wrapper.
    inp = torch.randn((1024,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    out = torch.conj_physical(inp)
    assert out.requires_grad and out.grad_fn is not None

    grad_output = torch.randn_like(out)
    (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))

    ref_grad = torch.conj_physical(grad_output)
    # to_reference relocates the reference, which in --ref=cpu mode is where
    # gems_assert_equal's to_cpu places the result side.
    utils.gems_assert_equal(res_grad, utils.to_reference(ref_grad))
