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

# Tests call flag_gems.conj_backward directly instead of the flag_gems.use_gems()
# context manager: the CI kernelgen-test rules forbid a real use_gems() call from
# test files, and the wrapper is what the PR exposes.


@pytest.mark.conj_backward
@pytest.mark.parametrize(
    "shape", [(256,), (32, 64), (2, 3, 4), (4096, 4096), (64, 512, 512)]
)
# conj_backward keeps the gradient dtype, so complex64 and complex128 (FP64
# gated on the device, as elsewhere in the repo) are both exercised.
@pytest.mark.parametrize(
    "dtype",
    [
        torch.complex64,
        pytest.param(torch.complex128, marks=utils._SKIP_FP64),
    ],
)
def test_conj_backward(shape, dtype):
    # complex inputs: gradient is the conjugate of grad_output.
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.complex(real, imag).to(dtype)
    ref_grad_output = utils.to_reference(grad_output)

    ref_out = torch.conj(ref_grad_output)
    res_out = flag_gems.conj_backward(grad_output)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conj_backward
# Real-dtype rejection is dtype-independent: one float32 case covers the guard.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conj_backward_real_rejects(dtype):
    # Real (non-complex) gradients raise, matching the _conj entry of the
    # same family: the FlagGems conj family is complex-only.
    grad_output = torch.randn((1024,), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="complex"):
        flag_gems.conj_backward(grad_output)


@pytest.mark.conj_backward
@pytest.mark.parametrize(
    "dtype",
    [
        torch.complex64,
        pytest.param(torch.complex128, marks=utils._SKIP_FP64),
    ],
)
def test_conj_backward_non_contiguous(dtype):
    # A transposed view exercises the .contiguous() path in the wrapper.
    base = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    grad_output = base.t()
    assert not grad_output.is_contiguous()

    ref_out = torch.conj(grad_output)
    res_out = flag_gems.conj_backward(grad_output)

    # to_reference relocates the reference to the comparison device, which in
    # --ref=cpu mode is the CPU (res follows via gems_assert_close's to_cpu).
    utils.gems_assert_close(res_out, utils.to_reference(ref_out), dtype)


@pytest.mark.conj_backward
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_conj_backward_zero_conjugate(dtype):
    # conj(zero) == zero and conj(-0i) == +0i: sign-bit flips must normalize
    # the imaginary zero, matching torch.conj bit-for-bit.
    data = torch.zeros((8, 8), dtype=dtype, device=flag_gems.device)
    data.imag = -0.0
    ref_out = torch.conj(data)
    res_out = flag_gems.conj_backward(data)

    assert torch.equal(res_out, ref_out)


@pytest.mark.conj_backward
@pytest.mark.parametrize("dtype", [torch.complex64])
def test_conj_backward_autograd(dtype):
    # Autograd bridge: driving torch.conj through the autograd engine must
    # produce the same gradient as the direct Triton wrapper.
    inp = torch.randn((1024,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    out = torch.conj(inp)
    assert out.requires_grad and out.grad_fn is not None

    grad_output = torch.randn_like(out)
    (res_grad,) = torch.autograd.grad(out, (inp,), (grad_output,))

    ref_grad = torch.conj(grad_output)
    # to_reference relocates the reference, which in --ref=cpu mode is where
    # gems_assert_equal's to_cpu places the result side.
    utils.gems_assert_equal(res_grad, utils.to_reference(ref_grad))
