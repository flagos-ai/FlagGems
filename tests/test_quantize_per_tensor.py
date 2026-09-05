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

# ``quantize_per_tensor`` only accepts float32 tensors on the quantized CUDA
# backend (Half/BFloat16/Double raise "Quantize only works on Float Tensor"),
# so there is no ``FLOAT_DTYPES`` parametrization here. The output is a
# quantized tensor whose ``int_repr`` matches the reference exactly (round to
# nearest, ties to even), hence ``gems_assert_equal`` on the int representation.
QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]
QUANT_SHAPES = (
    [(2, 19, 7)]
    if utils.QUICK_MODE
    else [(), (1,), (1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)]
)
SCALES = [0.1, 0.01, 1.0]
# ``zero_point`` must lie within the representable integer range of *every*
# tested quantized dtype. quint8 covers [0, 255], qint8 covers [-128, 127] and
# qint32 covers the full int32 range, so the common range is [0, 127]. PyTorch
# validates this bound on ``quantize_per_tensor`` and rejects out-of-range values.
ZERO_POINTS = [0, 10, 64]


def _make_input(shape, device="cuda"):
    # Spread values across a wide range so that clamping to the integer range
    # is exercised alongside ordinary in-range values.
    return torch.randn(shape, dtype=torch.float32, device=device) * 3.0


@pytest.mark.quantize_per_tensor
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("zero_point", ZERO_POINTS)
def test_quantize_per_tensor(shape, in_dtype, scale, zero_point):
    # The kernel computes the quantized integer in fp64 when the device supports
    # it (matching the CPU fp64 reference exactly) and falls back to fp32
    # otherwise, which can differ by +/-1 at half-way points. Under --ref=cpu
    # the reference runs on CPU (always fp64), so on a non-fp64 GPU the exact
    # int_repr comparison would be device-dependent -> skip there.
    if utils.TO_CPU and not utils.fp64_is_supported:
        pytest.skip(
            "quantize_per_tensor int_repr is fp64-dependent; non-fp64 GPU vs CPU ref mismatch"
        )
    res_inp = _make_input(shape)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantize_per_tensor(ref_inp, scale, zero_point, in_dtype)
    res_out = flag_gems.quantize_per_tensor(res_inp, scale, zero_point, in_dtype)

    utils.gems_assert_equal(res_out.int_repr(), ref_out.int_repr())
    assert res_out.dtype == in_dtype
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()


@pytest.mark.quantize_per_tensor_out
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("zero_point", ZERO_POINTS)
def test_quantize_per_tensor_out(shape, in_dtype, scale, zero_point):
    # See test_quantize_per_tensor: exact int_repr equality is fp64-dependent.
    if utils.TO_CPU and not utils.fp64_is_supported:
        pytest.skip(
            "quantize_per_tensor int_repr is fp64-dependent; non-fp64 GPU vs CPU ref mismatch"
        )
    res_inp = _make_input(shape)
    ref_inp = utils.to_reference(res_inp)

    # Pre-allocate a quantized `out` buffer with *different* scale/zero_point so
    # that we verify the kernel writes the passed parameters back onto it.
    res_out = torch.quantize_per_tensor(res_inp, 0.5, 100, in_dtype)
    ref_out = torch.quantize_per_tensor(ref_inp, 0.5, 100, in_dtype)

    ref_r = torch.ops.aten.quantize_per_tensor.out(
        ref_inp, scale, zero_point, in_dtype, out=ref_out
    )
    res_r = flag_gems.quantize_per_tensor_out(
        res_inp, scale, zero_point, in_dtype, out=res_out
    )

    assert res_r is res_out
    utils.gems_assert_equal(res_r.int_repr(), ref_r.int_repr())
    assert res_r.q_scale() == ref_r.q_scale()
    assert res_r.q_zero_point() == ref_r.q_zero_point()
