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

# torch.quantize_per_tensor_dynamic only supports float32 inputs and the
# quantized dtypes torch.quint8 / torch.qint8 (float16/bfloat16 raise on the
# torch side), so we parametrize over the quantized dtypes rather than
# utils.FLOAT_DTYPES and keep the input dtype fixed at float32.
QUANT_DTYPES = [torch.quint8, torch.qint8]


# Shapes covering scalar, 1d, 2d and higher-rank tensors, plus a few larger
# ones to exercise the two-pass reduction.
QUANT_SHAPES = [(), (1,), (4, 1024), (20, 320, 15), (16, 128, 64, 60)]


REDUCE_RANGE = [False, True]


def _assert_quantized_equal(res, ref):
    """Compare two dynamically-quantized tensors.

    The dynamic quantizer derives ``scale`` and ``zero_point`` from the input
    ``min``/``max`` and then maps each value to its integer bin with
    ``round((x - zp) * scale)``. CPU and GPU compute the reduction in fp32 with
    the same scale/zero_point, but a handful of values sitting exactly on a
    rounding boundary (``.5``) can round to the adjacent bin depending on the
    device's rounding of the last fp32 bit. That is a benign 1-ULP boundary
    effect, not a quantization error: the scale and zero_point must match
    exactly, and the integer representation is compared with ``atol=1``.
    """
    assert res.dtype == ref.dtype, f"dtype mismatch: {res.dtype} vs {ref.dtype}"
    assert (
        res.q_scale() == ref.q_scale()
    ), f"scale mismatch: {res.q_scale()} vs {ref.q_scale()}"
    assert (
        res.q_zero_point() == ref.q_zero_point()
    ), f"zero_point mismatch: {res.q_zero_point()} vs {ref.q_zero_point()}"
    # int_repr is uint8/quint8/qint8; compare in float32 with atol=1 to absorb
    # the rounding-boundary 1-ULP diff between CPU and GPU references.
    utils.gems_assert_close(
        res.int_repr().to(torch.float32),
        ref.int_repr().to(torch.float32),
        torch.float32,
        atol=1,
        equal_nan=True,
    )


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic(shape, dtype, reduce_range):
    res_inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantize_per_tensor_dynamic(ref_inp, dtype, reduce_range)
    # GEMS direct call: the kernel computes scale/zero_point dynamically and
    # builds the quantized tensor on the accelerator.
    res_out = flag_gems.quantize_per_tensor_dynamic(res_inp, dtype, reduce_range)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", QUANT_SHAPES)
def test_quantize_per_tensor_dynamic_all_positive(shape):
    # All-positive values: 0 is folded into the range as the lower bound.
    res_inp = torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 10.0
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantize_per_tensor_dynamic(ref_inp, torch.quint8, False)
    res_out = flag_gems.quantize_per_tensor_dynamic(res_inp, torch.quint8, False)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", QUANT_SHAPES)
def test_quantize_per_tensor_dynamic_all_negative(shape):
    # All-negative values: 0 is folded into the range as the upper bound.
    res_inp = -(torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 10.0)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantize_per_tensor_dynamic(ref_inp, torch.quint8, False)
    res_out = flag_gems.quantize_per_tensor_dynamic(res_inp, torch.quint8, False)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic
@pytest.mark.parametrize("shape", [(), (1,), (4, 1024)])
def test_quantize_per_tensor_dynamic_zeros(shape):
    # Degenerate range (min == max == 0): scale falls back to 0.1, zero_point 0.
    res_inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantize_per_tensor_dynamic(ref_inp, torch.quint8, False)
    res_out = flag_gems.quantize_per_tensor_dynamic(res_inp, torch.quint8, False)

    _assert_quantized_equal(res_out, ref_out)


@pytest.mark.quantize_per_tensor_dynamic_out
@pytest.mark.parametrize("shape", QUANT_SHAPES)
@pytest.mark.parametrize("reduce_range", REDUCE_RANGE)
@pytest.mark.parametrize("dtype", QUANT_DTYPES)
def test_quantize_per_tensor_dynamic_out(shape, dtype, reduce_range):
    res_inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.quantize_per_tensor_dynamic(ref_inp, dtype, reduce_range)

    # Pre-allocate a quantized ``out`` tensor via the public API; the kernel
    # overwrites its integer storage, so the initial content is irrelevant.
    out_tensor = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        ref_out.q_scale(),
        ref_out.q_zero_point(),
        dtype,
    )
    # GEMS direct call: ``quantize_per_tensor_dynamic_out`` writes into ``out``.
    res_r = flag_gems.quantize_per_tensor_dynamic_out(
        res_inp, dtype, reduce_range, out=out_tensor
    )

    assert res_r is out_tensor
    _assert_quantized_equal(res_r, ref_out)
