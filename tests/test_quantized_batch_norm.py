# Copyright 2026 FlagOS Contributors.
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

# quantized_batch_norm operates on a 4D (NCHW) quantized tensor and produces a
# quantized tensor with caller-supplied output_scale / output_zero_point. The
# reference aten kernel is only available on the CPU quantized backend, so the
# reference path always runs on CPU regardless of the ``--ref`` flag.

# Representative NCHW shapes covering small/medium/large batch and spatial sizes.
QBN_SHAPES = [
    (1, 3, 4, 4),
    (2, 3, 4, 4),
    (4, 8, 16, 16),
    (2, 16, 32, 32),
    (8, 3, 64, 64),
    (1, 64, 128, 128),
]

QBN_QUANT_DTYPES = [torch.quint8, torch.qint8]

# (output_scale, output_zero_point) pairs exercising a range of output
# quantization parameters, including values that clamp to the limits. The high
# zero_point is pinned to 127 (the qint8 maximum) rather than 128 so the same
# parametrization is valid for both quint8 and qint8: the public
# ``torch.quantize_per_tensor`` used to pre-allocate the ``out`` tensor validates
# the zero_point against the dtype range, and 128 is out of range for qint8.
QBN_OUT_PARAMS = [
    (0.1, 0),
    (0.1, 127),
    (0.01, 10),
    (0.25, 3),
]

# (input_scale, input_zero_point) pairs for the input quantized tensor.
QBN_IN_PARAMS = [
    (0.5, 0),
    (0.05, 10),
    (1.0, 3),
]


def _make_quantized_input(shape, scale, zero_point, dtype, device):
    fp = torch.randn(shape, device="cpu")
    return torch.quantize_per_tensor(fp, scale, zero_point, dtype).to(device)


def _qbn_params(C, device):
    weight = torch.randn(C, dtype=torch.float32, device="cpu")
    bias = torch.randn(C, dtype=torch.float32, device="cpu")
    mean = torch.randn(C, dtype=torch.float32, device="cpu")
    var = torch.rand(C, dtype=torch.float32, device="cpu") + 0.5
    return weight, bias, mean, var


def _make_quantized_from_int(int_tensor, scale, zero_point, dtype):
    """Reconstruct a per-tensor quantized tensor that shares ``int_tensor``'s
    exact integer representation, using only the public ``quantize_per_tensor``
    API (the private ``torch._make_per_tensor_quantized_tensor`` is forbidden by
    Rule 8). Dequantize the target integers to their float values, then
    re-quantize: the round-trip is exact for already-quantized integers.
    """
    return torch.quantize_per_tensor(
        (int_tensor.float() - zero_point) * scale, scale, zero_point, dtype
    )


def _assert_quant_equal(res, ref):
    """Compare two quantized tensors by their integer representations.

    quantized_batch_norm produces a fixed-precision quantized tensor whose
    values are an exact function of its inputs (round-to-even + clamp), so we
    compare the integer representation byte-for-byte rather than the
    dequantized float values (which would require tolerance).
    """
    res_int = res.int_repr()
    ref_int = ref.int_repr()
    if res_int.device != ref_int.device:
        res_int = res_int.to(ref_int.device)
    assert (
        res_int.dtype == ref_int.dtype
    ), f"int_repr dtype mismatch: {res_int.dtype} vs {ref_int.dtype}"
    assert (
        res_int.shape == ref_int.shape
    ), f"int_repr shape mismatch: {res_int.shape} vs {ref_int.shape}"
    utils.gems_assert_equal(res_int, ref_int)
    # Quantization parameters should also match.
    assert (
        abs(res.q_scale() - ref.q_scale()) < 1e-6
    ), f"output scale mismatch: {res.q_scale()} vs {ref.q_scale()}"
    assert (
        res.q_zero_point() == ref.q_zero_point()
    ), f"output zero_point mismatch: {res.q_zero_point()} vs {ref.q_zero_point()}"


@pytest.mark.quantized_batch_norm
@pytest.mark.parametrize("shape", QBN_SHAPES)
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
@pytest.mark.parametrize("in_params", QBN_IN_PARAMS)
@pytest.mark.parametrize("out_params", QBN_OUT_PARAMS)
def test_quantized_batch_norm(shape, in_dtype, in_params, out_params):
    in_scale, in_zero_point = in_params
    out_scale, out_zero_point = out_params
    C = shape[1]

    res_qx = _make_quantized_input(
        shape, in_scale, in_zero_point, in_dtype, flag_gems.device
    )
    # Rebuild the reference input to share the integer representation with the
    # GPU input so both paths see identical values.
    res_int = res_qx.int_repr().to("cpu")
    ref_qx = _make_quantized_from_int(res_int, in_scale, in_zero_point, in_dtype)

    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, out_scale, out_zero_point
    )

    res_out = flag_gems.quantized_batch_norm(
        res_qx, weight, bias, mean, var, 1e-5, out_scale, out_zero_point
    )

    _assert_quant_equal(res_out, ref_out)


@pytest.mark.quantized_batch_norm_out
@pytest.mark.parametrize("shape", QBN_SHAPES)
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
@pytest.mark.parametrize("out_params", QBN_OUT_PARAMS)
def test_quantized_batch_norm_out(shape, in_dtype, out_params):
    out_scale, out_zero_point = out_params
    C = shape[1]
    in_scale, in_zero_point = 0.5, 0

    res_qx = _make_quantized_input(
        shape, in_scale, in_zero_point, in_dtype, flag_gems.device
    )
    res_int = res_qx.int_repr().to("cpu")
    ref_qx = _make_quantized_from_int(res_int, in_scale, in_zero_point, in_dtype)

    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    # Pre-allocate the out tensor on GPU with the target output quantization params.
    res_out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        out_scale,
        out_zero_point,
        in_dtype,
    )

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, out_scale, out_zero_point
    )

    flag_gems.quantized_batch_norm_out(
        res_qx,
        weight,
        bias,
        mean,
        var,
        1e-5,
        out_scale,
        out_zero_point,
        out=res_out,
    )

    _assert_quant_equal(res_out, ref_out)
